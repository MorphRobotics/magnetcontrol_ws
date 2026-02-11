#!/usr/bin/env python3
"""
UR5e Airway Navigation Controller — ROS 2 Node
================================================
Closed-loop MSCR navigation through the bronchial tree using:
  - PRM roadmap for path planning (Dijkstra)
  - ONNX inverse Cosserat model for magnetic field inference
  - JointTrajectoryController for commanding the UR5e in Gazebo
  - TF2 for real-time TCP tracking
  - RViz markers for MSCR, path, and airway visualization

Works with the Gazebo simulation launched by ur5e_airway_sim.launch.py
or with a real UR5e via ros2_control.
"""

import os
import numpy as np
import scipy.io as sio
import onnxruntime as ort
import heapq

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Point, Vector3, Pose
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as DurationMsg

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
L_MSCR       = 60.0        # mm
ALPHA_SMOOTH = 0.15
B_MIN, B_MAX = 0.0005, 0.012
NAV_SPEED    = 2.0          # mm per step
MM_TO_M      = 1e-3
CONTROL_HZ   = 10.0


class UR5eAirwayNavigation(Node):
    """ROS 2 node for MSCR navigation through bronchial airways."""

    def __init__(self):
        super().__init__('ur5e_airway_navigation')

        # Parameters
        self.declare_parameter('prm_path', '')
        self.declare_parameter('obstacle_path', '')
        self.declare_parameter('stl_path', '')
        self.declare_parameter('auto_navigate', False)

        prm_path = self.get_parameter('prm_path').get_parameter_value().string_value
        obstacle_path = self.get_parameter('obstacle_path').get_parameter_value().string_value
        self.auto_navigate = self.get_parameter('auto_navigate').get_parameter_value().bool_value

        # Load NN model
        self.get_logger().info('Loading ONNX inverse Cosserat model...')
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(pkg_dir, 'mscr_inverse_model.onnx')
        norm_path = os.path.join(pkg_dir, 'inv_norm3.mat')
        self.ort_sess = ort.InferenceSession(onnx_path)
        mat = sio.loadmat(norm_path)
        norm = mat['invNorm'][0, 0]
        self.mu_in   = norm['mu_in'].flatten().astype(np.float32)
        self.sig_in  = norm['sig_in'].flatten().astype(np.float32)
        self.mu_out  = norm['mu_out'].flatten().astype(np.float32)
        self.sig_out = norm['sig_out'].flatten().astype(np.float32)

        # Load PRM roadmap
        self.get_logger().info('Loading PRM roadmap...')
        prm = sio.loadmat(prm_path)
        self.nodes = prm['nodesFiltered'].astype(np.float64)
        self.edges = prm['edgesFiltered'].astype(np.int32) - 1

        N = len(self.nodes)
        self.adj = [[] for _ in range(N)]
        for e in self.edges:
            d = np.linalg.norm(self.nodes[e[0]] - self.nodes[e[1]])
            self.adj[e[0]].append((e[1], d))
            self.adj[e[1]].append((e[0], d))

        # State
        z_vals = self.nodes[:, 2]
        self.start_node = int(np.argmax(z_vals))
        self.current_pos = self.nodes[self.start_node].copy()
        self.mscr_base = self.current_pos.copy()
        self.mscr_base[2] += L_MSCR
        self.magnet_pos_mm = self.current_pos + np.array([80.0, 0, 0])
        self.prev_magnet = self.magnet_pos_mm.copy()

        self.path_points = np.empty((0, 3))
        self.path_cursor = 0
        self.navigating = False
        self.target_node = None
        self.traversed_points = []

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Joint state tracking
        self.current_joint_state = None
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.create_subscription(JointState, '/joint_states',
                                  self._joint_state_cb, 10)

        # Action client for joint trajectory controller
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/mscr_visualization', 10)
        self.target_sub = self.create_subscription(
            Vector3, '/mscr/navigation_target', self._target_cb, 10
        )

        # Control timer
        self.create_timer(1.0 / CONTROL_HZ, self._control_loop)

        # Marker timer (slower rate)
        self.create_timer(0.2, self._publish_markers)

        self.get_logger().info('UR5e Airway Navigation node initialized.')
        self.get_logger().info(f'  PRM nodes: {len(self.nodes)}, edges: {len(self.edges)}')
        self.get_logger().info(f'  Start node: {self.start_node}')

        if self.auto_navigate:
            self.get_logger().info('Auto-navigation enabled. '
                                    'Will navigate to deepest airway node.')
            # Find a deep target (lowest Z = deepest bronchus)
            z_min_idx = int(np.argmin(self.nodes[:, 2]))
            self._plan_path(z_min_idx)

    def _joint_state_cb(self, msg):
        self.current_joint_state = msg

    def _target_cb(self, msg):
        """Handle navigation target from external topic."""
        target_3d = np.array([msg.x, msg.y, msg.z])
        target_idx = self._find_nearest_node(target_3d)
        self._plan_path(target_idx)

    def _plan_path(self, target_idx):
        src_idx = self._find_nearest_node(self.current_pos)
        if target_idx == src_idx:
            return

        self.get_logger().info(f'Planning path: node {src_idx} -> {target_idx}')
        path = self._dijkstra(src_idx, target_idx)
        if not path:
            self.get_logger().warn('No path found!')
            return

        raw_pts = self.nodes[path]
        self.path_points = self._interpolate_path(raw_pts, step_mm=NAV_SPEED)
        self.path_cursor = 0
        self.navigating = True
        self.target_node = target_idx
        self.traversed_points = []
        self.get_logger().info(
            f'Path: {len(path)} PRM nodes, {len(self.path_points)} waypoints'
        )

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        if not self.navigating:
            return
        if self.path_cursor >= len(self.path_points):
            self.navigating = False
            self.get_logger().info('Target reached!')
            return

        # Advance MSCR tip along path
        self.current_pos = self.path_points[self.path_cursor].copy()
        self.path_cursor += 1
        self.traversed_points.append(self.current_pos.copy())

        # Compute MSCR base and tangent
        if self.path_cursor >= 2:
            tangent = self.current_pos - self.path_points[max(0, self.path_cursor - 2)]
            t_norm = np.linalg.norm(tangent)
            tangent = tangent / t_norm if t_norm > 1e-6 else np.array([0, 0, 1.0])
        else:
            tangent = np.array([0, 0, 1.0])
        self.mscr_base = self.current_pos - tangent * L_MSCR

        # Tip deflection
        straight_tip = self.mscr_base + tangent * L_MSCR
        deflection_mm = self.current_pos - straight_tip
        if np.linalg.norm(deflection_mm) < 0.01:
            deflection_mm = np.array([0.1, 0.1, 0.05])

        # NN inference
        Bmag, az, el = self._nn_inference(deflection_mm)

        # Compute magnet position (mm)
        self.magnet_pos_mm = self._compute_magnet_pos(Bmag, az, el, self.mscr_base)

        # Convert to meters and send to UR5e
        magnet_m = self.magnet_pos_mm * MM_TO_M
        self._send_trajectory_command(magnet_m)

    def _send_trajectory_command(self, target_pos_m):
        """
        Send a position command to the UR5e joint trajectory controller.
        Uses simple IK to compute joint angles from target TCP position.
        """
        # For Gazebo integration, compute IK and send trajectory
        # Using the numerical IK from the standalone simulation
        if self.current_joint_state is None:
            return

        q_current = np.zeros(6)
        for i, name in enumerate(self.joint_names):
            if name in self.current_joint_state.name:
                idx = self.current_joint_state.name.index(name)
                q_current[i] = self.current_joint_state.position[idx]

        q_target = self._numerical_ik(target_pos_m, q_current)
        if q_target is None:
            return

        # Create trajectory message
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = q_target.tolist()
        point.time_from_start = DurationMsg(sec=0, nanosec=int(0.1 * 1e9))
        traj.points = [point]

        # Send via action client
        if not self.trajectory_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().warn('Trajectory action server not available')
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self.trajectory_client.send_goal_async(goal)

    def _numerical_ik(self, target_pos, q_init, max_iter=30, tol=1e-4):
        """Damped least-squares IK (position only)."""
        # UR5e DH parameters
        DH = np.array([
            [0,          0,       0.1625,  0],
            [-np.pi/2,   0,       0,       0],
            [0,         -0.4250,  0,       0],
            [0,         -0.3922,  0,       0],
            [np.pi/2,    0,       0.1333,  0],
            [-np.pi/2,   0,       0.0997,  0],
        ])
        TOOL_D = 0.0996

        def fk_pos(q):
            T = np.eye(4)
            for i in range(6):
                alpha, a, d, off = DH[i]
                theta = q[i] + off
                ct, st = np.cos(theta), np.sin(theta)
                ca, sa = np.cos(alpha), np.sin(alpha)
                T = T @ np.array([
                    [ct, -st, 0, a],
                    [st*ca, ct*ca, -sa, -sa*d],
                    [st*sa, ct*sa, ca, ca*d],
                    [0, 0, 0, 1],
                ])
            T = T @ np.array([
                [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, TOOL_D], [0, 0, 0, 1]
            ])
            return T[:3, 3]

        q = q_init.copy()
        damping = 0.01
        for _ in range(max_iter):
            pos = fk_pos(q)
            err = target_pos - pos
            if np.linalg.norm(err) < tol:
                return q
            J = np.zeros((3, 6))
            eps = 1e-6
            for j in range(6):
                qp = q.copy()
                qp[j] += eps
                J[:, j] = (fk_pos(qp) - pos) / eps
            dq = np.linalg.solve(J.T @ J + damping * np.eye(6), J.T @ err)
            scale = min(1.0, 0.3 / (np.max(np.abs(dq)) + 1e-10))
            q += scale * dq
        return q

    # ── RViz Markers ──────────────────────────────────────────────────────────

    def _publish_markers(self):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # MSCR body (line strip)
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = stamp
        m.ns = 'mscr'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.003  # line width in meters
        m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        base_m = self.mscr_base * MM_TO_M
        tip_m = self.current_pos * MM_TO_M
        m.points = [
            Point(x=float(base_m[0]), y=float(base_m[1]), z=float(base_m[2])),
            Point(x=float(tip_m[0]), y=float(tip_m[1]), z=float(tip_m[2])),
        ]
        markers.markers.append(m)

        # MSCR tip sphere
        m2 = Marker()
        m2.header.frame_id = 'world'
        m2.header.stamp = stamp
        m2.ns = 'mscr'
        m2.id = 1
        m2.type = Marker.SPHERE
        m2.action = Marker.ADD
        m2.pose.position = Point(x=float(tip_m[0]), y=float(tip_m[1]),
                                  z=float(tip_m[2]))
        m2.scale = Vector3(x=0.004, y=0.004, z=0.004)
        m2.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
        markers.markers.append(m2)

        # Planned path
        if len(self.path_points) > 1:
            m3 = Marker()
            m3.header.frame_id = 'world'
            m3.header.stamp = stamp
            m3.ns = 'path'
            m3.id = 0
            m3.type = Marker.LINE_STRIP
            m3.action = Marker.ADD
            m3.scale.x = 0.001
            m3.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.7)
            for pt in self.path_points[::2]:  # subsample for performance
                pt_m = pt * MM_TO_M
                m3.points.append(Point(x=float(pt_m[0]), y=float(pt_m[1]),
                                        z=float(pt_m[2])))
            markers.markers.append(m3)

        # Traversed path
        if len(self.traversed_points) > 1:
            m4 = Marker()
            m4.header.frame_id = 'world'
            m4.header.stamp = stamp
            m4.ns = 'traversed'
            m4.id = 0
            m4.type = Marker.LINE_STRIP
            m4.action = Marker.ADD
            m4.scale.x = 0.002
            m4.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
            for pt in self.traversed_points[::2]:
                pt_m = pt * MM_TO_M
                m4.points.append(Point(x=float(pt_m[0]), y=float(pt_m[1]),
                                        z=float(pt_m[2])))
            markers.markers.append(m4)

        # Magnetic field line (magnet -> tip)
        m5 = Marker()
        m5.header.frame_id = 'world'
        m5.header.stamp = stamp
        m5.ns = 'mag_field'
        m5.id = 0
        m5.type = Marker.LINE_STRIP
        m5.action = Marker.ADD
        m5.scale.x = 0.001
        m5.color = ColorRGBA(r=0.5, g=0.0, b=0.5, a=0.5)
        mag_m = self.magnet_pos_mm * MM_TO_M
        m5.points = [
            Point(x=float(mag_m[0]), y=float(mag_m[1]), z=float(mag_m[2])),
            Point(x=float(tip_m[0]), y=float(tip_m[1]), z=float(tip_m[2])),
        ]
        markers.markers.append(m5)

        self.marker_pub.publish(markers)

    # ── PRM / NN helpers ──────────────────────────────────────────────────────

    def _dijkstra(self, src, dst):
        N = len(self.nodes)
        dist = np.full(N, np.inf)
        prev = np.full(N, -1, dtype=int)
        dist[src] = 0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[dst] == np.inf:
            return []
        path = []
        c = dst
        while c != -1:
            path.append(c)
            c = prev[c]
        return path[::-1]

    def _find_nearest_node(self, point_3d):
        return int(np.argmin(np.linalg.norm(self.nodes - point_3d, axis=1)))

    def _interpolate_path(self, points, step_mm=2.0):
        if len(points) < 2:
            return points
        interp = [points[0]]
        residual = 0.0
        for i in range(1, len(points)):
            seg = points[i] - points[i-1]
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-9:
                continue
            direction = seg / seg_len
            travelled = residual
            while travelled < seg_len:
                interp.append(points[i-1] + direction * travelled)
                travelled += step_mm
            residual = travelled - seg_len
        interp.append(points[-1])
        return np.array(interp)

    def _nn_inference(self, tip_deflection_mm):
        dptip = (tip_deflection_mm * MM_TO_M).astype(np.float32)
        x_norm = ((dptip - self.mu_in) / self.sig_in).reshape(1, -1)
        y_norm = self.ort_sess.run(None, {'input': x_norm})[0].flatten()
        y = y_norm * self.sig_out + self.mu_out
        return float(y[0]), float(y[1]), float(y[2])

    def _compute_magnet_pos(self, Bmag, az, el, mscr_base_mm):
        R_mm = 40.0 + (1.0 - (np.clip(Bmag, B_MIN, B_MAX) / B_MAX)) * 60.0
        offset = np.array([
            R_mm * np.cos(el) * np.cos(az),
            R_mm * np.cos(el) * np.sin(az),
            R_mm * np.sin(el)
        ])
        pm_raw = mscr_base_mm + offset
        pm = ALPHA_SMOOTH * pm_raw + (1.0 - ALPHA_SMOOTH) * self.prev_magnet
        self.prev_magnet = pm.copy()
        return pm


def main(args=None):
    rclpy.init(args=args)
    node = UR5eAirwayNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
