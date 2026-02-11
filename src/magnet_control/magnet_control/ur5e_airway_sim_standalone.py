#!/usr/bin/env python3
"""
UR5e + MSCR Bronchial Airway Navigation — Standalone Physics Simulation
========================================================================
Full-pipeline simulation that runs without ROS 2 or Gazebo. Uses:
  - UR5e forward kinematics (DH parameters) for arm positioning
  - ONNX inverse Cosserat model for MSCR magnetic field mapping
  - PRM roadmap + Dijkstra for path planning in bronchial tree
  - Bronchial STL mesh for 3D visualization and collision context
  - Simulated physics: joint dynamics, gravity, magnetic force model

Interactive: click on PRM nodes to set navigation targets.
The UR5e arm moves the permanent magnet while the MSCR tip follows the
planned path through the bronchial tree.
"""

import sys
import os
import numpy as np
import scipy.io as sio
import onnxruntime as ort
from stl import mesh as stl_mesh
from collections import deque
import heapq
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import csv

# ──────────────────────────────────────────────────────────────────────────────
# IEEE PLOT STYLE
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':          10,
    'axes.labelsize':     10,
    'axes.titlesize':     11,
    'axes.labelweight':   'bold',
    'axes.titleweight':   'bold',
    'axes.linewidth':     1.2,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.major.width':  1.0,
    'ytick.major.width':  1.0,
    'legend.fontsize':    8,
    'legend.frameon':     False,
    'figure.dpi':         100,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.grid':          False,
    'mathtext.fontset':   'stix',
})
import time

# ──────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STL_PATH       = os.path.join(BASE_DIR, 'Bronchial tree anatomy-1mm-shell.STL')
PRM_PATH       = os.path.join(BASE_DIR, 'prm_roadmap.mat')
OBSTACLE_PATH  = os.path.join(BASE_DIR, 'bronchial_obstacle_map.mat')
ONNX_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mscr_inverse_model.onnx')
NORM_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inv_norm3.mat')

# ──────────────────────────────────────────────────────────────────────────────
# UR5e DH PARAMETERS (modified DH convention, meters)
# ──────────────────────────────────────────────────────────────────────────────
# [alpha, a, d, theta_offset]
UR5E_DH = np.array([
    [0,          0,       0.1625,  0],
    [-np.pi/2,   0,       0,       0],
    [0,         -0.4250,  0,       0],
    [0,         -0.3922,  0,       0],
    [np.pi/2,    0,       0.1333,  0],
    [-np.pi/2,   0,       0.0997,  0],
])
UR5E_TOOL_D = 0.0996  # wrist3 to tool0

# Joint limits (rad)
UR5E_JOINT_LIMITS = np.array([
    [-2*np.pi, 2*np.pi],
    [-2*np.pi, 2*np.pi],
    [-np.pi,   np.pi],
    [-2*np.pi, 2*np.pi],
    [-2*np.pi, 2*np.pi],
    [-2*np.pi, 2*np.pi],
])

# Maximum joint velocity (rad/s) and acceleration (rad/s^2)
UR5E_MAX_VEL = 3.14
UR5E_MAX_ACC = 5.0

# ──────────────────────────────────────────────────────────────────────────────
# MSCR / PHYSICS CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
L_MSCR         = 60.0          # mm
MAGNET_OFFSET  = 80.0          # mm
ALPHA_SMOOTH   = 0.15
B_MIN, B_MAX   = 0.0005, 0.012
NAV_SPEED      = 2.0           # mm per step
MM_TO_M        = 1e-3
SIM_DT         = 0.03          # simulation timestep (s)

# UR5e base position relative to airway (meters)
# Positioned so the arm can reach the bronchial phantom workspace
UR5E_BASE_POS = np.array([0.0, 0.0, 0.0])  # world frame (m)

# MSCR catheter entry point relative to UR5e base (meters)
CATHETER_BASE_M = np.array([-0.14397, -0.43562, -0.24807])


# ══════════════════════════════════════════════════════════════════════════════
# UR5e KINEMATICS
# ══════════════════════════════════════════════════════════════════════════════

def dh_transform(alpha, a, d, theta):
    """Compute 4x4 homogeneous transform from DH parameters."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,     -st,     0,    a],
        [st*ca,   ct*ca, -sa, -sa*d],
        [st*sa,   ct*sa,  ca,  ca*d],
        [0,       0,      0,   1],
    ])


def ur5e_fk(joint_angles):
    """
    Compute UR5e forward kinematics.
    Returns: dict with 'tool0_pos' (3,), 'tool0_rot' (3,3), 'joint_positions' list of (3,)
    """
    T = np.eye(4)
    positions = [T[:3, 3].copy()]

    for i in range(6):
        alpha, a, d, offset = UR5E_DH[i]
        theta = joint_angles[i] + offset
        T = T @ dh_transform(alpha, a, d, theta)
        positions.append(T[:3, 3].copy())

    # Tool0 frame
    T_tool = T @ dh_transform(0, 0, UR5E_TOOL_D, 0)
    # Rotate to tool0 convention (Z points along approach direction)
    R_tool = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T_tool = T_tool @ R_tool

    return {
        'tool0_pos': T_tool[:3, 3],
        'tool0_rot': T_tool[:3, :3],
        'joint_positions': positions,
        'T_tool': T_tool,
    }


def ur5e_ik_numerical(target_pos, target_rpy, q_init, max_iter=50, tol=1e-4):
    """
    Numerical IK using damped least-squares (Levenberg-Marquardt).
    target_pos: (3,) desired TCP position in meters
    target_rpy: (3,) desired roll-pitch-yaw
    q_init: (6,) initial joint angles
    Returns: (6,) joint angles or None if failed
    """
    q = q_init.copy()
    damping = 0.01

    for iteration in range(max_iter):
        fk = ur5e_fk(q)
        pos_err = target_pos - fk['tool0_pos']

        # Simple position-only IK (orientation less critical for magnet)
        err = pos_err
        if np.linalg.norm(err) < tol:
            return q

        # Numerical Jacobian (position only, 3x6)
        J = np.zeros((3, 6))
        eps = 1e-6
        for j in range(6):
            q_plus = q.copy()
            q_plus[j] += eps
            fk_plus = ur5e_fk(q_plus)
            J[:, j] = (fk_plus['tool0_pos'] - fk['tool0_pos']) / eps

        # Damped least-squares
        JtJ = J.T @ J + damping * np.eye(6)
        dq = np.linalg.solve(JtJ, J.T @ err)

        # Clamp joint velocity
        dq_max = 0.3  # rad per IK iteration
        scale = min(1.0, dq_max / (np.max(np.abs(dq)) + 1e-10))
        q += scale * dq

        # Enforce joint limits
        for j in range(6):
            q[j] = np.clip(q[j], UR5E_JOINT_LIMITS[j, 0], UR5E_JOINT_LIMITS[j, 1])

    return q  # best effort


class JointDynamics:
    """Simple second-order joint dynamics with velocity/acceleration limits."""

    def __init__(self, n_joints=6):
        self.n = n_joints
        self.q = np.zeros(n_joints)       # current positions
        self.qd = np.zeros(n_joints)      # current velocities
        self.q_target = np.zeros(n_joints) # target positions

    def set_target(self, q_target):
        self.q_target = q_target.copy()

    def step(self, dt):
        """Advance dynamics by dt seconds. Returns current joint positions."""
        for i in range(self.n):
            # PD control with acceleration limit
            pos_err = self.q_target[i] - self.q[i]
            vel_desired = np.clip(pos_err * 5.0, -UR5E_MAX_VEL, UR5E_MAX_VEL)
            acc = np.clip((vel_desired - self.qd[i]) / dt, -UR5E_MAX_ACC, UR5E_MAX_ACC)
            self.qd[i] += acc * dt
            self.qd[i] = np.clip(self.qd[i], -UR5E_MAX_VEL, UR5E_MAX_VEL)
            self.q[i] += self.qd[i] * dt
            # Joint limits
            self.q[i] = np.clip(self.q[i], UR5E_JOINT_LIMITS[i, 0], UR5E_JOINT_LIMITS[i, 1])
        return self.q.copy()


# ══════════════════════════════════════════════════════════════════════════════
# MAGNETIC FORCE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MagneticForceModel:
    """
    Simplified magnetic dipole-dipole interaction model.
    Computes the force/torque a permanent magnet exerts on the MSCR tip.
    """
    # Vacuum permeability
    MU_0 = 4 * np.pi * 1e-7  # T*m/A

    # Magnet parameters (NdFeB N52 cylinder, 30mm x 15mm dia)
    MAGNET_MOMENT = 1.2      # A*m^2 (remanence ~1.4T, volume ~5.3e-6 m^3)

    # MSCR tip embedded magnet (small, 2mm x 1mm)
    TIP_MOMENT = 0.005       # A*m^2

    @classmethod
    def compute_force(cls, magnet_pos_m, tip_pos_m, B_field_dir):
        """
        Compute magnetic force magnitude and direction.
        Returns force vector (N) and field magnitude (T) at tip.
        """
        r_vec = tip_pos_m - magnet_pos_m
        r = np.linalg.norm(r_vec)
        if r < 0.01:
            r = 0.01  # prevent singularity

        r_hat = r_vec / r

        # Dipole field magnitude at distance r
        B_mag = (cls.MU_0 / (4 * np.pi)) * cls.MAGNET_MOMENT / (r ** 3)

        # Force on tip dipole in gradient of B field
        # F ~ (m_tip * dB/dr) along r direction
        dBdr = -3 * (cls.MU_0 / (4 * np.pi)) * cls.MAGNET_MOMENT / (r ** 4)
        F_mag = cls.TIP_MOMENT * abs(dBdr)

        # Force direction: attractive (toward magnet)
        F_vec = -F_mag * r_hat

        return F_vec, B_mag


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class UR5eAirwaySimulation:
    """Full-pipeline UR5e + MSCR bronchial airway navigation simulation."""

    def __init__(self):
        print("=" * 65)
        print("  UR5e + MSCR BRONCHIAL AIRWAY PHYSICAL SIMULATION")
        print("=" * 65)

        print("[1/6] Loading ONNX inverse Cosserat model...")
        self._load_nn()

        print("[2/6] Loading PRM roadmap...")
        self._load_prm()

        print("[3/6] Loading bronchial obstacle map...")
        self._load_obstacles()

        print("[4/6] Loading bronchial STL mesh...")
        self._load_stl()

        print("[5/6] Initialising UR5e kinematics & dynamics...")
        self._init_robot()

        print("[6/6] Initialising visualisation...")
        self._init_state()
        self._init_figures()

        print()
        print("  Controls:")
        print("  LEFT-CLICK  on 3D view to set a navigation target")
        print("  The UR5e will move the magnet to guide the MSCR tip")
        print("  through the bronchial tree via PRM shortest path.")
        print("  Close the window to exit.")
        print("=" * 65 + "\n")

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_nn(self):
        self.ort_sess = ort.InferenceSession(ONNX_PATH)
        mat = sio.loadmat(NORM_PATH)
        norm = mat['invNorm'][0, 0]
        self.mu_in   = norm['mu_in'].flatten().astype(np.float32)
        self.sig_in  = norm['sig_in'].flatten().astype(np.float32)
        self.mu_out  = norm['mu_out'].flatten().astype(np.float32)
        self.sig_out = norm['sig_out'].flatten().astype(np.float32)

    def _load_prm(self):
        prm = sio.loadmat(PRM_PATH)
        self.nodes = prm['nodesFiltered'].astype(np.float64)   # (N, 3) in mm
        self.edges = prm['edgesFiltered'].astype(np.int32) - 1  # 0-indexed

        N = len(self.nodes)
        self.adj = [[] for _ in range(N)]
        for e in self.edges:
            d = np.linalg.norm(self.nodes[e[0]] - self.nodes[e[1]])
            self.adj[e[0]].append((e[1], d))
            self.adj[e[1]].append((e[0], d))

        gi = prm['gridInfo'][0, 0]
        self.grid_min = gi['minBounds'].flatten()
        self.grid_max = gi['maxBounds'].flatten()

    def _load_obstacles(self):
        obs = sio.loadmat(OBSTACLE_PATH)
        self.obstacle_map = obs['obstacleMap']
        self.airway_interior = obs['airwayInterior']

    def _load_stl(self):
        lung_mesh = stl_mesh.Mesh.from_file(STL_PATH)
        self.stl_vectors = lung_mesh.vectors
        n_tri = len(self.stl_vectors)
        step = max(1, n_tri // 4000)
        self.stl_sub = self.stl_vectors[::step]

    def _init_robot(self):
        # UR5e joint dynamics
        self.dynamics = JointDynamics(6)
        # Initial configuration: arm pointing roughly toward catheter base
        q_home = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
        self.dynamics.q = q_home.copy()
        self.dynamics.q_target = q_home.copy()
        self.current_q = q_home.copy()
        self.fk_result = ur5e_fk(q_home)

    # ── State ─────────────────────────────────────────────────────────────────

    def _init_state(self):
        z_vals = self.nodes[:, 2]
        self.start_node = int(np.argmax(z_vals))
        self.current_pos = self.nodes[self.start_node].copy()
        self.target_node = None
        self.path_indices = []
        self.path_points = np.empty((0, 3))
        self.path_cursor = 0
        self.navigating = False

        self.mscr_base = self.current_pos.copy()
        self.mscr_base[2] += L_MSCR

        self.magnet_pos_mm = self.current_pos + np.array([MAGNET_OFFSET, 0, 0])
        self.prev_magnet = self.magnet_pos_mm.copy()

        # History buffers
        self.max_hist = 300
        self.hist_B   = deque(maxlen=self.max_hist)
        self.hist_az  = deque(maxlen=self.max_hist)
        self.hist_el  = deque(maxlen=self.max_hist)
        self.hist_tipX = deque(maxlen=self.max_hist)
        self.hist_tipY = deque(maxlen=self.max_hist)
        self.hist_tipZ = deque(maxlen=self.max_hist)
        self.hist_magX = deque(maxlen=self.max_hist)
        self.hist_magY = deque(maxlen=self.max_hist)
        self.hist_magZ = deque(maxlen=self.max_hist)
        self.hist_F    = deque(maxlen=self.max_hist)
        self.hist_q1   = deque(maxlen=self.max_hist)
        self.hist_q2   = deque(maxlen=self.max_hist)
        self.hist_q3   = deque(maxlen=self.max_hist)
        self.hist_t    = deque(maxlen=self.max_hist)
        self.step_count = 0

        # CSV logging
        log_path = os.path.expanduser('~/ur5e_airway_sim_log.csv')
        self.csv_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'step', 'time_s',
            'tip_x_mm', 'tip_y_mm', 'tip_z_mm',
            'mag_x_mm', 'mag_y_mm', 'mag_z_mm',
            'B_T', 'az_rad', 'el_rad',
            'F_mag_N', 'B_at_tip_T',
            'q1', 'q2', 'q3', 'q4', 'q5', 'q6',
            'tcp_x_m', 'tcp_y_m', 'tcp_z_m',
        ])

    # ── Visualisation setup ───────────────────────────────────────────────────

    def _init_figures(self):
        plt.ion()
        self.fig = plt.figure(figsize=(22, 12))
        self.fig.suptitle(
            'UR5e + MSCR Bronchial Airway Navigation Simulation',
            fontsize=14, fontweight='bold', fontfamily='serif'
        )

        # Layout: 3 columns
        # Col 0: 3D lung view (rows 0-5)
        # Col 1: UR5e arm view (rows 0-2), joint angles (rows 3-5)
        # Col 2: NN outputs (rows 0-2), tip/magnet pos (rows 3-5)
        gs = self.fig.add_gridspec(6, 3, width_ratios=[1.3, 1.0, 0.9],
                                   hspace=0.55, wspace=0.40)

        # 3D lung + navigation view
        self.ax_lung = self.fig.add_subplot(gs[:, 0], projection='3d')

        # 3D UR5e arm view
        self.ax_arm = self.fig.add_subplot(gs[0:3, 1], projection='3d')

        # Joint angle plots
        self.ax_q1 = self.fig.add_subplot(gs[3, 1])
        self.ax_q2 = self.fig.add_subplot(gs[4, 1])
        self.ax_q3 = self.fig.add_subplot(gs[5, 1])

        # NN output plots
        self.ax_B  = self.fig.add_subplot(gs[0, 2])
        self.ax_az = self.fig.add_subplot(gs[1, 2])
        self.ax_el = self.fig.add_subplot(gs[2, 2])

        # Force and position
        self.ax_F  = self.fig.add_subplot(gs[3, 2])
        self.ax_tx = self.fig.add_subplot(gs[4, 2])
        self.ax_tz = self.fig.add_subplot(gs[5, 2])

        # Apply IEEE spine styling to all 2D axes
        for ax in [self.ax_q1, self.ax_q2, self.ax_q3,
                   self.ax_B, self.ax_az, self.ax_el,
                   self.ax_F, self.ax_tx, self.ax_tz]:
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            ax.tick_params(direction='in', top=True, right=True)

        self._draw_lung_mesh()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _draw_lung_mesh(self):
        ax = self.ax_lung
        poly = Poly3DCollection(self.stl_sub, alpha=0.06,
                                facecolor='lightskyblue', edgecolor='steelblue',
                                linewidth=0.1)
        ax.add_collection3d(poly)
        mn = self.stl_vectors.reshape(-1, 3).min(axis=0)
        mx = self.stl_vectors.reshape(-1, 3).max(axis=0)
        pad = 10
        ax.set_xlim(mn[0]-pad, mx[0]+pad)
        ax.set_ylim(mn[1]-pad, mx[1]+pad)
        ax.set_zlim(mn[2]-pad, mx[2]+pad)
        ax.set_xlabel('X (mm)', fontweight='bold', fontsize=10)
        ax.set_ylabel('Y (mm)', fontweight='bold', fontsize=10)
        ax.set_zlabel('Z (mm)', fontweight='bold', fontsize=10)

    # ── PRM shortest path (Dijkstra) ─────────────────────────────────────────

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
        dists = np.linalg.norm(self.nodes - point_3d, axis=1)
        return int(np.argmin(dists))

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

    # ── Neural network inference ──────────────────────────────────────────────

    def _nn_inference(self, tip_deflection_mm):
        dptip = (tip_deflection_mm * MM_TO_M).astype(np.float32)
        x_norm = ((dptip - self.mu_in) / self.sig_in).reshape(1, -1)
        y_norm = self.ort_sess.run(None, {'input': x_norm})[0].flatten()
        y = y_norm * self.sig_out + self.mu_out
        return float(y[0]), float(y[1]), float(y[2])

    def _compute_magnet_pos_mm(self, Bmag, az, el, mscr_base_mm):
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

    # ── Mouse interaction ─────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax_lung:
            return
        from mpl_toolkits.mplot3d import proj3d
        coords_2d = []
        for pt in self.nodes:
            x2, y2, _ = proj3d.proj_transform(pt[0], pt[1], pt[2],
                                                self.ax_lung.get_proj())
            coords_2d.append([x2, y2])
        coords_2d = np.array(coords_2d)
        click = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(coords_2d - click, axis=1)
        target_idx = int(np.argmin(dists))
        src_idx = self._find_nearest_node(self.current_pos)
        if target_idx == src_idx:
            return

        print(f"[NAV] Planning path: node {src_idx} -> node {target_idx}")
        path = self._dijkstra(src_idx, target_idx)
        if not path:
            print("[NAV] No path found!")
            return

        self.path_indices = path
        raw_pts = self.nodes[path]
        self.path_points = self._interpolate_path(raw_pts, step_mm=NAV_SPEED)
        self.path_cursor = 0
        self.navigating = True
        self.target_node = target_idx
        print(f"[NAV] {len(path)} PRM nodes, {len(self.path_points)} waypoints.")

    # ── Simulation step ───────────────────────────────────────────────────────

    def _navigation_step(self):
        if self.path_cursor >= len(self.path_points):
            self.navigating = False
            print("[NAV] Target reached!")
            return

        self.current_pos = self.path_points[self.path_cursor].copy()
        self.path_cursor += 1

        # Compute tangent for MSCR base placement
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

        # NN inference: deflection -> magnetic field params
        Bmag, az, el = self._nn_inference(deflection_mm)

        # Compute desired magnet position (mm)
        self.magnet_pos_mm = self._compute_magnet_pos_mm(Bmag, az, el, self.mscr_base)

        # Convert magnet position to meters for UR5e IK
        magnet_pos_m = self.magnet_pos_mm * MM_TO_M + UR5E_BASE_POS

        # Solve IK for UR5e to reach magnet position
        q_target = ur5e_ik_numerical(
            magnet_pos_m,
            target_rpy=np.array([0.0, np.pi, 0.0]),
            q_init=self.current_q,
        )
        if q_target is not None:
            self.dynamics.set_target(q_target)

        # Advance joint dynamics
        self.current_q = self.dynamics.step(SIM_DT)
        self.fk_result = ur5e_fk(self.current_q)

        # Compute magnetic force at tip
        magnet_m = self.fk_result['tool0_pos']
        tip_m = self.current_pos * MM_TO_M
        F_vec, B_at_tip = MagneticForceModel.compute_force(
            magnet_m, tip_m, np.array([0, 0, 1])
        )
        F_mag = np.linalg.norm(F_vec)

        # Record history
        self.step_count += 1
        t = self.step_count
        self.hist_t.append(t)
        self.hist_B.append(Bmag * 1000)
        self.hist_az.append(np.degrees(az))
        self.hist_el.append(np.degrees(el))
        self.hist_tipX.append(self.current_pos[0])
        self.hist_tipY.append(self.current_pos[1])
        self.hist_tipZ.append(self.current_pos[2])
        self.hist_magX.append(self.magnet_pos_mm[0])
        self.hist_magY.append(self.magnet_pos_mm[1])
        self.hist_magZ.append(self.magnet_pos_mm[2])
        self.hist_F.append(F_mag * 1000)  # mN
        self.hist_q1.append(np.degrees(self.current_q[0]))
        self.hist_q2.append(np.degrees(self.current_q[1]))
        self.hist_q3.append(np.degrees(self.current_q[2]))

        # CSV logging
        self.csv_writer.writerow([
            t, t * SIM_DT,
            self.current_pos[0], self.current_pos[1], self.current_pos[2],
            self.magnet_pos_mm[0], self.magnet_pos_mm[1], self.magnet_pos_mm[2],
            Bmag, az, el,
            F_mag, B_at_tip,
            *self.current_q,
            *self.fk_result['tool0_pos'],
        ])

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _update_plots(self):
        # ── 3D lung navigation view ──
        ax = self.ax_lung
        while len(ax.lines) > 0:
            ax.lines[0].remove()
        while len(ax.collections) > 1:
            ax.collections[-1].remove()

        ax.scatter(self.nodes[::3, 0], self.nodes[::3, 1], self.nodes[::3, 2],
                   c='grey', s=1, alpha=0.3)

        if len(self.path_points) > 0:
            ax.plot(self.path_points[:, 0], self.path_points[:, 1],
                    self.path_points[:, 2], 'lime', linewidth=2, label='PRM path')

        if self.path_cursor > 1:
            trav = self.path_points[:self.path_cursor]
            ax.plot(trav[:, 0], trav[:, 1], trav[:, 2],
                    'r-', linewidth=2.5, label='Traversed')

        ax.plot([self.mscr_base[0], self.current_pos[0]],
                [self.mscr_base[1], self.current_pos[1]],
                [self.mscr_base[2], self.current_pos[2]],
                'r-', linewidth=4)

        ax.scatter(*self.current_pos, color='red', s=100, zorder=5,
                   depthshade=False, label='MSCR tip')
        ax.scatter(*self.magnet_pos_mm, color='black', s=120, marker='D',
                   zorder=5, depthshade=False, label='UR5e magnet')

        ax.plot([self.magnet_pos_mm[0], self.current_pos[0]],
                [self.magnet_pos_mm[1], self.current_pos[1]],
                [self.magnet_pos_mm[2], self.current_pos[2]],
                'k--', linewidth=0.8, alpha=0.5)

        ax.scatter(*self.nodes[self.start_node], color='blue', s=80,
                   marker='^', depthshade=False, label='Start')
        if self.target_node is not None:
            ax.scatter(*self.nodes[self.target_node], color='gold', s=120,
                       marker='*', depthshade=False, label='Target')
        ax.set_title('Bronchial Tree Navigation', fontweight='bold', fontsize=11)
        ax.legend(loc='upper left', fontsize=8, markerscale=0.5, frameon=False)

        # ── 3D UR5e arm view ──
        ax2 = self.ax_arm
        ax2.cla()
        joint_pos = self.fk_result['joint_positions']
        xs = [p[0] for p in joint_pos]
        ys = [p[1] for p in joint_pos]
        zs = [p[2] for p in joint_pos]
        ax2.plot(xs, ys, zs, 'b-o', linewidth=3, markersize=6, label='UR5e links')
        # Tool position
        tcp = self.fk_result['tool0_pos']
        ax2.scatter(tcp[0], tcp[1], tcp[2], color='red', s=100,
                    marker='D', label='TCP (magnet)')
        # Draw magnet cylinder direction
        mag_dir = self.fk_result['tool0_rot'][:, 2] * 0.04
        ax2.plot([tcp[0], tcp[0]+mag_dir[0]],
                 [tcp[1], tcp[1]+mag_dir[1]],
                 [tcp[2], tcp[2]+mag_dir[2]],
                 'r-', linewidth=4, label='Magnet')
        ax2.set_xlabel('X (m)', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Y (m)', fontweight='bold', fontsize=10)
        ax2.set_zlabel('Z (m)', fontweight='bold', fontsize=10)
        ax2.set_title('UR5e Arm Configuration', fontweight='bold', fontsize=11)
        ax2.legend(fontsize=8, frameon=False)
        ax2.set_xlim(-0.8, 0.8)
        ax2.set_ylim(-0.8, 0.8)
        ax2.set_zlim(-0.2, 1.0)
        ax2.tick_params(labelsize=9)

        # ── Time-series (IEEE format) ──
        t = list(self.hist_t)
        if len(t) == 0:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return

        xlim = (max(0, t[-1] - self.max_hist), t[-1] + 5)

        def _ieee_ax(ax_ts):
            """Apply IEEE formatting after cla() resets axis state."""
            ax_ts.grid(False)
            ax_ts.tick_params(direction='in', top=True, right=True,
                              labelsize=9)
            for spine in ax_ts.spines.values():
                spine.set_linewidth(1.2)

        def _ts(ax_ts, data, ylabel, title, color):
            ax_ts.cla()
            ax_ts.plot(t, list(data), color=color, linewidth=1.4)
            ax_ts.set_ylabel(ylabel, fontweight='bold', fontsize=10)
            ax_ts.set_title(title, fontweight='bold', fontsize=10)
            ax_ts.set_xlim(xlim)
            _ieee_ax(ax_ts)

        _ts(self.ax_B,  self.hist_B,
            r'$|\mathbf{B}|$ (mT)', 'Magnetic Field Magnitude', '#d62728')
        _ts(self.ax_az, self.hist_az,
            r'$\phi$ (deg)', 'Azimuth Angle', '#1f77b4')
        _ts(self.ax_el, self.hist_el,
            r'$\theta$ (deg)', 'Elevation Angle', '#2ca02c')
        _ts(self.ax_F,  self.hist_F,
            r'$|\mathbf{F}|$ (mN)', 'Magnetic Force Magnitude', '#ff7f0e')

        # Joint angles
        _ts(self.ax_q1, self.hist_q1,
            r'$q_1$ (deg)', 'Joint 1 (Shoulder Pan)', '#9467bd')
        _ts(self.ax_q2, self.hist_q2,
            r'$q_2$ (deg)', 'Joint 2 (Shoulder Lift)', '#17becf')
        _ts(self.ax_q3, self.hist_q3,
            r'$q_3$ (deg)', 'Joint 3 (Elbow)', '#d62728')
        self.ax_q3.set_xlabel('Simulation Step', fontweight='bold', fontsize=10)

        # Tip X vs Magnet X
        self.ax_tx.cla()
        self.ax_tx.plot(t, list(self.hist_tipX), '#d62728', linewidth=1.4,
                        label='MSCR Tip')
        self.ax_tx.plot(t, list(self.hist_magX), '#d62728', linewidth=1.2,
                        ls='--', label='Magnet (UR5e)')
        self.ax_tx.set_ylabel('X (mm)', fontweight='bold', fontsize=10)
        self.ax_tx.set_title('X-Axis Position', fontweight='bold', fontsize=10)
        self.ax_tx.legend(fontsize=8, frameon=False)
        self.ax_tx.set_xlim(xlim)
        _ieee_ax(self.ax_tx)

        # Tip Z vs Magnet Z
        self.ax_tz.cla()
        self.ax_tz.plot(t, list(self.hist_tipZ), '#2ca02c', linewidth=1.4,
                        label='MSCR Tip')
        self.ax_tz.plot(t, list(self.hist_magZ), '#2ca02c', linewidth=1.2,
                        ls='--', label='Magnet (UR5e)')
        self.ax_tz.set_ylabel('Z (mm)', fontweight='bold', fontsize=10)
        self.ax_tz.set_xlabel('Simulation Step', fontweight='bold', fontsize=10)
        self.ax_tz.set_title('Z-Axis Position', fontweight='bold', fontsize=10)
        self.ax_tz.legend(fontsize=8, frameon=False)
        self.ax_tz.set_xlim(xlim)
        _ieee_ax(self.ax_tz)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            while plt.fignum_exists(self.fig.number):
                if self.navigating:
                    self._navigation_step()
                else:
                    # Even when idle, update dynamics toward current target
                    self.current_q = self.dynamics.step(SIM_DT)
                    self.fk_result = ur5e_fk(self.current_q)
                self._update_plots()
                plt.pause(SIM_DT)
        except KeyboardInterrupt:
            pass

        print("\nSimulation ended.")
        if hasattr(self, 'csv_file'):
            self.csv_file.flush()
            self.csv_file.close()
            print("Log saved to ~/ur5e_airway_sim_log.csv")
        plt.close('all')


def main():
    sim = UR5eAirwaySimulation()
    sim.run()


if __name__ == '__main__':
    main()
