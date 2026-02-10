#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Vector3

import numpy as np
import onnxruntime as ort
import scipy.io as sio
import csv

# TF2 for robust coordinate tracking
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class MSCRInverseControl(Node):
    def __init__(self):
        super().__init__('mscr_inverse_control')

        # --- TF2 Setup ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Publishers & Subscribers ---
        self.urscript_pub = self.create_publisher(String, '/urscript_interface/script_command', 10)
        self.desired_tip_sub = self.create_subscription(Vector3, '/mscr/desired_tip_delta', self.desired_tip_callback, 10)

        # --- Load ONNX network + norms ---
        self.get_logger().info('Loading inverse MLP and normalization...')
        self.ort_sess = ort.InferenceSession(str(self.get_package_file('inv_net.onnx')))
        
        norm = sio.loadmat(self.get_package_file('inv_norm.mat'))
        self.mu_in   = norm['mu_in'].flatten().astype(np.float32)
        self.sig_in  = norm['sig_in'].flatten().astype(np.float32)
        self.mu_out  = norm['mu_out'].flatten().astype(np.float32)
        self.sig_out = norm['sig_out'].flatten().astype(np.float32)

        # --- Robot/Physical Params ---
        self.cat_tip_center = np.array([-0.14397, -0.43562, -0.24807], dtype=np.float32)
        self.z_rel_mag = 0.07 
        self.B_min, self.B_max = 0.001, 0.05

        # --- Low-Pass Filter State ---
        self.alpha = 0.2  # Smoothing factor: 0.0 (static) to 1.0 (no filtering)
        # Initialize previous target at the catheter center to prevent startup jumps
        self.prev_pm = self.cat_tip_center + np.array([0.05, 0.0, self.z_rel_mag])

        # --- State Tracking ---
        self.current_tcp_pos = np.array([0.0, 0.0, 0.0])
        self.step_idx = 0
        self.timer = self.create_timer(5.0, self.timer_step)

        # --- CSV Logging ---
        self.csv_file = open('mscr_live_results.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'dx', 'dy', 'dz',
            'B_pred', 'az_pred', 'el_pred',
            'target_x', 'target_y', 'target_z',
            'act_x', 'act_y', 'act_z'
        ])

        self.get_logger().info('MSCR Node Active: LPF implemented (alpha=0.2).')

    def get_package_file(self, name: str):
        from importlib.resources import files
        return files('magnet_control') / name

    def timer_step(self):
        r = 0.005 # Scaled to 5mm
        theta = self.step_idx * (np.pi / 10.0)
        self.step_idx += 1
        msg = Vector3(x=float(r * np.cos(theta)), y=float(r * np.sin(theta)), z=0.0)
        self.desired_tip_callback(msg)

    def desired_tip_callback(self, msg: Vector3):
        # 1. TF2 Lookup for Actual Pose
        try:
            t = self.tf_buffer.lookup_transform('base', 'tool0', rclpy.time.Time())
            self.current_tcp_pos = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ])
        except TransformException as ex:
            self.get_logger().warn(f'TF2 Lookup Failed: {ex}')

        # 2. Neural Inference
        dptip = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        x_norm = ((dptip - self.mu_in) / self.sig_in).reshape(1, -1)
        outputs = self.ort_sess.run(None, {'input': x_norm})
        y = np.array(outputs[0]).reshape(-1) * self.sig_out + self.mu_out
        Bmag, az, el = float(y[0]), float(y[1]), float(y[2])

        # 3. Workspace Mapping (Raw Prediction)
        Bclamped = float(np.clip(Bmag, self.B_min, self.B_max))
        R = 0.05 + ((Bclamped - self.B_min) / (self.B_max - self.B_min + 1e-8)) * 0.15
        pm_raw = self.cat_tip_center + np.array([R * np.cos(az), R * np.sin(az), self.z_rel_mag])

        # 4. Apply Low-Pass Filter
        # pm_filtered = (alpha * new_data) + ((1 - alpha) * old_data)
        pm = (self.alpha * pm_raw) + ((1.0 - self.alpha) * self.prev_pm)
        self.prev_pm = pm # Update state for next cycle

        # 5. Command URScript
        script = f"def mscr_move():\n  movej(p[{pm[0]:.4f}, {pm[1]:.4f}, {pm[2]:.4f}, 0.0, 3.14, 0.0], a=1.0, v=0.2, r=0)\nend"
        self.urscript_pub.publish(String(data=script))

        # 6. Logging
        self.get_logger().info(
            f"dP: [{msg.x*1000:.1f}, {msg.y*1000:.1f}] mm | Filtered Target: {pm}"
        )
        self.csv_writer.writerow([
            msg.x, msg.y, msg.z, Bmag, az, el,
            pm[0], pm[1], pm[2],
            self.current_tcp_pos[0], self.current_tcp_pos[1], self.current_tcp_pos[2]
        ])

def main(args=None):
    rclpy.init(args=args)
    node = MSCRInverseControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
