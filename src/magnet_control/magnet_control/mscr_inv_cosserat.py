#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import numpy as np
import onnxruntime as ort
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import csv
import os

# TF2 for robust coordinate tracking
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class MSCRAdvancedVisualizer(Node):
    def __init__(self):
        super().__init__('mscr_advanced_visualizer')

        # --- Neural Network Loading ---
        self.ort_sess = ort.InferenceSession(str(self.get_package_file('msrc_inv_model.onnx')))
        mat_data = sio.loadmat(str(self.get_package_file('inv_norm3.mat')))
        
        norm = mat_data['invNorm'][0, 0]
        self.mu_in = norm['mu_in'].flatten().astype(np.float32)
        self.sig_in = norm['sig_in'].flatten().astype(np.float32)
        self.mu_out = norm['mu_out'].flatten().astype(np.float32)
        self.sig_out = norm['sig_out'].flatten().astype(np.float32)

        # --- Control Publishers ---
        self.urscript_pub = self.create_publisher(String, '/urscript_interface/script_command', 10)

        # --- TF2 Listener for Actual TCP Tracking ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Constants & State Tracking ---
        self.cat_base = np.array([-0.14397, -0.43562, -0.24807])
        self.alpha = 0.12 
        self.prev_pm = self.cat_base + np.array([0.08, 0.0, 0.04])
        self.step_idx = 0
        self.B_min, self.B_max = 0.0005, 0.012 
        
        # --- Buffers for Diagnostic Graphs ---
        self.max_len = 100
        self.traj_history = deque(maxlen=50)
        self.history_B = deque(maxlen=self.max_len)
        self.history_az = deque(maxlen=self.max_len)
        self.history_el = deque(maxlen=self.max_len)
        self.time_axis = deque(maxlen=self.max_len)

        # --- Setup Matplotlib Figure Grid ---
        plt.ion()
        self.fig = plt.figure(figsize=(15, 8))
        self.gs = self.fig.add_gridspec(3, 2)
        self.ax3d = self.fig.add_subplot(self.gs[:, 0], projection='3d')
        self.ax_b = self.fig.add_subplot(self.gs[0, 1])
        self.ax_az = self.fig.add_subplot(self.gs[1, 1])
        self.ax_el = self.fig.add_subplot(self.gs[2, 1])
        
        # --- CSV Logging (Updated Header) ---
        home_path = os.path.expanduser('~')
        self.csv_file = open(os.path.join(home_path, 'mscr_live_results3.csv'), mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'dx', 'dy', 'dz',                # Target tip deflection
            'B_pred', 'az_pred', 'el_pred',  # NN outputs
            'pm_x', 'pm_y', 'pm_z',          # Target magnet position
            'act_x', 'act_y', 'act_z'        # Actual TCP position (TF2)
        ])
        self.csv_file.flush()
        self.timer = self.create_timer(0.1, self.main_loop)
        self.get_logger().info('Cosserat Node Initialized with TF2 Tracking.')

    def get_package_file(self, name: str):
        from importlib.resources import files
        return files('magnet_control') / name

    def main_loop(self):
        # 1. Target Generation
        t = self.step_idx * 0.05
        dx, dy, dz = 0.008*np.sin(t), 0.008*np.sin(2*t), 0.003*np.cos(t)
        self.step_idx += 1
        
        # 2. NN Inference
        dptip = np.array([dx, dy, dz], dtype=np.float32)
        x_norm = ((dptip - self.mu_in) / self.sig_in).astype(np.float32).reshape(1, -1)
        y = self.ort_sess.run(None, {'input': x_norm})[0].flatten() * self.sig_out + self.mu_out
        Bmag, az, el = float(y[0]), float(y[1]), float(y[2])

        # 3. Physics Mapping
        R = 0.04 + (1.0 - (np.clip(Bmag, self.B_min, self.B_max) / self.B_max)) * 0.06
        pm_raw = self.cat_base + np.array([R*np.cos(el)*np.cos(az), R*np.cos(el)*np.sin(az), R*np.sin(el)])
        pm = (self.alpha * pm_raw) + ((1.0 - self.alpha) * self.prev_pm)
        self.prev_pm = pm

        # 4. Command Robot
        script = f"def cosserat_move():\n  movej(p[{pm[0]:.4f}, {pm[1]:.4f}, {pm[2]:.4f}, 0.0, 3.14, 0.0], a=0.5, v=0.1, r=0.005)\nend"
        self.urscript_pub.publish(String(data=script))

        # 5. Get Actual TCP via TF2
        try:
            # Looking up transform from robot base to tool0 (TCP)
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('base', 'tool0', now)
            act_x = trans.transform.translation.x
            act_y = trans.transform.translation.y
            act_z = trans.transform.translation.z
        except TransformException as ex:
            self.get_logger().warn(f'Could not get TF: {ex}')
            act_x, act_y, act_z = 0.0, 0.0, 0.0

        # 6. History and Visuals
        self.history_B.append(Bmag * 1000)
        self.history_az.append(az)
        self.history_el.append(el)
        self.time_axis.append(self.step_idx)
        self.update_visuals(pm, [dx, dy, dz])

        # 7. Record to CSV
        self.csv_writer.writerow([
            dx, dy, dz, 
            Bmag, az, el, 
            pm[0], pm[1], pm[2],
            act_x, act_y, act_z
        ])

    def update_visuals(self, pm, tip_delta):
        [ax.cla() for ax in [self.ax3d, self.ax_b, self.ax_az, self.ax_el]]
        actual_tip = self.cat_base + np.array([0, 0, 0.06]) + tip_delta
        self.traj_history.append(actual_tip)
        hist = np.array(self.traj_history)
        
        self.ax3d.plot(hist[:,0], hist[:,1], hist[:,2], 'g-', alpha=0.5)
        self.ax3d.scatter(pm[0], pm[1], pm[2], color='black', s=80, label='Magnet')
        self.ax3d.plot([self.cat_base[0], actual_tip[0]], [self.cat_base[1], actual_tip[1]], 
                      [self.cat_base[2], actual_tip[2]], 'r-', linewidth=3, label='MSCR')
        
        self.ax3d.set_axis_off()
        self.ax3d.grid(False)
        self.ax3d.set_title("Cosserat Curvature Tracking")
        self.ax3d.set_zlim(self.cat_base[2], self.cat_base[2]+0.1)

        self.ax_b.plot(list(self.time_axis), list(self.history_B), 'r-')
        self.ax_b.set_ylabel('|B| (mT)')
        self.ax_az.plot(list(self.time_axis), list(self.history_az), 'b-')
        self.ax_az.set_ylabel('Azimuth (rad)')
        self.ax_el.plot(list(self.time_axis), list(self.history_el), 'g-')
        self.ax_el.set_ylabel('Elevation (rad)')
        plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = MSCRAdvancedVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Simulation stopped by user.')
    finally:
        # CRITICAL: This ensures the data is physically written to the disk
        if hasattr(node, 'csv_file'):
            node.csv_file.flush() # Force write remaining data
            node.csv_file.close()
            print(f"SUCCESS: Trajectory data saved.")
            
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import numpy as np
import onnxruntime as ort
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import csv

# TF2 for robust coordinate tracking
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class MSCRAdvancedVisualizer(Node):
    def __init__(self):
        super().__init__('mscr_advanced_visualizer')

        # --- Neural Network Loading ---
        self.ort_sess = ort.InferenceSession(str(self.get_package_file('mscr_inverse_model.onnx')))
        mat_data = sio.loadmat(str(self.get_package_file('inv_norm3.mat')))
        
        norm = mat_data['invNorm'][0, 0]
        self.mu_in = norm['mu_in'].flatten().astype(np.float32)
        self.sig_in = norm['sig_in'].flatten().astype(np.float32)
        self.mu_out = norm['mu_out'].flatten().astype(np.float32)
        self.sig_out = norm['sig_out'].flatten().astype(np.float32)

        # --- Control Publishers ---
        self.urscript_pub = self.create_publisher(String, '/urscript_interface/script_command', 10)

        # --- TF2 Listener for Actual TCP Tracking ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Constants & State Tracking ---
        self.cat_base = np.array([-0.14397, -0.43562, -0.24807])
        self.alpha = 0.12 
        self.prev_pm = self.cat_base + np.array([0.08, 0.0, 0.04])
        self.step_idx = 0
        self.B_min, self.B_max = 0.0005, 0.012 
        
        # --- Buffers for Diagnostic Graphs ---
        self.max_len = 100
        self.traj_history = deque(maxlen=50)
        self.history_B = deque(maxlen=self.max_len)
        self.history_az = deque(maxlen=self.max_len)
        self.history_el = deque(maxlen=self.max_len)
        self.time_axis = deque(maxlen=self.max_len)

        # --- Setup Matplotlib Figure Grid ---
        plt.ion()
        self.fig = plt.figure(figsize=(15, 8))
        self.gs = self.fig.add_gridspec(3, 2)
        self.ax3d = self.fig.add_subplot(self.gs[:, 0], projection='3d')
        self.ax_b = self.fig.add_subplot(self.gs[0, 1])
        self.ax_az = self.fig.add_subplot(self.gs[1, 1])
        self.ax_el = self.fig.add_subplot(self.gs[2, 1])
        
        # --- CSV Logging (Updated Header) ---
        self.csv_file = open('mscr_live_results2.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'dx', 'dy', 'dz',                # Target tip deflection
            'B_pred', 'az_pred', 'el_pred',  # NN outputs
            'pm_x', 'pm_y', 'pm_z',          # Target magnet position
            'act_x', 'act_y', 'act_z'        # Actual TCP position (TF2)
        ])

        self.timer = self.create_timer(0.033, self.main_loop)
        self.get_logger().info('Cosserat Node Initialized with TF2 Tracking.')

    def get_package_file(self, name: str):
        from importlib.resources import files
        return files('magnet_control') / name

    def main_loop(self):
        # 1. Target Generation
        t = self.step_idx * 0.05
        dx, dy, dz = 0.008*np.sin(t), 0.008*np.sin(2*t), 0.003*np.cos(t)
        self.step_idx += 1
        
        # 2. NN Inference
        dptip = np.array([dx, dy, dz], dtype=np.float32)
        x_norm = ((dptip - self.mu_in) / self.sig_in).astype(np.float32).reshape(1, -1)
        y = self.ort_sess.run(None, {'input': x_norm})[0].flatten() * self.sig_out + self.mu_out
        Bmag, az, el = float(y[0]), float(y[1]), float(y[2])

        # 3. Physics Mapping
        R = 0.04 + (1.0 - (np.clip(Bmag, self.B_min, self.B_max) / self.B_max)) * 0.06
        pm_raw = self.cat_base + np.array([R*np.cos(el)*np.cos(az), R*np.cos(el)*np.sin(az), R*np.sin(el)])
        pm = (self.alpha * pm_raw) + ((1.0 - self.alpha) * self.prev_pm)
        self.prev_pm = pm

        # 4. Command Robot
        script = f"def cosserat_move():\n  movej(p[{pm[0]:.4f}, {pm[1]:.4f}, {pm[2]:.4f}, 0.0, 3.14, 0.0], a=0.5, v=0.1, r=0.005)\nend"
        self.urscript_pub.publish(String(data=script))

        # 5. Get Actual TCP via TF2
        try:
            # Looking up transform from robot base to tool0 (TCP)
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('base', 'tool0', now)
            act_x = trans.transform.translation.x
            act_y = trans.transform.translation.y
            act_z = trans.transform.translation.z
        except TransformException as ex:
            self.get_logger().warn(f'Could not get TF: {ex}')
            act_x, act_y, act_z = 0.0, 0.0, 0.0

        # 6. History and Visuals
        self.history_B.append(Bmag * 1000)
        self.history_az.append(az)
        self.history_el.append(el)
        self.time_axis.append(self.step_idx)
        self.update_visuals(pm, [dx, dy, dz])

        # 7. Record to CSV
        self.csv_writer.writerow([
            dx, dy, dz, 
            Bmag, az, el, 
            pm[0], pm[1], pm[2],
            act_x, act_y, act_z
        ])

    def update_visuals(self, pm, tip_delta):
        [ax.cla() for ax in [self.ax3d, self.ax_b, self.ax_az, self.ax_el]]
        actual_tip = self.cat_base + np.array([0, 0, 0.06]) + tip_delta
        self.traj_history.append(actual_tip)
        hist = np.array(self.traj_history)
        
        self.ax3d.plot(hist[:,0], hist[:,1], hist[:,2], 'g-', alpha=0.5)
        self.ax3d.scatter(pm[0], pm[1], pm[2], color='black', s=80, label='Magnet')
        self.ax3d.plot([self.cat_base[0], actual_tip[0]], [self.cat_base[1], actual_tip[1]], 
                      [self.cat_base[2], actual_tip[2]], 'r-', linewidth=3, label='MSCR')
        
        self.ax3d.set_axis_off()
        self.ax3d.grid(False)
        self.ax3d.set_title("Cosserat Curvature Tracking")
        self.ax3d.set_zlim(self.cat_base[2], self.cat_base[2]+0.1)

        self.ax_b.plot(list(self.time_axis), list(self.history_B), 'r-')
        self.ax_b.set_ylabel('|B| (mT)')
        self.ax_az.plot(list(self.time_axis), list(self.history_az), 'b-')
        self.ax_az.set_ylabel('Azimuth (rad)')
        self.ax_el.plot(list(self.time_axis), list(self.history_el), 'g-')
        self.ax_el.set_ylabel('Elevation (rad)')
        plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = MSCRAdvancedVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down MSCR Node...')
    finally:
        # Crucial for saving the trajectory results
        if hasattr(node, 'csv_file'):
            node.csv_file.close()
            print(f"File closed. Check your home directory for mscr_live_results3.csv")
            
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
