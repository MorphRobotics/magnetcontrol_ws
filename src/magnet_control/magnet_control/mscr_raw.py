#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import onnxruntime as ort
import scipy.io as sio
import csv
import os

class MSCRRawDataLogger(Node):
    def __init__(self):
        super().__init__('mscr_raw_logger')

        # --- Load Model & Normalization ---
        # Note: Ensure these filenames match your setup.py exactly
        self.ort_sess = ort.InferenceSession(str(self.get_package_file('mscr_inverse_model.onnx')))
        mat_data = sio.loadmat(str(self.get_package_file('inv_norm3.mat')))
        norm = mat_data['invNorm'][0, 0]
        
        # Scaling Parameters
        self.mu_in = norm['mu_in'].flatten().astype(np.float32)
        self.sig_in = norm['sig_in'].flatten().astype(np.float32)
        self.mu_out = norm['mu_out'].flatten().astype(np.float32)
        self.sig_out = norm['sig_out'].flatten().astype(np.float32)

        # --- State Variables ---
        self.cat_base = np.array([-0.14397, -0.43562, -0.24807])
        self.step_idx = 0
        self.prev_pm = self.cat_base + np.array([0.08, 0.0, 0.04])
        self.alpha = 0.12 # Smoothing

        # --- CSV Setup for Excel Replotting ---
        home = os.path.expanduser('~')
        self.csv_path = os.path.join(home, 'mscr_raw_butterfly_data.csv')
        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        # Comprehensive Header for Excel
        self.writer.writerow([
            'Step', 'Target_dx', 'Target_dy', 'Target_dz', 
            'Raw_B_Tesla', 'Raw_Azimuth_rad', 'Raw_Elevation_rad',
            'Magnet_X', 'Magnet_Y', 'Magnet_Z'
        ])

        # 10 Hz Control Loop
        self.timer = self.create_timer(0.1, self.main_loop)
        self.get_logger().info(f'Logging raw data to {self.csv_path}')

    def get_package_file(self, name: str):
        from importlib.resources import files
        return files('magnet_control') / name

    def main_loop(self):
        # 1. Target Butterfly Generation
        t = self.step_idx * 0.05
        dx, dy, dz = 0.008*np.sin(t), 0.008*np.sin(2*t), 0.003*np.cos(t)
        
        # 2. NN Inference
        dptip = np.array([dx, dy, dz], dtype=np.float32)
        x_norm = ((dptip - self.mu_in) / self.sig_in).reshape(1, -1)
        y = self.ort_sess.run(None, {'input': x_norm})[0].flatten() * self.sig_out + self.mu_out
        Bmag, az, el = float(y[0]), float(y[1]), float(y[2])

        # 3. Physics Mapping to Robot Coordinates
        # Mapping B-field magnitude to radial distance
        R = 0.04 + (1.0 - (np.clip(Bmag, 0.0005, 0.012) / 0.012)) * 0.06
        pm_raw = self.cat_base + np.array([
            R * np.cos(el) * np.cos(az), 
            R * np.cos(el) * np.sin(az), 
            R * np.sin(el)
        ])
        
        # Exponential Smoothing
        pm = (self.alpha * pm_raw) + ((1.0 - self.alpha) * self.prev_pm)
        self.prev_pm = pm

        # 4. Save Raw Data Row
        self.writer.writerow([
            self.step_idx, dx, dy, dz, 
            Bmag, az, el, 
            pm[0], pm[1], pm[2]
        ])
        
        self.step_idx += 1
        if self.step_idx % 50 == 0:
            self.csv_file.flush() # Ensure data is written even if script crashes

def main(args=None):
    rclpy.init(args=args)
    node = MSCRRawDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.csv_file.close()
        print(f"Dataset successfully saved to ~/mscr_raw_butterfly_data.csv")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
