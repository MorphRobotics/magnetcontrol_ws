#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import os


class URJointTrajectoryPlayer(Node):
    def __init__(self):
        super().__init__('ur_joint_trajectory_player')

        # ---- USER PARAMETERS ----
        # Path to CSV file exported from Excel.
        # Format: 6 rows (q1..q6), 10001 columns (time steps).
        self.trajectory_file = '/home/dozie/train_traj.csv'

        # Time between points [s] (adjust to match how fast you want to move)
        self.dt = 0.02

        # UR motion parameters
        self.a = 1.2   # acceleration
        self.v = 0.25  # velocity

        # -------------------------
        self.pub = self.create_publisher(
            String,
            '/urscript_interface/script_command',
            10
        )

        # Load trajectory from file
        self.joint_traj = self.load_trajectory(self.trajectory_file)
        self.num_points = self.joint_traj.shape[1]
        self.index = 0

        # Create timer to step through the trajectory
        self.timer = self.create_timer(self.dt, self.send_next_point)

        self.get_logger().info(
            f"URJointTrajectoryPlayer initialized with {self.num_points} points."
        )

    # --------------------------------------------------------------
    # Load 6xN joint matrix from CSV (rows q1..q6, columns timesteps)
    # --------------------------------------------------------------
    def load_trajectory(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Trajectory file not found: {path}. "
                "Make sure you exported the Excel sheet as CSV."
            )

        # CSV should have 6 rows, N columns, no headers.
        data = np.loadtxt(path, delimiter=',')
        if data.shape[0] != 6:
            raise ValueError(
                f"Expected 6 rows (q1..q6), but file has {data.shape[0]} rows."
            )

        self.get_logger().info(
            f"Loaded trajectory from {path} with shape {data.shape} (rows x cols)."
        )
        return data

    # --------------------------------------------------------------
    # Timer callback: send next joint position as URScript movej
    # --------------------------------------------------------------
    def send_next_point(self):
        if self.index >= self.num_points:
            self.get_logger().info("All trajectory points sent.")
            self.destroy_timer(self.timer)
            return

        # Extract column 'index' as [q1..q6]
        q = self.joint_traj[:, self.index]  # shape (6,)

        # Build URScript program to move to this joint position
        script = f"""def follow_traj():
  set_digital_out(1, True)
  movej([{q[0]}, {q[1]}, {q[2]}, {q[3]}, {q[4]}, {q[5]}], a={self.a}, v={self.v}, r=0)
  textmsg("Reached trajectory point {self.index}")
end"""

        msg = String()
        msg.data = script
        self.pub.publish(msg)

        self.get_logger().info(
            f"Sent point {self.index + 1}/{self.num_points}: "
            f"[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}, {q[4]:.4f}, {q[5]:.4f}]"
        )

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = URJointTrajectoryPlayer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

