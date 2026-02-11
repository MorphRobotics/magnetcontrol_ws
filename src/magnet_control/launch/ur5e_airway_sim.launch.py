"""
Launch file for UR5e + MSCR bronchial airway navigation simulation in Gazebo.

Launches:
  1. Gazebo with bronchial airway world
  2. Robot state publisher (UR5e + magnet URDF)
  3. ros2_control controller manager (joint trajectory + state broadcaster)
  4. MSCR navigation controller (PRM + ONNX inverse model)
  5. RViz for MSCR/path visualization
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('magnet_control')
    ws_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkg_share))))

    # Paths
    urdf_xacro = os.path.join(pkg_share, 'urdf', 'ur5e_magnet.urdf.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'bronchial_airway.world')
    controllers_yaml = os.path.join(pkg_share, 'config', 'ur5e_controllers.yaml')
    rviz_config = os.path.join(pkg_share, 'config', 'airway_sim.rviz')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    paused = LaunchConfiguration('paused', default='false')

    # Process xacro to URDF
    robot_description = ParameterValue(
        Command(['xacro ', urdf_xacro]), value_type=str
    )

    # 1. Gazebo
    gazebo = ExecuteProcess(
        cmd=[
            'gazebo', '--verbose', world_file,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
        ],
        output='screen',
    )

    # 2. Spawn UR5e in Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'ur5e_magnet',
            '-x', '0.0', '-y', '0.0', '-z', '0.0',
        ],
        output='screen',
    )

    # 3. Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    # 4. Joint state broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # 5. Joint trajectory controller
    joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # 6. MSCR navigation controller (delayed to let Gazebo + controllers start)
    mscr_navigation_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='magnet_control',
                executable='ur5e_airway_navigation',
                name='mscr_airway_navigation',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'prm_path': os.path.join(ws_root, 'prm_roadmap.mat'),
                    'obstacle_path': os.path.join(ws_root, 'bronchial_obstacle_map.mat'),
                    'stl_path': os.path.join(ws_root, 'Bronchial tree anatomy-1mm-shell.STL'),
                    'auto_navigate': True,
                }],
            ),
        ],
    )

    # 7. RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('paused', default_value='false'),
        gazebo,
        robot_state_publisher,
        spawn_robot,
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner,
        mscr_navigation_node,
        rviz_node,
    ])
