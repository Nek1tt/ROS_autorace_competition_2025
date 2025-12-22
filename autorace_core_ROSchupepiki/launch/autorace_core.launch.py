import os
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_autorace_camera = get_package_share_directory('autorace_camera')

    
    camera_calibration = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_autorace_camera, 'launch', 'extrinsic_camera_calibration.launch.py')))
    
    lane_follower = Node(
        package='autorace_core_ROSchupepiki',
        executable='lane_follower',
        output='screen'
    )

    monitor = Node(
        package='autorace_core_ROSchupepiki',
        executable='monitor',
        output='screen'
    )
    konus_detection = Node(
        package='autorace_core_ROSchupepiki',
        executable='konus_detection',
        output='screen'
    )
    tunnel_detection = Node(
        package='autorace_core_ROSchupepiki',
        executable='tunnel_detection',
        output='screen'
    )

    aruco_detection = Node(
        package='autorace_core_ROSchupepiki',
        executable='aruco_detection',
        output='screen'
    )

    return LaunchDescription([
        camera_calibration,
        lane_follower,
        monitor,
        konus_detection,
        tunnel_detection,
        aruco_detection
     ])
