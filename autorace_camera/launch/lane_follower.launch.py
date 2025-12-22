import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    """Generate launch description for lane following system."""
    
    # Get package directory
    pkg_dir = get_package_share_directory('autorace_camera')
    
    # Launch arguments
    is_calibrating_arg = DeclareLaunchArgument(
        'is_calibrating',
        default_value='false',
        description='Enable calibration mode with visualization overlay'
    )
    
    enable_projection_arg = DeclareLaunchArgument(
        'enable_projection',
        default_value='true',
        description='Enable perspective projection node (disable if image already bird\'s-eye)'
    )
    
    # Config files
    projection_config = os.path.join(
        pkg_dir,
        'calibration/extrinsic_calibration',
        'projection_improved.yaml'
    )
    
    lane_follower_config = os.path.join(
        pkg_dir,
        'calibration/extrinsic_calibration',
        'lane_follower.yaml'
    )
    
    # Only launch if enable_projection is true
    node_projection = Node(
        package='autorace_camera',
        namespace='camera',
        executable='image_projection_improved',
        name='image_projection',
        parameters=[
            projection_config,
            {
                'is_calibrating': LaunchConfiguration('is_calibrating'),
                'input_format': 'raw',
                'output_format': 'raw',
            }
        ],
        remappings=[
            ('/color/image', '/color/image'),  # Input from camera
            ('/color/image_projected', '/color/image_projected'),  # Output
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_projection'))
    )
    
    node_lane_follower = Node(
        package='autorace_camera',
        executable='lane_follower',
        name='lane_follower',
        parameters=[
            lane_follower_config,
            {
                'is_calibrating': LaunchConfiguration('is_calibrating')
            }
        ],
        remappings=[
            ('/color/image_projected', '/color/image_projected'),  # Input
            ('/cmd_vel', '/cmd_vel'),  # Output
        ],
        output='screen'
    )
    
    # Launch description
    ld = LaunchDescription([
        is_calibrating_arg,
        enable_projection_arg,
        node_projection,
        node_lane_follower,
    ])
    
    return ld