from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Lidar Node
        Node(
            package='qcar2_nodes',
            executable='lidar',
            name='Lidar',
            parameters=[{
                'device_type': 'virtual',
                'uri': 'tcpip://127.0.0.1:18960'
            }]
        ),
        
        # 2. Realsense Camera Node
        Node(
            package='qcar2_nodes',
            executable='rgbd',
            name='RealsenseCamera',
            parameters=[{
                'device_type': 'virtual',
                'uri': 'tcpip://127.0.0.1:18965',
                'frame_width_rgb': 640,
                'frame_height_rgb': 480,
                'frame_width_depth': 640,
                'frame_height_depth': 480
            }]
        ),
        
        # 3. CSI Camera Node
        Node(
            package='qcar2_nodes',
            executable='csi',
            name='csi_camera',
            parameters=[{
                'device_type': 'virtual',
                'uri': 'video://127.0.0.1:1@tcpip://127.0.0.1:18962',
                'frame_width': 410,
                'frame_height': 205,
                'frame_rate': 8.0,
                'camera_num': 1
            }]
        ),        
        
        # 4. Hardware Node
        Node(
            package='qcar2_nodes',
            executable='qcar2_hardware',
            name='qcar2_hardware',
            parameters=[{
                'device_type': 'virtual',
                'uri': 'tcpip://127.0.0.1:18960'
            }]
        )
    ])
