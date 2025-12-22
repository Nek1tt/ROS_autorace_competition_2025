#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from math import pi

class CircleDriveNode(Node):
    def __init__(self):
        super().__init__('circle_drive_node')
        
        # Параметры движения
        self.linear_speed = 0.5   # м/с
        self.angular_speed = 0.2 # рад/с (R = v/ω = 10м)
        
        # Создание издателя
        self.publisher = self.create_publisher(Twist, '/cmd_vel_tunnel', 10)
        
        # Таймер для публикации (10 Гц)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info('Узел движения по кругу запущен (R=10м)')
    
    def timer_callback(self):
        msg = Twist()
        msg.linear.x = self.linear_speed
        msg.angular.z = self.angular_speed
        
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CircleDriveNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
