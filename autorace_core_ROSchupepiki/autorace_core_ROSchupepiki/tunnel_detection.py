#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class TunnelDetectorNode(Node):
    def __init__(self):
        super().__init__('tunnel_detector_node')

        # Параметры движения
        self.linear_speed = 0.5   # м/с
        self.angular_speed = 0.2  # рад/с

        self.bridge = CvBridge()
        self.tunnel_mode = False

        # Паблишеры
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.is_tunnel_pub = self.create_publisher(Bool, '/is_tunnel', 1)
    
        # self.finish_pub = self.create_publisher(Bool, '/finish_line', 1)

        # Подписка на камеру
        self.image_sub = self.create_subscription(
            Image, '/color/image', self.image_callback, 10
        )

        # Таймер на управление
        self.timer = self.create_timer(0.1, self.timer_callback)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        dark_ratio, bright_ratio = self.get_ratios(frame)

        if not self.tunnel_mode:
            # ВХОД В ТУННЕЛЬ: стало темно
            if dark_ratio > 0.6 and bright_ratio < 0.2:
                self.tunnel_mode = True
                self.is_tunnel_pub.publish(Bool(data=True)) # забираем управление у ПИДа
                self.get_logger().info('>>> ENTER TUNNEL!')
        else:
            # ВЫХОД ИЗ ТУННЕЛЯ: стало очень светло
            if bright_ratio > 0.65:
                self.get_logger().info('<<< EXIT TUNNEL!')
                
                # Публикуем сигнал финиша
                # self.finish_pub.publish(Bool(data=True))
                self.is_tunnel_pub.publish(Bool(data=False))
                
                # Выключаем этот узел
                self.tunnel_mode = False
                self.timer.cancel()

    def timer_callback(self):
        if self.tunnel_mode:
            cmd = Twist()
            cmd.linear.x = self.linear_speed
            cmd.angular.z = self.angular_speed
            self.cmd_pub.publish(cmd)

    def get_ratios(self, frame: np.ndarray):
        h = frame.shape[0]
        roi = frame[0:h // 2, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        
        dark_ratio = float(np.count_nonzero(v < 80)) / v.size
        bright_ratio = float(np.count_nonzero(v > 200)) / v.size
        return dark_ratio, bright_ratio


def main(args=None):
    rclpy.init(args=args)
    node = TunnelDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
