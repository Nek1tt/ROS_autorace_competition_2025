import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

class TunnelDetectorColor(Node):
    def __init__(self):
        super().__init__('tunnel_detector_color')
        
        self.subscription = self.create_subscription(
            Image,
            '/color/image',
            self.image_callback,
            10)
        
        self.publisher_ = self.create_publisher(Bool, 'is_tunnel', 10)
        
        self.bridge = CvBridge()
        
        # Порог: минимальный процент разметки для "не туннеля"
        self.marking_threshold_percent = 5.0  # Если меньше -> туннель
        
        self.get_logger().info('Tunnel Detector (Yellow/White) Started')

    def image_callback(self, msg):
        try:
            # Конвертация в OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Конвертируем в HSV
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # --- ЖЕЛТЫЙ ЦВЕТ ---
            # Hue: 15-35, Saturation: высокая, Value: высокий
            lower_yellow = np.array([15, 80, 80])
            upper_yellow = np.array([35, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # --- БЕЛЫЙ ЦВЕТ ---
            # Белый: любой Hue, низкая Saturation, высокий Value (яркость)
            lower_white = np.array([0, 0, 200])      # Почти чистый белый
            upper_white = np.array([180, 30, 255])   # Допускаем слабый оттенок
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Объединяем маски (желтый ИЛИ белый)
            combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
            
            # Считаем процент пикселей с разметкой
            marking_pixels = np.sum(combined_mask > 0)
            total_pixels = combined_mask.size
            marking_percent = (marking_pixels / total_pixels) * 100.0
            
            # Если разметки НЕТ или очень мало -> туннель
            is_tunnel = marking_percent < self.marking_threshold_percent
            
            # Публикация
            out_msg = Bool()
            out_msg.data = bool(is_tunnel)
            self.publisher_.publish(out_msg)
            
            if is_tunnel:
                self.get_logger().info(f'TUNNEL: Marking {marking_percent:.2f}%')
            # else:
                # self.get_logger().info(f'ROAD: Marking {marking_percent:.2f}%')
                
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TunnelDetectorColor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
