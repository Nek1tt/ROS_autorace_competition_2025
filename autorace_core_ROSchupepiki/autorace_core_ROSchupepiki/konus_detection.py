import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np


class KonusDetection(Node):
    def __init__(self):
        super().__init__('konus_detection')

        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.subscription = self.create_subscription(
            Image,
            '/color/image',
            self.listener_callback,
            qos_profile)

        self.is_konus_pub = self.create_publisher(Bool, '/is_cone', 1)

        self.bridge = CvBridge()
        self.debug_mode = False

        # ТОЛЬКО ОРАНЖЕВЫЙ/КРАСНЫЙ
        # Диапазон оранжевый (5-20 в HSV, исключая желтый >25)
        self.lower_orange1 = np.array([0, 80, 80])
        self.upper_orange1 = np.array([20, 255, 255])
        
        # Диапазон красный на противоположном конце спектра
        self.lower_orange2 = np.array([160, 80, 80])
        self.upper_orange2 = np.array([180, 255, 255])

        self.get_logger().info("Konus Detection started (ORANGE ONLY)")


    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            debug_img = cv_image.copy()

            found_cone = self.process_cones(cv_image, debug_img)

            if found_cone:
                msg_bool = Bool()
                msg_bool.data = True
                self.is_konus_pub.publish(msg_bool)

            if self.debug_mode:
                cv2.imshow("Konus Debug", debug_img)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')


    def process_cones(self, image, debug_img):
        """
        Строгая детекция только оранжевых конусов
        """
        h, w = image.shape[:2]
        
        # ROI: нижние 60% кадра
        roi_start = int(h * 0.4)
        roi = image[roi_start:, :]
        
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # МАСКА ТОЛЬКО ДЛЯ ОРАНЖЕВОГО
        mask1 = cv2.inRange(hsv, self.lower_orange1, self.upper_orange1)
        mask2 = cv2.inRange(hsv, self.lower_orange2, self.upper_orange2)
        mask_orange = cv2.bitwise_or(mask1, mask2)

        # Морфология для очистки шума
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cone_detected = False

        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # Фильтр по площади
                if area < 300 or area > 50000:
                    continue

                x, y, w_box, h_box = cv2.boundingRect(cnt)
                y_real = y + roi_start
                
                aspect_ratio = float(w_box) / h_box if h_box > 0 else 0

                # СТРОГИЕ ФИЛЬТРЫ ДЛЯ КОНУСОВ
                # Пропорции: конус вертикальный (w/h от 0.3 до 1.0)
                if not (0.3 <= aspect_ratio <= 1.0):
                    continue
                
                # Минимальная высота (конус должен быть достаточно высоким)
                if h_box < 20:
                    continue
                
                # Проверка выпуклости (конус имеет простую форму)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if solidity < 0.75:  # Конус довольно выпуклый
                    continue
                
                # Проверка позиции: не берем объекты слишком близко к краям
                margin = int(w * 0.05)
                if x < margin or (x + w_box) > (w - margin):
                    continue
                
                # ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ - ЭТО КОНУС
                cone_detected = True
                
                if self.debug_mode:
                    cv2.rectangle(debug_img, (x, y_real), (x + w_box, y_real + h_box), (0, 255, 0), 3)
                    cv2.putText(debug_img, f"CONE A={int(area)} R={aspect_ratio:.2f}", 
                               (x, y_real - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Показываем только оранжевую маску
        if self.debug_mode:
            # Создаем цветную визуализацию для понимания
            mask_colored = cv2.cvtColor(mask_orange, cv2.COLOR_GRAY2BGR)
            combined = np.vstack([roi, mask_colored])
            cv2.imshow("Orange Mask Only", combined)

        return cone_detected


def main(args=None):
    rclpy.init(args=args)
    node = KonusDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

