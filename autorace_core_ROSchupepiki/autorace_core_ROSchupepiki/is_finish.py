import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class FinishLineDetector(Node):
    def __init__(self):
        super().__init__('finish_detector')

        # ПАРАМЕТРЫ
        self.declare_parameter('show_gui', True)

        # Настройка порога срабатывания (сколько пикселей должно быть)
        # Если финишная линия занимает хотя бы 2-5% от ROI - считаем, что нашли.
        self.declare_parameter('pixel_threshold_pct', 0.02)  # 2%

        self.subscription = self.create_subscription(
            Image,
            '/color/image_projected',
            self.listener_callback,
            10)

        self.publisher_ = self.create_publisher(Bool, '/is_finish', 10)

        self.bridge = CvBridge()
        self.get_logger().info('Finish Line Detector Started. Looking for Checkerboard pattern...')

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

        height, width, _ = cv_image.shape

        # ОПРЕДЕЛЯЕМ ROI
        roi_y_start = int(height * 0.6)
        roi_y_end = height
        roi_x_start = int(width * 0.2)
        roi_x_end = int(width * 0.8)

        roi = cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_area = roi.shape[0] * roi.shape[1]

        if roi_area == 0:
            return

        # ПОИСК ЧЕРНОГО И БЕЛОГО
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Маска черных клеток (очень темные пиксели, < 50)
        _, mask_black = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)

        # Маска белых клеток (очень светлые пиксели, > 200)
        _, mask_white = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)

        # Считаем количество пикселей
        count_black = cv2.countNonZero(mask_black)
        count_white = cv2.countNonZero(mask_white)

        # Вычисляем проценты заполнения
        pct_black = count_black / roi_area
        pct_white = count_white / roi_area

        threshold = self.get_parameter('pixel_threshold_pct').get_parameter_value().double_value

        # Финишная прямая - это КОНТРАСТ. Должно быть И черное, И белое.
        # Обычная дорога - серая (не попадет ни туда, ни туда).
        # Разметка - белая (попадет только в white).

        is_finish_detected = False

        # Если есть и черное, и белое в достаточном количестве
        if pct_black > threshold and pct_white > threshold:
            is_finish_detected = True

        # 4. ПУБЛИКАЦИЯ
        msg_bool = Bool()
        msg_bool.data = is_finish_detected
        self.publisher_.publish(msg_bool)

        if is_finish_detected:
            self.get_logger().info(f"FINISH LINE DETECTED! (B: {pct_black:.2f}, W: {pct_white:.2f})")

        # 5. ВИЗУАЛИЗАЦИЯ (GUI)
        show_gui = self.get_parameter('show_gui').get_parameter_value().bool_value
        if show_gui:
            # Рисуем прямоугольник ROI на оригинале
            debug_img = cv_image.copy()
            color = (0, 255, 0) if is_finish_detected else (0, 0, 255)

            cv2.rectangle(debug_img, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), color, 2)

            # Текст статуса
            status_text = "FINISH!" if is_finish_detected else "Road"
            cv2.putText(debug_img, f"Status: {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Вывод статистики
            cv2.putText(debug_img, f"Black: {pct_black:.3f} | White: {pct_white:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Finish Detector", debug_img)

            # Можно показать маски для отладки, если нужно
            # cv2.imshow("Mask Black", mask_black)
            # cv2.imshow("Mask White", mask_white)

            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FinishLineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()