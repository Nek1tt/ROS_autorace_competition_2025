import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import sys


class ArucoOneShotNode(Node):
    def __init__(self):
        super().__init__('aruco_oneshot_node')

        self.declare_parameter('crop_x_start_pct', 0.6)
        self.declare_parameter('crop_y_end_pct', 0.4)
        self.declare_parameter('show_gui', True)
        self.declare_parameter('min_marker_area', 3000)

        self.mission_completed = False

        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            if hasattr(cv2.aruco, 'DetectorParameters_create'):
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            else:
                self.aruco_params = cv2.aruco.DetectorParameters()
        except Exception as e:
            self.get_logger().error(f"ArUco init failed: {e}")
            sys.exit(1)

        self.subscription = self.create_subscription(Image, '/color/image', self.listener_callback, 10)
        self.img_pub = self.create_publisher(Image, '/color/image/cropped', 10)
        self.mission_pub = self.create_publisher(Float32, '/mission_aruco', 10)
        self.bridge = CvBridge()

        self.get_logger().info('Ready. Waiting for CLOSE marker to publish ONCE.')

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if cv_image is None: return
        except CvBridgeError:
            return

        # Читаем параметры
        min_area = self.get_parameter('min_marker_area').get_parameter_value().integer_value
        show_gui = self.get_parameter('show_gui').get_parameter_value().bool_value

        # Делаем Кроп
        h, w = cv_image.shape[:2]
        x_pct = self.get_parameter('crop_x_start_pct').get_parameter_value().double_value
        y_pct = self.get_parameter('crop_y_end_pct').get_parameter_value().double_value

        x_start = int(w * max(0.0, min(1.0, x_pct)))
        y_end = int(h * max(0.0, min(1.0, y_pct)))

        if (w - x_start) < 10 or y_end < 10: return

        cropped = cv_image[0:y_end, x_start:w].copy()

        # Ищем Aruco
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        # Логика обработки
        found_valid_marker = False

        if ids is not None and len(ids) > 0:
            # Проходимся по всем найденным маркерам
            for i in range(len(ids)):
                current_id = int(ids[i][0])
                current_corners = corners[i][0]  # Получаем 4 точки маркера

                # Считаем ПЛОЩАДЬ маркера
                area = cv2.contourArea(current_corners)

                # Для визуализации (Красный = далеко/уже было, Зеленый = подходит)
                color = (0, 0, 255)  # Red по дефолту

                # ПРОВЕРКА УСЛОВИЙ:
                # 1. Площадь больше порога (мы близко)
                # 2. Миссия еще НЕ выполнена
                if area > min_area:
                    if not self.mission_completed:
                        # !!! ВЫПОЛНЯЕМ МИССИЮ !!!
                        sqrt_val = round(math.sqrt(current_id), 3)

                        msg_out = Float32()
                        msg_out.data = float(sqrt_val)
                        self.mission_pub.publish(msg_out)

                        self.mission_completed = True  # Блокируем повторную отправку
                        self.get_logger().warning(f"MISSION DONE! ID: {current_id}, Area: {area:.0f}, Sent: {sqrt_val}")

                        color = (0, 255, 0)  # Green (Успех)
                    else:
                        # Мы близко, но миссия уже была выполнена ранее
                        color = (255, 255, 0)  # Cyan (Уже было)

                # Рисуем рамку и инфо
                cv2.polylines(cropped, [current_corners.astype(np.int32)], True, color, 2)

                # Пишем площадь на экране, чтобы ты мог настроить параметр
                status_text = "DONE" if self.mission_completed else "WAIT"
                cv2.putText(cropped, f"Area:{int(area)} ({status_text})",
                            (int(current_corners[0][0]), int(current_corners[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Публикуем картинку кропа
        out_msg = self.bridge.cv2_to_imgmsg(cropped, "bgr8")
        out_msg.header = msg.header
        self.img_pub.publish(out_msg)

        # Показываем GUI
        # if show_gui:
        #     cv2.imshow("Crop View", cropped)
        #     cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoOneShotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()