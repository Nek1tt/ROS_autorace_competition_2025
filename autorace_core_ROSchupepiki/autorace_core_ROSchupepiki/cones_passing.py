import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque
from std_msgs.msg import Float64, Bool


class LaneAndConeFollower(Node):
    def __init__(self):
        super().__init__('LaneAndConeFollower')

        # === ПАРАМЕТРЫ ===
        self.declare_parameter('max_len', 5)
        self.declare_parameter('start_speed', 0.23)
        self.declare_parameter('start_offset', 0.0)

        # PID
        self.declare_parameter('Kp_ang', 0.0078)
        self.declare_parameter('Ki_ang', 0.0)
        self.declare_parameter('Kd_ang', 0.001)

        # КОНУСЫ
        self.declare_parameter('cone_h_min', 0)
        self.declare_parameter('cone_h_max', 15)
        self.declare_parameter('cone_s_min', 100)
        self.declare_parameter('cone_v_min', 100)

        self.declare_parameter('cone_ignore_margin', 0.33)
        self.declare_parameter('avoid_offset_pixels', 175.0)

        # === ПЕРЕМЕННЫЕ ===
        maxlen = self.get_parameter('max_len').value
        self.instant_errors = deque([0], maxlen=maxlen)
        self.error_differences = deque(maxlen=maxlen)

        self.is_green = False
        self.is_first_call = True
        self.true_center = None

        self.current_avoidance_offset = 0.0

        # состояние конусов
        self.is_cone = False          # что сейчас публикуем
        self.cone_seen_frames = 0     # подряд кадров с конусами
        self.cone_lost_frames = 0     # подряд кадров без конусов

        # пороги гистерезиса (в кадрах)
        self.frames_to_set_true = 3   # сколько кадров подряд надо видеть, чтобы включить is_cone=True
        self.frames_to_set_false = 10 # сколько кадров подряд НЕ видеть, чтобы выключить

        self.start_time = None
        self.cv_bridge = CvBridge()

        # === ПОДПИСЧИКИ ===
        self.img_proj_sub = self.create_subscription(
            Image, '/color/image_projected', self.lines_processing, 1)

        self.img_raw_sub = self.create_subscription(
            Image, '/color/image', self.cones_processing, 1)

        self.is_green_sub = self.create_subscription(
            Bool, '/is_green', self.is_green_processing, 1)

        # === ПУБЛИКАТОРЫ ===
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_cones', 1)
        self.debug_lines_pub = self.create_publisher(Image, '/debug/lines', 1)
        self.debug_cones_pub = self.create_publisher(Image, '/debug/cones', 1)
        self.is_cone_pub = self.create_publisher(Bool, '/is_cone', 1)

        def publish_is_cone(self, new_state: bool):
            if new_state == self.is_cone:
                return
            self.is_cone = new_state
            msg = Bool()
            msg.data = self.is_cone
            self.is_cone_pub.publish(msg)
            self.get_logger().info(f"[CONE STATE] is_cone = {self.is_cone}")


    def is_green_processing(self, msg):
        self.is_green = msg.data

    # ==========================================
    # ЛОГИКА 1: ОБРАБОТКА КОНУСОВ
    # ==========================================
    def cones_processing(self, msg):
        if not self.is_green:
            return

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        h, w = cv_image.shape[:2]

        roi_top = int(h * 0.4)
        roi = cv_image[roi_top:h, 0:w]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        c_h_min = self.get_parameter('cone_h_min').value
        c_h_max = self.get_parameter('cone_h_max').value
        c_s_min = self.get_parameter('cone_s_min').value
        c_v_min = self.get_parameter('cone_v_min').value

        cone_mask = cv2.inRange(hsv, (c_h_min, c_s_min, c_v_min), (c_h_max, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        cone_mask = cv2.dilate(cone_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(cone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detected_cone_x = -1
        max_area = 0
        valid_cones_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                valid_cones_count += 1
                if area > max_area:
                    max_area = area
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        detected_cone_x = int(M['m10'] / M['m00'])

        # ---------- ОБНОВЛЕНИЕ is_cone С ГИСТЕРЕЗИСОМ ----------
        # ---------- ОБНОВЛЕНИЕ is_cone ----------
        if valid_cones_count > 0:
            # как только увидели хотя бы один конус – сразу True
            self.cone_lost_frames = 0
            self.publish_is_cone(True)
        else:
            # не видим ни одного – начинаем считать кадры "потери"
            self.cone_lost_frames += 1
            if self.cone_lost_frames >= self.frames_to_set_false:
                self.publish_is_cone(False)
        # ----------------------------------------


        # === ЗАПУСК ТАЙМЕРА (ЕСЛИ ВИДИМ ГРУППУ) ===
        if self.start_time is None:
            if valid_cones_count >= 2:
                self.start_time = self.get_clock().now()

        # === ПРОВЕРКА ВРЕМЕНИ И ПРИНУДИТЕЛЬНЫЙ ПОВОРОТ ===
        is_forced_mode = False
        current_seconds = 0.0

        if self.start_time is not None:
            current_time = self.get_clock().now()
            duration = current_time - self.start_time
            current_seconds = duration.nanoseconds / 1e9

            # ========================================================
            # НАСТРОЙКИ ВРЕМЕННОГО ИНТЕРВАЛА
            # ========================================================
            start_force_time = 10.0  # Когда начинаем поворачивать (сек)
            forced_duration = 2.5  # Сколько секунд длится поворот (сек) <--- НАСТРАИВАЙ ТУТ
            forced_left_strength = 100.0  # Сила поворота влево
            # ========================================================

            # Если время внутри окна [10.0 ... 12.5]
            if start_force_time < current_seconds < (start_force_time + forced_duration):
                is_forced_mode = True

                # Принудительно влево
                self.current_avoidance_offset = -forced_left_strength

                cv2.putText(cv_image, f"!!! FORCE LEFT ({forced_duration}s) !!!", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # === ОБЫЧНАЯ ЛОГИКА (ЕСЛИ НЕ В РЕЖИМЕ FORCE) ===
        # Сюда заходим ДО 10 сек и ПОСЛЕ (10 + forced_duration) сек
        if not is_forced_mode:
            margin_percent = self.get_parameter('cone_ignore_margin').value
            min_valid_x = int(w * margin_percent)
            max_valid_x = int(w * (1.0 - margin_percent))

            cv2.line(roi, (min_valid_x, 0), (min_valid_x, h), (255, 0, 0), 2)
            cv2.line(roi, (max_valid_x, 0), (max_valid_x, h), (255, 0, 0), 2)

            strength_pixels = self.get_parameter('avoid_offset_pixels').value
            self.current_avoidance_offset = 0.0

            if detected_cone_x != -1:
                if min_valid_x < detected_cone_x < max_valid_x:
                    cv2.circle(roi, (detected_cone_x, 50), 10, (0, 255, 0), -1)
                    raw_center = w / 2

                    if detected_cone_x < raw_center:
                        self.current_avoidance_offset = strength_pixels + 500
                        cv2.putText(roi, "GO RIGHT >>", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        self.current_avoidance_offset = -strength_pixels
                        cv2.putText(roi, "<< GO LEFT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.circle(roi, (detected_cone_x, 50), 8, (255, 0, 0), -1)
                    cv2.putText(roi, "IGNORED", (detected_cone_x - 40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                                1)

        # Возвращаем ROI и рисуем таймер
        cv_image[roi_top:h, 0:w] = roi

        if self.start_time is not None:
            text = f"TIMER: {current_seconds:.2f} s"
            # Красный цвет во время форсажа, Зеленый в остальное время
            color = (0, 0, 255) if is_forced_mode else (0, 255, 0)
            cv2.putText(cv_image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        else:
            cv2.putText(cv_image, "WAITING FOR GROUP...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        self.debug_cones_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    # ==========================================
    # ЛОГИКА 2: ДВИЖЕНИЕ ПО ЛИНИЯМ
    # ==========================================
    def lines_processing(self, msg):
        if not self.is_green:
            return

        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            return

        h, w = image.shape[:2]
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        roi_h = int(h * 0.4)
        roi_start_y = h - roi_h
        hsv_roi = hsv_image[roi_start_y:h, 0:w]

        yellow_mask = cv2.inRange(hsv_roi, (20, 100, 100), (30, 255, 255))
        white_mask = cv2.inRange(hsv_roi, (0, 0, 230), (255, 0, 255))

        _, yellow_mask = cv2.threshold(cv2.blur(yellow_mask, (3, 3)), 1, 255, cv2.THRESH_BINARY)
        _, white_mask = cv2.threshold(cv2.blur(white_mask, (3, 3)), 1, 255, cv2.THRESH_BINARY)

        M_yellow = cv2.moments(yellow_mask)
        M_white = cv2.moments(white_mask)

        yellow_center_x = 0 if M_yellow['m00'] == 0 else int(M_yellow['m10'] // M_yellow['m00'])
        white_center_x = w if M_white['m00'] == 0 else int(M_white['m10'] // M_white['m00'])

        if white_center_x < yellow_center_x:
            white_center_x = w

        lane_center = (yellow_center_x + white_center_x) / 2.0
        final_target = lane_center + self.current_avoidance_offset

        draw_y = roi_start_y + (roi_h // 2)
        if M_yellow['m00'] > 0: cv2.circle(image, (int(yellow_center_x), draw_y), 5, (0, 255, 255), -1)
        if M_white['m00'] > 0: cv2.circle(image, (int(white_center_x), draw_y), 5, (255, 255, 0), -1)

        cv2.circle(image, (int(final_target), draw_y), 8, (0, 0, 255), -1)

        if self.current_avoidance_offset != 0:
            cv2.putText(image, f"AVOID: {self.current_avoidance_offset}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.debug_lines_pub.publish(self.cv_bridge.cv2_to_imgmsg(image, "bgr8"))
        self.compute_and_publish_velocity(final_target)

    def compute_and_publish_velocity(self, cur_center):
        if self.is_first_call:
            self.is_first_call = False
            self.true_center = cur_center
            return

        Kp_ang = self.get_parameter('Kp_ang').value
        Ki_ang = self.get_parameter('Ki_ang').value
        Kd_ang = self.get_parameter('Kd_ang').value

        speed = self.get_parameter('start_speed').value
        offset = self.get_parameter('start_offset').value

        error = (self.true_center + offset) - cur_center

        self.error_differences.append(error - self.instant_errors[-1])
        self.instant_errors.append(error)

        twist = Twist()

        if self.current_avoidance_offset != 0:
            twist.linear.x = speed * 0.45
        else:
            twist.linear.x = speed

        twist.angular.z = (Kp_ang * error +
                           Ki_ang * np.sum(self.instant_errors) +
                           Kd_ang * np.sum(self.error_differences))

        self.cmd_vel_pub.publish(twist)


def main():
    rclpy.init()
    node = LaneAndConeFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()