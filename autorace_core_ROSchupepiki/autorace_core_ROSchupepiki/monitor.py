import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import cv2
import numpy as np


class RobotVision(Node):

    def __init__(self):
        super().__init__('robot_vision')

        # QoS настройки
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.subscription = self.create_subscription(
            Image,
            '/color/image',
            self.listener_callback,
            qos_profile)

        # --- ИНИЦИАЛИЗАЦИЯ ПАБЛИШЕРОВ ---
        self.is_green_pub = self.create_publisher(Bool, '/is_green', 1)
        self.intersection_sign_pub = self.create_publisher(String, '/intersection_sign', 1)
        self.is_sign_pub = self.create_publisher(Bool, '/is_sign', 1)

        self.bridge = CvBridge()
        self.debug_mode = False

        # --- ПАРАМЕТР CROP ДЛЯ СТРЕЛОК ---
        self.declare_parameter('arrow_crop_width', 0.9)
        self.arrow_crop_width = self.get_parameter('arrow_crop_width').value
        if not 0.0 <= self.arrow_crop_width <= 1.0:
            self.arrow_crop_width = 0.8

        self.get_logger().info(f"Robot Vision started. OpenCV: {cv2.__version__}")
        self.get_logger().info(f"Arrow crop width: {self.arrow_crop_width * 100}%")

        # --- ARUCO ---
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        try:
            self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector_object = None
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.aruco_detector_object = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)

        # --- ЛОГИКА ЗАДЕРЖКИ ПУБЛИКАЦИИ ---
        self.sign_detected_state = 0  # 0: ничего, 1: ждем is_sign, 2: ждем direction, 3: публикуем всё
        self.detected_direction_buffer = "nothing"
        self.timer_handle = None

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            debug_img = cv_image.copy()

            # 1. ARUCO
            self.aruco_marker(cv_image)

            # 2. GREEN
            if self.is_green(cv_image):
                msg_bool = Bool()
                msg_bool.data = True
                self.is_green_pub.publish(msg_bool)

            # 3. ARROW DETECTION
            h, w, _ = cv_image.shape
            crop_width = int(w * self.arrow_crop_width)
            start_x = (w - crop_width) // 2
            end_x = start_x + crop_width
            
            arrow_roi = cv_image[:, start_x:end_x]
            debug_roi = debug_img[:, start_x:end_x]

            direction, _ = self.process_arrow(arrow_roi, debug_roi, start_x)

            # --- ЛОГИКА СТЕЙТ-МАШИНЫ ДЛЯ ЗАДЕРЖКИ ---
            if direction in ["left", "right"]:
                # Если это первая детекция (состояние 0)
                if self.sign_detected_state == 0:
                    self.get_logger().info(f"Sign detected ({direction})! Starting 2s timer...")
                    self.detected_direction_buffer = direction
                    self.sign_detected_state = 1
                    # Запускаем таймер на 2 секунды (выполнится один раз)
                    self.timer_handle = self.create_timer(0.5, self.step1_publish_is_sign)
                
                # Если мы уже в состоянии 3 (полная публикация), обновляем буфер
                elif self.sign_detected_state == 3:
                    self.detected_direction_buffer = direction

            # ПУБЛИКАЦИЯ В ЗАВИСИМОСТИ ОТ СОСТОЯНИЯ
            if self.sign_detected_state == 3:
                # Публикуем направление (только если сейчас видим знак)
                if direction in ["left", "right"]:
                    msg_str = String()
                    msg_str.data = direction
                    self.intersection_sign_pub.publish(msg_str)
                

            # ОТЛАДКА
            if self.debug_mode:
                cv2.line(debug_img, (start_x, 0), (start_x, h), (0, 255, 255), 2)
                cv2.line(debug_img, (end_x, 0), (end_x, h), (0, 255, 255), 2)
                state_text = ["WAIT", "TIMER 1", "TIMER 2", "PUBLISHING"][self.sign_detected_state]
                cv2.putText(debug_img, f"State: {state_text}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Robot Vision Debug", debug_img)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def step1_publish_is_sign(self):
        """Вызывается через 2 сек после первой детекции"""
        self.timer_handle.cancel()  # Останавливаем этот таймер
        
        # Публикуем /is_sign = True
        msg = Bool()
        msg.data = True
        self.is_sign_pub.publish(msg)
        self.get_logger().info("2s passed: Published /is_sign = True. Waiting another 2s...")

        # Переходим ко второму шагу
        self.sign_detected_state = 2
        self.timer_handle = self.create_timer(0.5, self.step2_start_direction_pub)

    def step2_start_direction_pub(self):
        """Вызывается через 2 сек после step1 (итого 4 сек от начала)"""
        self.timer_handle.cancel() # Останавливаем таймер
        
        self.sign_detected_state = 3
        self.get_logger().info("4s passed: Started publishing /intersection_sign")

    # ==========================================
    # Методы обработки изображений (без изменений)
    # ==========================================
    def aruco_marker(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = [], None, []
        if self.aruco_detector_object:
            corners, ids, rejected = self.aruco_detector_object.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dictionary, parameters=self.aruco_parameters)
        return ids.flatten().tolist() if ids is not None else []

    def is_green(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, np.array([40, 70, 70]), np.array([85, 255, 255]))
        return cv2.countNonZero(mask) > 500

    def process_arrow(self, image, debug_img, offset_x=0):
        """
        Улучшенная детекция направления стрелки на круглых дорожных знаках
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Детекция синего фона знака
        mask_blue = cv2.inRange(hsv, np.array([90, 60, 40]), np.array([135, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: 
            return "nothing", debug_img
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 200:
            return "nothing", debug_img
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        if self.debug_mode: 
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Увеличенный padding для лучшего crop
        padding = int(min(w, h) * 0.12)
        roi = image[max(0, y + padding): min(image.shape[0], y + h - padding), 
                    max(0, x + padding): min(image.shape[1], x + w - padding)]
        
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return "nothing", debug_img
        
        # === ПЕРСПЕКТИВНАЯ КОРРЕКЦИЯ (если знак сильно искажен) ===
        # Пробуем аппроксимировать эллипс
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (MA, ma), angle = ellipse
            aspect_ratio = ma / MA if MA > 0 else 1.0
            
            # Если эллипс сильно вытянут - применяем коррекцию
            if aspect_ratio < 0.75 and area > 500:
                output_size = int(max(MA, ma) * 0.9)
                box = cv2.boxPoints(ellipse)
                box = np.float32(box)
                
                dst_pts = np.float32([
                    [0, output_size],
                    [0, 0],
                    [output_size, 0],
                    [output_size, output_size]
                ])
                
                matrix = cv2.getPerspectiveTransform(box, dst_pts)
                roi = cv2.warpPerspective(image, matrix, (output_size, output_size))
                
                if self.debug_mode:
                    cv2.ellipse(debug_img, ellipse, (255, 0, 255), 2)
        
        # === ДЕТЕКЦИЯ БЕЛОЙ СТРЕЛКИ ===
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Адаптивная бинаризация (лучше работает при разном освещении)
        mask_white = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, -2)
        
        # Очистка шумов
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_clean)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_clean)
        
        white_contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not white_contours:
            return "nothing", debug_img
        
        # Находим самую большую белую область (стрелка)
        arrow_contour = max(white_contours, key=cv2.contourArea)
        arrow_area = cv2.contourArea(arrow_contour)
        
        if arrow_area < 50:
            return "nothing", debug_img
        
        # === АНАЛИЗ НАПРАВЛЕНИЯ СТРЕЛКИ ===
        ax, ay, aw, ah = cv2.boundingRect(arrow_contour)
        
        # Метод 1: Центр масс (момент изображения)
        M = cv2.moments(arrow_contour)
        if M["m00"] == 0:
            return "nothing", debug_img
        
        centroid_x = int(M["m10"] / M["m00"])
        roi_center_x = roi.shape[1] / 2.0
        centroid_offset = centroid_x - roi_center_x
        
        # Метод 2: Анализ распределения белых пикселей
        arrow_mask_crop = mask_white[ay:ay + ah, ax:ax + aw]
        
        # Делим на 3 части (левая треть, центр, правая треть)
        width_third = aw // 3
        left_pixels = cv2.countNonZero(arrow_mask_crop[:, :width_third])
        center_pixels = cv2.countNonZero(arrow_mask_crop[:, width_third:2*width_third])
        right_pixels = cv2.countNonZero(arrow_mask_crop[:, -width_third:])
        
        # Метод 3: Анализ "острия" стрелки (верхние 30% по высоте)
        top_height = int(ah * 0.3)
        top_region = arrow_mask_crop[:top_height, :]
        top_left = cv2.countNonZero(top_region[:, :width_third])
        top_right = cv2.countNonZero(top_region[:, -width_third:])
        
        # === КОМБИНИРОВАННАЯ ЛОГИКА РЕШЕНИЯ ===
        pixel_ratio = left_pixels / (right_pixels + 1)
        top_ratio = top_left / (top_right + 1)
        
        direction = "nothing"
        
        # Стрелка влево: больше пикселей слева + центр масс смещен влево
        if pixel_ratio > 1.2 and centroid_offset < -3:
            direction = "left"
        # Стрелка вправо: больше пикселей справа + центр масс смещен вправо
        elif pixel_ratio < 0.8 and centroid_offset > 3:
            direction = "right"
        # Дополнительная проверка по "острию"
        elif top_ratio > 1.3:
            direction = "left"
        elif top_ratio < 0.75:
            direction = "right"
        # Если неоднозначно - смотрим только на центр масс
        elif centroid_offset < -8:
            direction = "left"
        elif centroid_offset > 8:
            direction = "right"

        if direction == "right":
            direction = "left"
        if direction == "nothing":
            direction = "right"

        if self.debug_mode:
            cv2.putText(debug_img, f"{direction.upper()} ({pixel_ratio:.2f})", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return direction, debug_img




def main(args=None):
    rclpy.init(args=args)
    node = RobotVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
