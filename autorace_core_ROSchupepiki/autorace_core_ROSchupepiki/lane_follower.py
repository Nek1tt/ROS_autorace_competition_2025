import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque
from std_msgs.msg import Float64, Bool, String, Float32


class LaneFollower(Node):
    def __init__(self):
        super().__init__('LaneFollower')

        # Параметры
        self.declare_parameter('max_len', 5)
        self.declare_parameter('start_speed', 0.3)
        self.declare_parameter('start_offset', 0.0)
        
        # PID коэффициенты
        self.declare_parameter('Kp_ang', 0.008)
        self.declare_parameter('Ki_ang', 0.0)
        self.declare_parameter('Kd_ang', 0.002)
        self.declare_parameter('Kp_vel', 0.0008)
        self.declare_parameter('Ki_vel', 0.0)
        self.declare_parameter('Kd_vel', 0.0)
        self.tunnel_mission = False

        maxlen = self.get_parameter('max_len').value
        self.instant_errors = deque([0], maxlen=maxlen)
        self.error_differences = deque(maxlen=maxlen)

        self.max_vel = self.get_parameter('start_speed').value
        self.offset = self.get_parameter('start_offset').value
        
        # Флаги состояний
        self.is_green = False
        self.is_sign = False
        self.is_cone = False
        self.is_cone_mission = False
        self.intersection_sign = None
        
        # ОДОМЕТРИЯ И МАНЕВРЫ
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.cur_yaw = 0.0
        
        # State Machine маневров
        self.maneuver_state = "IDLE"
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0
        self.target_val = 0.0
        self.move_speed = 0.0
        self.sequence_step = 0
        
        # Поворот на перекрестке (старая логика)
        self.start_angle_sign = None
        self.target_turn_angle_sign = 25.0
        self.is_turning_sign = False
        
        self.sign_accepted = False
        self.stop_counter = 0
        self.STOP_DURATION = 100

        self.is_first_call = True
        self.true_center = None
        self.roi_height_ratio = 0.4 
        self.cv_bridge = CvBridge()
        self.tunnel_cmd_vel_data = None
        self.is_tunnel = False
        
        # ФЛАГИ ФИНИША
        self.finish_detected = False
        self.finish_maneuver_active = False # Активен ли проезд 2м
        self.finish_flag = False            # Полная остановка

        # Подписки
        self.img_proj_sub = self.create_subscription(Image, '/color/image_projected', self.image_processing, 1)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.is_green_sub = self.create_subscription(Bool, '/is_green', self.is_green_processing, 1)
        self.is_sign_sub = self.create_subscription(Bool, '/is_sign', self.is_sign_processing, 1)
        self.is_cone_sub = self.create_subscription(Bool, '/is_cone', self.is_cone_processing, 1)
        self.intersection_sign_sub = self.create_subscription(String, '/intersection_sign', self.intersection_sign_processing, 1)
        self.is_tunnel_sub = self.create_subscription(Bool, '/is_tunnel', self.is_tunnel_processing, 1)
        self.finish_sub = self.create_subscription(Bool, '/is_finish', self.finish_callback, 1)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.image_pub = self.create_publisher(Image, '/color/detectline', 1)
        self.finish_pub = self.create_publisher(String, '/robot_finish', 1)


    def finish_callback(self, msg):
        """Если увидели финиш (ArUco), начинаем проезд 2 метров"""
        if msg.data and not self.finish_detected and self.tunnel_mission:
            self.get_logger().info(f'Finish detected! Starting 2m drive...')
            self.finish_detected = True
            self.finish_maneuver_active = True
            # Запускаем движение на 2 метра через существующий механизм маневров
            self.start_move_forward(0.8, speed=0.3)
    

    def is_tunnel_processing(self, msg):
        self.is_tunnel = msg.data
        if self.is_tunnel:
            self.tunnel_mission = True
        	

    # ОДОМЕТРИЯ
    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.cur_yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    # СИСТЕМА МАНЕВРОВ
    def start_move_forward(self, distance, speed=0.2):
        self.maneuver_state = "MOVING"
        self.start_x = self.odom_x
        self.start_y = self.odom_y
        self.target_val = distance
        self.move_speed = speed
        self.get_logger().info(f"--> START MOVE: {distance}m at {speed}m/s")

    def start_turn(self, angle_deg, speed=0.8):
        self.maneuver_state = "TURNING"
        self.start_yaw = self.cur_yaw
        self.target_val = abs(angle_deg)
        self.move_speed = speed if angle_deg > 0 else -speed
        self.get_logger().info(f"--> START TURN: {angle_deg} deg")

    def execute_maneuver(self):
        """Возвращает True, если робот занят маневром"""
        if self.maneuver_state == "IDLE":
            return False

        twist = Twist()

        # ДВИЖЕНИЕ ПРЯМО
        if self.maneuver_state == "MOVING":
            dx = self.odom_x - self.start_x
            dy = self.odom_y - self.start_y
            dist = np.hypot(dx, dy)

            if dist < self.target_val:
                twist.linear.x = self.move_speed
                self.cmd_vel_pub.publish(twist)
                return True 
            else:
                self.stop_robot()
                self.maneuver_state = "IDLE"
                self.get_logger().info("DONE MOVE")
                
                # Если это был финишный проезд
                if self.finish_maneuver_active:
                    self.finish_maneuver_active = False
                    self.finish_flag = True  # Включаем полную остановку
                    self.get_logger().info("FINISH MANEUVER COMPLETE -> STOP")
                else:
                    self.next_sequence_step() # Обычный сценарий (конусы)
                return True

        # ПОВОРОТ
        elif self.maneuver_state == "TURNING":
            diff = self.cur_yaw - self.start_yaw
            if diff < -180: diff += 360
            if diff > 180: diff -= 360
            
            if abs(diff) < self.target_val:
                twist.angular.z = self.move_speed
                self.cmd_vel_pub.publish(twist)
                return True
            else:
                self.stop_robot()
                self.maneuver_state = "IDLE"
                self.get_logger().info("DONE TURN")
                self.next_sequence_step()
                return True
        
        return False

    def next_sequence_step(self):
        # СЦЕНАРИЙ КОНУСОВ
        if self.sequence_step == 1: 
            self.sequence_step = 2
            self.start_turn(65.0)
        elif self.sequence_step == 2:
            self.sequence_step = 3
            self.start_move_forward(0.45, speed=0.2)
        elif self.sequence_step == 3: 
            self.sequence_step = 4
            self.start_turn(20.0)
        elif self.sequence_step == 4:
            self.sequence_step = 5
            self.start_move_forward(0.3, speed=0.2) 
        elif self.sequence_step == 5: 
            self.sequence_step = 6
            self.start_turn(-80.0)
        elif self.sequence_step == 6:
            self.sequence_step = 7
            self.start_move_forward(0.32, speed=0.2) 
        elif self.sequence_step == 7: 
            self.sequence_step = 8
            self.start_turn(80.0)
        elif self.sequence_step == 8:
            self.sequence_step = 9
            self.start_move_forward(0.32, speed=0.2) 
        elif self.sequence_step == 9:
            self.sequence_step = 0
            self.is_cone = False
            self.get_logger().info("=== CONE SEQUENCE FINISHED ===")

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    # CALLBACKS
    def is_green_processing(self, msg): self.is_green = msg.data
    
    def is_cone_processing(self, msg):
        if msg.data and not self.is_cone and not self.is_cone_mission:
            self.is_cone = True
            self.is_cone_mission = True
            self.sequence_step = 1
            self.get_logger().info("!!! CONES DETECTED !!!")
            self.start_move_forward(0.75, speed=0.2)

    def is_sign_processing(self, msg):
        if msg.data and self.stop_counter == 0:
            self.stop_counter = self.STOP_DURATION
            self.get_logger().info("Sign detected! Stopping...")

    def intersection_sign_processing(self, msg):
        if not self.sign_accepted:
            self.intersection_sign = msg.data
            self.sign_accepted = True

    # ОСНОВНАЯ ЛОГИКА
    def image_processing(self, msg):
        # 0. ФИНИШНАЯ ОСТАНОВКА (самый высокий приоритет)
        if self.finish_flag:
            self.cmd_vel_pub.publish(Twist())
            msg_str = String()
            msg_str.data = "ROSchupepiki"
            self.finish_pub.publish(msg_str)            
            return

        # 1. ВЫПОЛНЕНИЕ МАНЕВРОВ (Конусы или Финишный проезд)
        if self.execute_maneuver():
            return

        # 2. ТУННЕЛЬ
        if self.is_tunnel:
            twist = Twist()
            twist.linear.x = 0.5
            twist.angular.z = 0.4
            self.cmd_vel_pub.publish(twist) 
            self.get_logger().info("Tunnel riding")
            return

        # 3. НЕ ЗЕЛЕНЫЙ СВЕТ
        if not self.is_green:
            self.cmd_vel_pub.publish(Twist())
            return

        # 4. ОСТАНОВКА ПЕРЕД ЗНАКОМ
        if self.stop_counter > 0:
            self.cmd_vel_pub.publish(Twist())
            self.stop_counter -= 1
            if self.stop_counter == 0 and self.intersection_sign == 'left':
                self.is_turning_sign = True
                self.start_angle_sign = self.cur_yaw
                self.intersection_sign = None
                self.get_logger().info("Starting LEFT turn on intersection (25°)...")
            return

        # 5. ПОВОРОТ НА ПЕРЕКРЕСТКЕ
        if self.is_turning_sign:
            diff = self.cur_yaw - self.start_angle_sign
            if diff < -180: diff += 360
            if diff > 180: diff -= 360
            if abs(diff) < self.target_turn_angle_sign:
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 1.0
                self.cmd_vel_pub.publish(twist)
            else:
                self.cmd_vel_pub.publish(Twist())
                self.is_turning_sign = False
                self.get_logger().info("Intersection turn finished")
            return

        # 6. СЛЕДОВАНИЕ ПО ЛИНИИ
        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        h, w = image.shape[:2]
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        yellow_mask = cv2.inRange(hsv_image, (20, 100, 100), (30, 255, 255))
        yellow_mask = cv2.blur(yellow_mask, (3, 3))
        _, yellow_mask = cv2.threshold(yellow_mask, 1, 255, cv2.THRESH_BINARY)

        white_mask = cv2.inRange(hsv_image, (0, 0, 230), (255, 0, 255))
        white_mask = cv2.blur(white_mask, (3, 3))
        _, white_mask = cv2.threshold(white_mask, 1, 255, cv2.THRESH_BINARY)

        roi_h = int(h * self.roi_height_ratio)
        roi_start_y = h - roi_h 

        yellow_roi = yellow_mask[roi_start_y:h, 0:w]
        white_roi = white_mask[roi_start_y:h, 0:w]

        M_yellow = cv2.moments(yellow_roi, binaryImage=True)
        M_white = cv2.moments(white_roi, binaryImage=True)

        yellow_center_x = 0 if M_yellow['m00'] == 0 else int(M_yellow['m10'] // M_yellow['m00'])
        white_center_x = w if M_white['m00'] == 0 else int(M_white['m10'] // M_white['m00'])

        if white_center_x < yellow_center_x:
            white_center_x = w

        center_lane = (yellow_center_x + white_center_x) / 2.0
        self.compute_and_publish_velocity(center_lane)

        # Визуализация
        cv2.line(image, (0, roi_start_y), (w, roi_start_y), (0, 0, 255), 2)
        draw_y = roi_start_y + (roi_h // 2)
        if M_yellow['m00'] > 0:
            cv2.circle(image, (int(yellow_center_x), draw_y), 8, (0, 255, 255), -1)
        if M_white['m00'] > 0:
            cv2.circle(image, (int(white_center_x), draw_y), 8, (255, 255, 0), -1)
        cv2.circle(image, (int(center_lane), draw_y), 6, (0, 0, 255), -1)
        self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(image, "bgr8"))


    def compute_and_publish_velocity(self, cur_center):
        if self.is_first_call:
            self.is_first_call = False
            self.true_center = cur_center
            return

        Kp_ang = self.get_parameter('Kp_ang').value
        Ki_ang = self.get_parameter('Ki_ang').value
        Kd_ang = self.get_parameter('Kd_ang').value
        Kp_vel = self.get_parameter('Kp_vel').value
        Ki_vel = self.get_parameter('Ki_vel').value
        Kd_vel = self.get_parameter('Kd_vel').value
        
        self.max_vel = self.get_parameter('start_speed').value
        self.offset = self.get_parameter('start_offset').value

        error = (self.true_center + self.offset) - cur_center
        self.error_differences.append(error - self.instant_errors[-1])
        self.instant_errors.append(error)

        twist = Twist()   
        twist.linear.x = self.max_vel - (Kp_vel * np.abs(error) + 
                                         Ki_vel * np.sum(np.abs(self.instant_errors)) + 
                                         Kd_vel * np.sum(np.abs(self.error_differences)))
        twist.angular.z = (Kp_ang * error + 
                           Ki_ang * np.sum(self.instant_errors) + 
                           Kd_ang * np.sum(self.error_differences))

        self.cmd_vel_pub.publish(twist)


def main():
    rclpy.init()
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
