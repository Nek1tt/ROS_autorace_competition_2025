import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist


class LaneFollowerController(Node):
    def __init__(self):
        super().__init__('lane_follower')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                # PID Parameters
                ('pid_kp', 0.001),
                ('pid_ki', 0.0001),
                ('pid_kd', 0.0005),
                
                # Velocity parameters
                ('linear_velocity', 0.05),
                ('max_angular_velocity', 1.0),
                
                # Lane detection parameters - Yellow (Left)
                ('yellow_h_min', 15),
                ('yellow_h_max', 35),
                ('yellow_s_min', 100),
                ('yellow_s_max', 255),
                ('yellow_v_min', 100),
                ('yellow_v_max', 255),
                
                # Lane detection parameters - White (Right)
                ('white_h_min', 0),
                ('white_h_max', 180),
                ('white_s_min', 0),
                ('white_s_max', 30),
                ('white_v_min', 200),
                ('white_v_max', 255),
                
                # Image processing parameters
                ('input_format', 'raw'),  # 'raw' or 'compressed'
                ('blur_kernel', 5),
                ('canny_threshold1', 50),
                ('canny_threshold2', 150),
                ('hough_rho', 1),
                ('hough_theta', np.pi/180),
                ('hough_threshold', 30),
                ('hough_min_length', 30),
                ('hough_max_gap', 10),
                
                # Debugging
                ('publish_debug_image', True),
                ('verbose_logging', False),
                ('is_calibrating', False),
            ]
        )
        
        # Load initial parameters
        self._load_parameters()
        
        # PID state
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.last_time = self.get_clock().now()
        
        # Image subscription
        topic = '/color/image_projected/compressed' if self.input_format == 'compressed' else '/color/image_projected'
        msg_type = CompressedImage if self.input_format == 'compressed' else Image
        
        self.sub_image = self.create_subscription(
            msg_type,
            topic,
            self.cb_process_image,
            1
        )
        
        # Velocity publisher
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        
        # Debug image publisher
        if self.publish_debug_image:
            debug_topic = '/lane_debug/compressed' if self.input_format == 'compressed' else '/lane_debug'
            debug_msg_type = CompressedImage if self.input_format == 'compressed' else Image
            self.pub_debug = self.create_publisher(debug_msg_type, debug_topic, 1)
        
        # CV Bridge
        self.cvBridge = CvBridge()
        
        # Parameter callback
        self.add_on_set_parameters_callback(self._on_parameters_changed)
        
        self.get_logger().info('Lane Follower Controller initialized')
        self.get_logger().info(f'PID: Kp={self.pid_kp}, Ki={self.pid_ki}, Kd={self.pid_kd}')
    
    def _load_parameters(self):
        """Load all parameters"""
        # PID
        self.pid_kp = self.get_parameter('pid_kp').value
        self.pid_ki = self.get_parameter('pid_ki').value
        self.pid_kd = self.get_parameter('pid_kd').value
        
        # Velocity
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        
        # Yellow (Left) lane HSV range
        self.yellow_h_min = self.get_parameter('yellow_h_min').value
        self.yellow_h_max = self.get_parameter('yellow_h_max').value
        self.yellow_s_min = self.get_parameter('yellow_s_min').value
        self.yellow_s_max = self.get_parameter('yellow_s_max').value
        self.yellow_v_min = self.get_parameter('yellow_v_min').value
        self.yellow_v_max = self.get_parameter('yellow_v_max').value
        
        # White (Right) lane HSV range
        self.white_h_min = self.get_parameter('white_h_min').value
        self.white_h_max = self.get_parameter('white_h_max').value
        self.white_s_min = self.get_parameter('white_s_min').value
        self.white_s_max = self.get_parameter('white_s_max').value
        self.white_v_min = self.get_parameter('white_v_min').value
        self.white_v_max = self.get_parameter('white_v_max').value
        
        # Image processing
        self.input_format = self.get_parameter('input_format').value
        self.blur_kernel = self.get_parameter('blur_kernel').value
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.canny_threshold1 = self.get_parameter('canny_threshold1').value
        self.canny_threshold2 = self.get_parameter('canny_threshold2').value
        self.hough_rho = self.get_parameter('hough_rho').value
        self.hough_theta = self.get_parameter('hough_theta').value
        self.hough_threshold = int(self.get_parameter('hough_threshold').value)
        self.hough_min_length = int(self.get_parameter('hough_min_length').value)
        self.hough_max_gap = int(self.get_parameter('hough_max_gap').value)
        
        # Debug
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.verbose_logging = self.get_parameter('verbose_logging').value
        self.is_calibrating = self.get_parameter('is_calibrating').value
    
    def _on_parameters_changed(self, params):
        """Callback for parameter changes"""
        for param in params:
            if param.name.startswith('pid_'):
                if param.name == 'pid_kp':
                    self.pid_kp = param.value
                elif param.name == 'pid_ki':
                    self.pid_ki = param.value
                elif param.name == 'pid_kd':
                    self.pid_kd = param.value
                self.get_logger().info(f'Updated {param.name} = {param.value}')
            
            elif param.name == 'linear_velocity':
                self.linear_velocity = param.value
            elif param.name == 'max_angular_velocity':
                self.max_angular_velocity = param.value
            
            elif param.name.startswith('yellow_'):
                setattr(self, param.name, param.value)
            elif param.name.startswith('white_'):
                setattr(self, param.name, param.value)
            elif param.name == 'is_calibrating':
                self.is_calibrating = param.value
        
        return SetParametersResult(successful=True)
    
    def _decode_image(self, msg):
        """Decode image based on input format"""
        try:
            if self.input_format == 'compressed':
                np_arr = np.frombuffer(msg.data, np.uint8)
                return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                return self.cvBridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to decode image: {e}')
            return None
    
    def _publish_debug_image(self, cv_image):
        """Publish debug image"""
        try:
            if self.input_format == 'compressed':
                msg = self.cvBridge.cv2_to_compressed_imgmsg(cv_image, 'jpg')
            else:
                msg = self.cvBridge.cv2_to_imgmsg(cv_image, 'bgr8')
            self.pub_debug.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')
    
    def _detect_lanes(self, cv_image):
        """Detect yellow and white lanes"""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create masks for yellow and white
        lower_yellow = np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min])
        upper_yellow = np.array([self.yellow_h_max, self.yellow_s_max, self.yellow_v_max])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        lower_white = np.array([self.white_h_min, self.white_s_min, self.white_v_min])
        upper_white = np.array([self.white_h_max, self.white_s_max, self.white_v_max])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations to clean masks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        
        return mask_yellow, mask_white
    
    def _find_lane_lines(self, mask):
        """Find lane lines using Hough transform"""
        # Apply Canny edge detection
        edges = cv2.Canny(mask, self.canny_threshold1, self.canny_threshold2)
        
        # Apply Hough line transform
        lines = cv2.HoughLinesP(
            edges,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap
        )
        
        return lines
    
    def _get_lane_center(self, lines):
        """Get center x-coordinate of detected lane"""
        if lines is None or len(lines) == 0:
            return None
        
        x_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
        
        if not x_coords:
            return None
        
        return np.mean(x_coords)
    
    def _calculate_steering(self, left_center, right_center):
        """
        Calculate steering command using PID controller
        
        Args:
            left_center: x-coordinate of left (yellow) lane center
            right_center: x-coordinate of right (white) lane center
        
        Returns:
            angular_velocity (rad/s)
        """
        # If both lanes detected, target is midpoint
        if left_center is not None and right_center is not None:
            target_x = (left_center + right_center) / 2.0
            lane_width = right_center - left_center
        # If only left lane detected
        elif left_center is not None:
            target_x = left_center + 100  # Assume 200 pixel lane width
            lane_width = 200
        # If only right lane detected
        elif right_center is not None:
            target_x = right_center - 100
            lane_width = 200
        # If no lanes detected
        else:
            if self.verbose_logging:
                self.get_logger().warn('No lanes detected')
            return 0.0
        
        # Image center (bird's eye view)
        image_center = 424
        
        # Calculate error (positive = need to turn right, negative = need to turn left)
        error = (image_center - target_x)
        
        # Current time
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        if dt <= 0:
            dt = 0.016  # Assume 60 Hz if dt is invalid
        
        # PID calculation
        # Proportional
        p_term = self.pid_kp * error
        
        # Integral (with anti-windup)
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -100, 100)  # Anti-windup
        i_term = self.pid_ki * self.integral_error
        
        # Derivative
        d_term = self.pid_kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        # Total control output
        angular_velocity = p_term + i_term + d_term
        angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
        
        if self.verbose_logging:
            self.get_logger().debug(
                f'Error: {error:.1f}, P: {p_term:.3f}, I: {i_term:.3f}, D: {d_term:.3f}, '
                f'ω: {angular_velocity:.3f}'
            )
        
        return angular_velocity
    
    def cb_process_image(self, msg):
        """Main image processing callback"""
        # Decode image
        cv_image = self._decode_image(msg)
        if cv_image is None:
            return
        
        debug_image = cv_image.copy() if self.publish_debug_image else None
        
        # Detect lanes
        mask_yellow, mask_white = self._detect_lanes(cv_image)
        
        # Find lane lines
        lines_yellow = self._find_lane_lines(mask_yellow)
        lines_white = self._find_lane_lines(mask_white)
        
        # Get lane centers
        left_center = self._get_lane_center(lines_yellow)
        right_center = self._get_lane_center(lines_white)
        
        # Calculate steering command
        angular_velocity = self._calculate_steering(left_center, right_center)
        
        # Create and publish velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = self.linear_velocity
        cmd_vel.angular.z = angular_velocity
        if self.is_calibrating:
            # Publish zero velocity to stop robot
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd_vel)
        else:
            self.pub_cmd_vel.publish(cmd_vel)
        
        # Publish debug image
        if self.publish_debug_image:
            # Draw detected lines
            if lines_yellow is not None:
                for line in lines_yellow:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow
            
            if lines_white is not None:
                for line in lines_white:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(debug_image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White
            
            # Draw lane centers
            if left_center is not None:
                cv2.circle(debug_image, (int(left_center), 240), 5, (0, 255, 255), -1)
            if right_center is not None:
                cv2.circle(debug_image, (int(right_center), 240), 5, (255, 255, 255), -1)
            
            # Draw target center line
            if left_center is not None and right_center is not None:
                target_x = int((left_center + right_center) / 2)
                cv2.line(debug_image, (target_x, 0), (target_x, 480), (0, 255, 0), 2)
                cv2.circle(debug_image, (target_x, 240), 10, (0, 255, 0), 2)
            
            # Add text info
            cv2.putText(debug_image, f'v={self.linear_velocity:.2f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_image, f'w={angular_velocity:.3f}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            self._publish_debug_image(debug_image)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot
        cmd_vel = Twist()
        node.pub_cmd_vel.publish(cmd_vel)
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    