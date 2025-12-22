import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage


class ImageProjection(Node):
    def __init__(self):
        super().__init__('image_projection')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('image_width', 848),
                ('image_height', 480),
                ('image_center_x', 424),
                ('image_center_y', 240),
                ('top_x', 72),
                ('top_y', 4),
                ('bottom_x', 115),
                ('bottom_y', 120),
                ('output_top', 0),
                ('output_bottom', 480),
                ('output_left', 148),
                ('output_right', 600),
                ('is_calibrating', False),
                ('enable_blur', True),
                ('blur_kernel', 5),
                ('input_format', 'raw'),  # 'raw' or 'compressed'
                ('output_format', 'raw'),  # 'raw' or 'compressed'
            ]
        )
        
        # Load initial parameter values
        self._load_parameters()
        
        # Setup subscriptions
        self._setup_subscriptions()
        
        # Setup publishers
        self._setup_publishers()
        
        # CV Bridge
        self.cvBridge = CvBridge()
        
        # Register parameter callback
        self.add_on_set_parameters_callback(self._on_parameters_changed)
        
        self.get_logger().info('ImageProjection node initialized')
    
    def _load_parameters(self):
        """Load all parameters into instance variables"""
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.image_center_x = self.get_parameter('image_center_x').value
        self.image_center_y = self.get_parameter('image_center_y').value
        
        self.top_x = self.get_parameter('top_x').value
        self.top_y = self.get_parameter('top_y').value
        self.bottom_x = self.get_parameter('bottom_x').value
        self.bottom_y = self.get_parameter('bottom_y').value
        
        self.output_top = self.get_parameter('output_top').value
        self.output_bottom = self.get_parameter('output_bottom').value
        self.output_left = self.get_parameter('output_left').value
        self.output_right = self.get_parameter('output_right').value
        
        self.is_calibrating = self.get_parameter('is_calibrating').value
        self.enable_blur = self.get_parameter('enable_blur').value
        self.blur_kernel = self.get_parameter('blur_kernel').value
        
        self.input_format = self.get_parameter('input_format').value
        self.output_format = self.get_parameter('output_format').value
    
    def _on_parameters_changed(self, params):
        """Callback for parameter changes"""
        for param in params:
            if param.name == 'top_x':
                self.top_x = param.value
            elif param.name == 'top_y':
                self.top_y = param.value
            elif param.name == 'bottom_x':
                self.bottom_x = param.value
            elif param.name == 'bottom_y':
                self.bottom_y = param.value
            elif param.name == 'is_calibrating':
                self.is_calibrating = param.value
            elif param.name == 'enable_blur':
                self.enable_blur = param.value
            elif param.name == 'blur_kernel':
                self.blur_kernel = param.value if param.value % 2 == 1 else param.value + 1
        
        return SetParametersResult(successful=True)
    
    def _setup_subscriptions(self):
        """Setup image subscriptions based on format"""
        topic = '/color/image/compressed' if self.input_format == 'compressed' else '/color/image'
        msg_type = CompressedImage if self.input_format == 'compressed' else Image
        
        self.sub_image = self.create_subscription(
            msg_type,
            topic,
            self.cb_image_projection,
            1
        )
    
    def _setup_publishers(self):
        """Setup image publishers based on format"""
        # Main projected image
        proj_topic = '/color/image_projected/compressed' if self.output_format == 'compressed' else '/color/image_projected'
        proj_msg_type = CompressedImage if self.output_format == 'compressed' else Image
        self.pub_projected = self.create_publisher(proj_msg_type, proj_topic, 1)
        
        # Calibration visualization (if enabled)
        if self.is_calibrating:
            calib_topic = '/color/image_calib/compressed' if self.output_format == 'compressed' else '/color/image_calib'
            calib_msg_type = CompressedImage if self.output_format == 'compressed' else Image
            self.pub_calib = self.create_publisher(calib_msg_type, calib_topic, 1)
    
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
    
    def _publish_image(self, publisher, cv_image, encoding='bgr8'):
        """Publish image based on output format"""
        try:
            if self.output_format == 'compressed':
                msg = self.cvBridge.cv2_to_compressed_imgmsg(cv_image, 'jpg')
            else:
                msg = self.cvBridge.cv2_to_imgmsg(cv_image, encoding)
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {e}')
    
    def cb_image_projection(self, msg):
        """Main image processing callback"""
        # Decode input image
        cv_image = self._decode_image(msg)
        if cv_image is None:
            return
        
        # Publish calibration visualization if in calibration mode
        if self.is_calibrating:
            self._publish_calibration_image(cv_image)
        
        # Apply Gaussian blur if enabled
        if self.enable_blur:
            cv_image = cv2.GaussianBlur(cv_image, (self.blur_kernel, self.blur_kernel), 0)
        
        # Perform homography transform
        cv_image_projected = self._apply_homography(cv_image)
        
        # Publish projected image
        self._publish_image(self.pub_projected, cv_image_projected)
    
    def _publish_calibration_image(self, cv_image):
        """Publish calibration overlay image"""
        cv_calib = cv_image.copy()
        
        # Draw trapezoid outline for calibration
        cx, cy = self.image_center_x, self.image_center_y
        
        pts = [
            (cx - self.top_x, cy - self.top_y),
            (cx + self.top_x, cy - self.top_y),
            (cx + self.bottom_x, cy + self.bottom_y),
            (cx - self.bottom_x, cy + self.bottom_y),
        ]
        
        # Draw lines
        cv2.line(cv_calib, pts[0], pts[1], (0, 0, 255), 2)  # top
        cv2.line(cv_calib, pts[1], pts[2], (0, 0, 255), 2)  # right
        cv2.line(cv_calib, pts[2], pts[3], (0, 0, 255), 2)  # bottom
        cv2.line(cv_calib, pts[3], pts[0], (0, 0, 255), 2)  # left
        
        # Draw center point
        cv2.circle(cv_calib, (cx, cy), 5, (0, 255, 0), -1)
        
        self._publish_image(self.pub_calib, cv_calib)
    
    def _apply_homography(self, cv_image):
        """Apply perspective transform (homography)"""
        cx, cy = self.image_center_x, self.image_center_y
        
        # Source points (trapezoid in image)
        pts_src = np.array([
            [cx - self.top_x, cy - self.top_y],
            [cx + self.top_x, cy - self.top_y],
            [cx + self.bottom_x, cy + self.bottom_y],
            [cx - self.bottom_x, cy + self.bottom_y]
        ], dtype=np.float32)
        
        # Destination points (rectangular bird's-eye view)
        pts_dst = np.array([
            [self.output_left, self.output_top],
            [self.output_right, self.output_top],
            [self.output_right, self.output_bottom],
            [self.output_left, self.output_bottom]
        ], dtype=np.float32)
        
        # Find homography matrix
        h, _ = cv2.findHomography(pts_src, pts_dst)
        
        if h is None:
            self.get_logger().warn('Homography calculation failed')
            return cv_image
        
        # Apply perspective transform
        output_width = self.image_width
        output_height = self.image_height
        cv_image_warped = cv2.warpPerspective(cv_image, h, (output_width, output_height))
        
        return cv_image_warped


def main(args=None):
    rclpy.init(args=args)
    node = ImageProjection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
