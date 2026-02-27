import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class AllCameraViewer(Node):
    def __init__(self):
        super().__init__('all_camera_viewer')
        
        # إعدادات QoS لتتوافق مع أي كاميرا
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.bridge = CvBridge()

        self.get_logger().info('🕵️‍♂️ Scanning for ALL cameras...')

        # --- القائمة الشاملة للكاميرات ---

        # 1. الكاميرا الأمامية الذكية (RealSense RGB)
        self.create_subscription(Image, '/camera/color_image/compressed', self.save_realsense, qos_profile)



    def save_image(self, msg, filename, label):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # كتابة الاسم على الصورة
            cv2.putText(img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(f'/workspaces/isaac_ros-dev/{filename}', img)
            self.get_logger().info(f'📸 Captured: {label}')
        except Exception: pass

    # --- دوال المعالجة لكل كاميرا ---

    def save_realsense(self, msg):
        self.save_image(msg, '/camera/color_image/compressed', "RealSense Front")


    def save_depth(self, msg):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            # تحويل العمق لصورة مرئية
            depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            depth_visual = np.uint8(depth_normalized)
            depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
            
            cv2.putText(depth_colored, "Depth Map", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite('/workspaces/isaac_ros-dev/cam_5_depth.jpg', depth_colored)
            self.get_logger().info('📸 Captured: Depth Map')
        except Exception: pass

def main(args=None):
    rclpy.init(args=args)
    node = AllCameraViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()