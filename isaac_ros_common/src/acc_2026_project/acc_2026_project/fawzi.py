#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from qcar2_interfaces.msg import MotorCommands
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import threading
from pal.utilities.vision import Camera2D

# ------------------------------------------------------------
#  PID Controller class (من الكود الثاني)
# ------------------------------------------------------------
class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=None, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, error, current_time):
        if self.prev_time is None:
            dt = 0.01
        else:
            dt = current_time - self.prev_time
            if dt <= 0.0:
                dt = 0.01

        p_term = self.kp * error
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        self.prev_error = error
        self.prev_time = current_time
        return output

# ------------------------------------------------------------
#  Main Node
# ------------------------------------------------------------
class TrafficAwareBlackRoadFollower(Node):
    def __init__(self):
        super().__init__('traffic_aware_black_road_follower')

        # Parameters (القيم من الكود الثاني مع تعديل السرعة لـ 0.45)
        self.declare_parameter('speed', 0.45)
        self.declare_parameter('kp', 0.004)
        self.declare_parameter('ki', 0.0002)
        self.declare_parameter('kd', 0.001)
        self.declare_parameter('integral_limit', 0.5)
        self.declare_parameter('output_limit', 0.8)
        self.declare_parameter('max_angle', 0.8)
        self.declare_parameter('show_debug', True)
        self.declare_parameter('safe_distance', 0.5)
        self.declare_parameter('danger_distance', 0.25)
        self.declare_parameter('wall_gain', 0.8)

        # Camera parameters for myCam4 (المسرب الأيسر)
        self.declare_parameter('camera_id', '3@tcpip://localhost:18964')
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('frame_rate', 30)

        self.base_speed = self.get_parameter('speed').value
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.integral_limit = self.get_parameter('integral_limit').value
        self.output_limit = self.get_parameter('output_limit').value
        self.max_angle = self.get_parameter('max_angle').value
        self.show_debug = self.get_parameter('show_debug').value
        self.safe_distance = self.get_parameter('safe_distance').value
        self.danger_distance = self.get_parameter('danger_distance').value
        self.wall_gain = self.get_parameter('wall_gain').value

        camera_id = self.get_parameter('camera_id').value
        frame_width = self.get_parameter('frame_width').value
        frame_height = self.get_parameter('frame_height').value
        frame_rate = self.get_parameter('frame_rate').value

        # Publishers & Subscribers
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pub_cmd = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)

        # Subscription to the second camera (للإشارات)
        self.sub_sign_cam = self.create_subscription(
            Image, '/camera/color_image', self.sign_image_callback, 10)
        self.bridge = CvBridge()

        # Initialize myCam4 (لمتابعة المسار)
        self.myCam4 = Camera2D(
            cameraId=camera_id,
            frameWidth=frame_width,
            frameHeight=frame_height,
            frameRate=frame_rate
        )
        self.get_logger().info(f'📷 Camera myCam4 initialized with ID: {camera_id} (للمسرب الأيسر)')

        # PID controller
        self.pid_steering = PIDController(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            integral_limit=self.integral_limit,
            output_limit=self.output_limit
        )

        # State machine (تماماً من الكود الثاني)
        self.state = "DRIVING"
        self.state_start_time = 0.0
        self.stop_duration = 3.0
        self.yield_max_wait = 5.0
        self.last_stop_time = 0.0
        self.stop_cooldown = 5.0

        # Lidar distances
        self.left_dist = 100.0
        self.right_dist = 100.0
        self.front_dist = 100.0

        # Control variables
        self.lock = threading.Lock()
        self.current_steering = 0.0
        self.current_speed = 0.0

        # Detected sign (من كاميرا الإشارات)
        self.detected_sign = None
        self.last_sign_frame = None
        self.sign_lock = threading.Lock()

        # Camera status
        self.camera_ok = False

        # Timer
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.create_timer(2.0, self.check_health)

        self.get_logger().info('🚗 متابعة المسرب الأيسر مع إشارات (stop/yield) تماماً كما في الكود الثاني. السرعة=0.45')

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------
    def check_health(self):
        if not self.camera_ok:
            self.get_logger().warn('⏳ في انتظار صور myCam4...')
        if self.last_sign_frame is None:
            self.get_logger().warn('⏳ في انتظار topic /camera/color_image...')

    # --------------------------------------------------------
    # Timer callback
    # --------------------------------------------------------
    def timer_callback(self):
        # قراءة صورة من myCam4
        flag = self.myCam4.read()
        if flag:
            self.camera_ok = True
            frame = self.myCam4.imageData

            with self.sign_lock:
                current_sign = self.detected_sign

            steering, speed, lane_frame = self.process_lane_and_sign(frame, current_sign)

            with self.lock:
                self.current_steering = steering
                self.current_speed = speed

            if self.show_debug and lane_frame is not None:
                cv2.imshow("المسرب الأيسر", lane_frame)
                cv2.waitKey(1)
        else:
            self.camera_ok = False
            self.get_logger().warn('⚠️ فشل قراءة الصورة من myCam4', throttle_duration_sec=2.0)

        # عرض صورة الإشارات
        with self.sign_lock:
            if self.last_sign_frame is not None:
                sign_disp = self.last_sign_frame.copy()
                if self.detected_sign:
                    cv2.putText(sign_disp, f"الإشارة: {self.detected_sign}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("كشف الإشارات", sign_disp)
                cv2.waitKey(1)

        # نشر الأوامر
        with self.lock:
            steering = self.current_steering
            speed = self.current_speed

        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(steering), float(speed)]
        self.pub_cmd.publish(msg)

    # --------------------------------------------------------
    # معالجة المسار والإشارات (باستخدام دالة المسرب الأيسر)
    # --------------------------------------------------------
    def process_lane_and_sign(self, frame, sign):
        # 1. كشف الخطأ للمسرب الأيسر (معدل)
        lane_error, road_mask, white_mask, leftmost_point = self.detect_left_lane_error(frame)

        # 2. الوقت الحالي
        current_time = self.get_clock().now().nanoseconds / 1e9

        # 3. حساب زاوية التوجيه من PID
        pid_output = self.pid_steering.update(lane_error, current_time)
        steering_lane = -pid_output

        # 4. تجنب الحواجز الجانبية
        steering_avoid = 0.0
        if self.left_dist < self.safe_distance:
            steering_avoid += (self.safe_distance - self.left_dist) * self.wall_gain
        if self.right_dist < self.safe_distance:
            steering_avoid -= (self.safe_distance - self.right_dist) * self.wall_gain

        steering_raw = steering_lane + steering_avoid
        steering_raw = np.clip(steering_raw, -self.max_angle, self.max_angle)

        # 5. آلة الحالات (من الكود الثاني بالكامل)
        speed = self.base_speed
        steering = steering_raw

        if self.state == "DRIVING":
            # خفض السرعة قرب الرصيف
            if self.left_dist < self.danger_distance or self.right_dist < self.danger_distance:
                speed = 0.2
            else:
                speed = self.base_speed

            # معالجة إشارة STOP
            if sign == 'stop':
                if current_time - self.last_stop_time > self.stop_cooldown:
                    self.get_logger().info('🛑 STOP! توقف لـ 3 ثوان')
                    self.state = "STOPPED"
                    self.state_start_time = current_time
                    self.last_stop_time = current_time
                    speed = 0.0
                    steering = 0.0
                    self.pid_steering.reset()
                else:
                    self.get_logger().info('🛑 STOP (فترة تبريد)')

            # معالجة إشارة YIELD
            elif sign == 'yield':
                if self.front_dist < 2.0:
                    self.get_logger().info(f'⚠️ YIELD: عائق أمامي على مسافة {self.front_dist:.2f}m، انتظار')
                    self.state = "YIELDING"
                    self.state_start_time = current_time
                    speed = 0.0
                    steering = 0.0
                    self.pid_steering.reset()
                else:
                    self.get_logger().info('➡️ YIELD: الطريق خالي، متابعة')

        elif self.state == "STOPPED":
            speed = 0.0
            steering = 0.0
            if current_time - self.state_start_time > self.stop_duration:
                self.get_logger().info('✅ انتهى التوقف، استئناف القيادة')
                self.state = "DRIVING"
                self.pid_steering.reset()

        elif self.state == "YIELDING":
            speed = 0.0
            steering = 0.0
            if self.front_dist > 2.0 or (current_time - self.state_start_time > self.yield_max_wait):
                self.get_logger().info('✅ انتهى انتظار YIELD، استئناف')
                self.state = "DRIVING"
                self.pid_steering.reset()

        # إنشاء إطار العرض للمسار
        lane_frame = None
        if self.show_debug:
            lane_frame = self.create_lane_debug_frame(frame, road_mask, white_mask, lane_error, steering, speed, sign, self.state, leftmost_point)

        return steering, speed, lane_frame

    # --------------------------------------------------------
    # كشف المسرب الأيسر (معدل من الكود الأول)
    # --------------------------------------------------------
    def detect_left_lane_error(self, image):
        height, width = image.shape[:2]
        target_x = width * 0.25  # نستهدف 25% من عرض الصورة (المسرب الأيسر)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # قناع الأسود (الطريق)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)

        # قناع الأبيض (الرصيف / العلامات)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # الطريق = أسود وليس أبيض
        road_mask = cv2.bitwise_and(black_mask, cv2.bitwise_not(white_mask))

        # تنظيف القناع
        kernel = np.ones((5,5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        # منطقة الاهتمام (أسفل 40% من الصورة)
        roi_height = int(height * 0.4)
        roi_y_start = height - roi_height
        slices = 5
        slice_height = roi_height // slices
        points = []        # كل نقاط الطريق
        left_points = []   # النقاط في النصف الأيسر فقط

        for i in range(slices):
            y_start = roi_y_start + i * slice_height
            y_end = y_start + slice_height
            slice_mask = road_mask[y_start:y_end, :]
            black_pixels = np.where(slice_mask > 0)
            if len(black_pixels[1]) > 100:
                avg_x = np.mean(black_pixels[1])
                avg_y = (y_start + y_end) // 2
                points.append((avg_x, avg_y))
                if avg_x < width / 2:  # في النصف الأيسر
                    left_points.append((avg_x, avg_y))

        # نستخدم النقاط اليسرى إن وجدت، وإلا كل النقاط (حالة طارئة)
        if len(left_points) > 0:
            closest_point = max(left_points, key=lambda p: p[1])  # الأقرب للسيارة (أكبر y)
            road_center_x = closest_point[0]
            error = road_center_x - target_x
        elif len(points) > 0:
            closest_point = max(points, key=lambda p: p[1])
            road_center_x = closest_point[0]
            error = road_center_x - target_x
        else:
            error = 0.0
            closest_point = None

        return error, road_mask, white_mask, closest_point

    # --------------------------------------------------------
    # كشف الإشارات (من الكود الثاني - مبسط)
    # --------------------------------------------------------
    def sign_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            sign = self.detect_traffic_sign(cv_image)

            with self.sign_lock:
                self.detected_sign = sign
                self.last_sign_frame = cv_image.copy()
        except Exception as e:
            self.get_logger().error(f'خطأ في sign_image_callback: {e}')

    def detect_traffic_sign(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]

        # قناع أحمر (مجالين)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

        # نأخذ فقط الجزء العلوي 40% للإشارات الحمراء
        roi_sign = red_mask[0:int(height*0.4), :]
        red_pixels = cv2.countNonZero(roi_sign)

        # قناع أبيض لخط التوقف (أسفل الصورة)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        roi_stop_line = white_mask[int(height*0.7):, int(width*0.2):int(width*0.8)]
        white_pixels = cv2.countNonZero(roi_stop_line)

        if red_pixels > 500 and white_pixels > 300:
            return 'stop'
        elif red_pixels > 200 and white_pixels < 200:
            return 'yield'
        else:
            return None

    # --------------------------------------------------------
    # Lidar callback
    # --------------------------------------------------------
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges <= 0] = 100.0
        ranges[np.isinf(ranges)] = 100.0

        n = len(ranges)
        left_indices = list(range(int(n*0.22), int(n*0.28)))
        self.left_dist = np.min(ranges[left_indices])
        right_indices = list(range(int(n*0.72), int(n*0.78)))
        self.right_dist = np.min(ranges[right_indices])
        front_indices = list(range(0, int(n*0.06))) + list(range(int(n*0.94), n))
        self.front_dist = np.min(ranges[front_indices])

    # --------------------------------------------------------
    # إنشاء إطار العرض للمسار (مع خط الهدف للمسرب الأيسر)
    # --------------------------------------------------------
    def create_lane_debug_frame(self, image, road_mask, white_mask, error, steering, speed, sign, state, leftmost_point):
        try:
            height, width = image.shape[:2]
            target_x = int(width * 0.25)
            road_vis = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
            white_vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
            road_vis[road_mask > 0] = [0, 255, 0]      # أخضر للطريق
            white_vis[white_mask > 0] = [255, 0, 0]    # أحمر للعلامات البيضاء
            combined = cv2.addWeighted(image, 0.7, road_vis, 0.3, 0)
            combined = cv2.addWeighted(combined, 1, white_vis, 0.3, 0)

            # خط الهدف (المسرب الأيسر)
            cv2.line(combined, (target_x, 0), (target_x, height), (255, 255, 0), 2)

            # رسم النقطة الأقرب وخط الخطأ
            if leftmost_point is not None:
                cv2.circle(combined, (int(leftmost_point[0]), int(leftmost_point[1])), 5, (0, 255, 255), -1)
                cv2.line(combined, (target_x, height-20),
                         (int(leftmost_point[0]), int(leftmost_point[1])), (0, 255, 255), 2)

            # معلومات
            info = [
                f"Steering: {steering:.2f} rad",
                f"Error: {error:.1f} px",
                f"Speed: {speed:.2f} m/s",
                f"State: {state}",
                f"Sign: {sign if sign else 'None'}"
            ]
            for i, txt in enumerate(info):
                cv2.putText(combined, txt, (10, 30 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "أخضر = طريق", (10, height-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(combined, "أحمر = علامات بيضاء", (10, height-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(combined, "هدف المسرب الأيسر", (target_x-70, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            return combined
        except Exception as e:
            self.get_logger().error(f'خطأ في create_lane_debug_frame: {e}')
            return None

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TrafficAwareBlackRoadFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('👋 إيقاف...')
    finally:
        stop_msg = MotorCommands()
        stop_msg.motor_names = ['steering_angle', 'motor_throttle']
        stop_msg.values = [0.0, 0.0]
        node.pub_cmd.publish(stop_msg)
        node.myCam4.terminate()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()