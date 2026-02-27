#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from qcar2_interfaces.msg import MotorCommands

# ------------------------------------------------------------
#  فئة متحكم PID بسيطة
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
            dt = 0.01  # قيمة افتراضية للمرة الأولى
        else:
            dt = current_time - self.prev_time
            if dt <= 0.0:
                dt = 0.01
        
        # P
        p_term = self.kp * error
        
        # I
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # D
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # المجموع
        output = p_term + i_term + d_term
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)
        
        # تخزين القيم للدورة القادمة
        self.prev_error = error
        self.prev_time = current_time
        
        return output

# ------------------------------------------------------------
#  العقدة الرئيسية
# ------------------------------------------------------------
class TrafficAwareLaneFollower(Node):
    def __init__(self):
        super().__init__('traffic_aware_lane_follower')
        
        # --- المشتركات والناشرون ---
        self.sub_cam = self.create_subscription(Image, '/camera/color_image', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pub_cmd = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)
        
        self.bridge = CvBridge()
        
        # --- متحكم PID للتوجيه ---
        # قيم تجريبية - تحتاج تعديل حسب مسارك
        self.pid_steering = PIDController(
            kp=0.004,           # المعامل التناسبي (P)
            ki=0.0002,          # المعامل التكاملي (I) - يزيل الخطأ الثابت
            kd=0.001,           # المعامل التفاضلي (D) - يخفف التذبذب
            integral_limit=0.5, # منع تراكم التكامل الزائد
            output_limit=0.8    # الحد الأقصى لزاوية التوجيه
        )
        
        # --- متغيرات تتبع المسار (لنحتاجها فقط لحساب الخطأ) ---
        self.lane_error = 0.0
        
        # --- تجنب الرصيف ---
        self.left_dist = 100.0
        self.right_dist = 100.0
        self.front_dist = 100.0
        self.wall_gain = 0.8
        self.safe_distance = 0.5
        self.danger_distance = 0.25
        
        # --- السرعة ---
        self.base_speed = 0.5
        self.speed = self.base_speed
        
        # --- آلة الحالة للإشارات ---
        self.state = "DRIVING"
        self.state_start_time = 0.0
        self.stop_duration = 3.0
        self.yield_max_wait = 5.0
        
        # --- منع تكرار إشارة الوقف ---
        self.last_stop_time = 0.0
        self.stop_cooldown = 5.0
        
        self.get_logger().info("🚗 PID Lane Follower + Traffic Signs + Curb Avoidance")

    # ------------------------------------------------------------
    # دالة الليدار
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # دالة الكاميرا
    # ------------------------------------------------------------
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.process_image(cv_image)
        except Exception:
            pass

    # ------------------------------------------------------------
    # المعالجة الرئيسية
    # ------------------------------------------------------------
    def process_image(self, frame):
        # ----- 1. حساب خطأ المسار (lane error) -----
        self.lane_error = self.detect_lane_center(frame)
        
        # ----- 2. تطبيق PID على الخطأ -----
        current_time = time.time()
        pid_output = self.pid_steering.update(self.lane_error, current_time)
        
        # التوجيه من PID (الإشارة سالبة لأن التوجيه الموجب = يسار حسب QCar2)
        self.steering_lane = -pid_output
        
        # ----- 3. تجنب الرصيف -----
        steering_avoid = 0.0
        if self.left_dist < self.safe_distance:
            steering_avoid += (self.safe_distance - self.left_dist) * self.wall_gain
        if self.right_dist < self.safe_distance:
            steering_avoid -= (self.safe_distance - self.right_dist) * self.wall_gain
        
        self.steering = self.steering_lane + steering_avoid
        self.steering = np.clip(self.steering, -0.8, 0.8)
        
        # ----- 4. كشف إشارات المرور -----
        sign = self.detect_traffic_sign(frame)
        
        # ----- 5. آلة الحالة -----
        if self.state == "DRIVING":
            # ضبط السرعة
            if self.left_dist < self.danger_distance or self.right_dist < self.danger_distance:
                self.speed = 0.2
            else:
                self.speed = self.base_speed
            
            # STOP
            if sign == 'stop':
                if time.time() - self.last_stop_time > self.stop_cooldown:
                    self.get_logger().info("🛑 STOP! توقف 3 ثواني")
                    self.state = "STOPPED"
                    self.state_start_time = time.time()
                    self.last_stop_time = time.time()
                    self.speed = 0.0
                    self.steering = 0.0
                    self.pid_steering.reset()  # إعادة ضبط PID عند التوقف
                else:
                    self.get_logger().info("🛑 STOP (توقف مؤخراً - نتجاوز)")
            
            # YIELD
            elif sign == 'yield':
                if self.front_dist < 2.0:
                    self.get_logger().info(f"⚠️ YIELD: عائق {self.front_dist:.2f}م - انتظار")
                    self.state = "YIELDING"
                    self.state_start_time = time.time()
                    self.speed = 0.0
                    self.steering = 0.0
                    self.pid_steering.reset()
                else:
                    self.get_logger().info("➡️ YIELD: الطريق خالٍ")
        
        elif self.state == "STOPPED":
            self.speed = 0.0
            self.steering = 0.0
            if time.time() - self.state_start_time > self.stop_duration:
                self.get_logger().info("✅ انتهى التوقف - انطلاق")
                self.state = "DRIVING"
                self.pid_steering.reset()  # إعادة ضبط PID بعد التوقف
        
        elif self.state == "YIELDING":
            self.speed = 0.0
            self.steering = 0.0
            if self.front_dist > 2.0 or (time.time() - self.state_start_time > self.yield_max_wait):
                self.get_logger().info("✅ انتهى الانتظار - انطلاق")
                self.state = "DRIVING"
                self.pid_steering.reset()
        
        # ----- 6. إرسال الأمر -----
        self.publish_command(self.steering, self.speed)

    # ------------------------------------------------------------
    # دالة تحديد منتصف المسار (تُرجع الخطأ)
    # ------------------------------------------------------------
    def detect_lane_center(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = frame.shape
        
        # الأصفر
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # الأبيض
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        roi_y = int(height * 0.6)
        yellow_roi = yellow_mask[roi_y:, :]
        white_roi = white_mask[roi_y:, :]
        
        y_pts = cv2.findNonZero(yellow_roi)
        w_pts = cv2.findNonZero(white_roi)
        
        left_x, right_x = None, None
        if y_pts is not None:
            left_x = np.mean(y_pts[:, 0, 0])
        if w_pts is not None:
            right_x = np.mean(w_pts[:, 0, 0])
        
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2
        elif left_x is not None:
            lane_center = left_x + 150
        elif right_x is not None:
            lane_center = right_x - 150
        else:
            return 0.0
        
        error = lane_center - (width / 2)
        return error

    # ------------------------------------------------------------
    # دالة كشف الإشارات
    # ------------------------------------------------------------
    def detect_traffic_sign(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        roi_sign = red_mask[0:int(height*0.4), :]
        red_pixels = cv2.countNonZero(roi_sign)
        
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

    # ------------------------------------------------------------
    # نشر الأمر
    # ------------------------------------------------------------
    def publish_command(self, steering, throttle):
        msg = MotorCommands()
        msg.motor_names = ["steering_angle", "motor_throttle"]
        msg.values = [float(steering), float(throttle)]
        self.pub_cmd.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrafficAwareLaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = MotorCommands()
        stop_msg.motor_names = ["steering_angle", "motor_throttle"]
        stop_msg.values = [0.0, 0.0]
        node.pub_cmd.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
