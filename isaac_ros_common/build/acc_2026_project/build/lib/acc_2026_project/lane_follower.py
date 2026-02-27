"""
Enhanced Lane Follower - With White Line Recovery
==================================================
Version: 2.0 - Fixed white line collision recovery

New Feature:
- Reverse and correct steering when hitting white line
- Progressive recovery system
- Emergency backup maneuver
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from qcar2_interfaces.msg import MotorCommands
import time
from collections import deque
from enum import Enum

class DrivingState(Enum):
    """State machine for different driving scenarios"""
    LANE_FOLLOWING = 1
    CURVE_HANDLING = 2
    WHITE_LINE_COLLISION = 3  # ← جديد!
    REVERSING = 4             # ← جديد!
    RECOVERY = 5

class KalmanFilter:
    """Simple 1D Kalman filter for steering smoothing"""
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.estimate_error = 1.0
    
    def update(self, measurement):
        """Update filter with new measurement"""
        # Prediction
        prediction_error = self.estimate_error + self.process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate

class CompetitionLaneFollowerV2(Node):
    """
    Enhanced lane follower with white line collision recovery
    """
    
    def __init__(self):
        super().__init__('competition_lane_follower_v2')
        
        # QoS Profile - BEST_EFFORT for QLabs performance
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # ROS2 subscribers and publishers
        self.subscription = self.create_subscription(
            Image,
            '/camera/color_image',
            self.image_callback,
            qos_profile
        )
        
        self.publisher_ = self.create_publisher(
            MotorCommands,
            '/qcar2_motor_speed_cmd',
            10
        )
        
        self.bridge = CvBridge()
        
        # ===== PID Controller Configuration =====
        self.kp = 0.60
        self.ki = 0.018
        self.kd = 0.70
        
        # PID state variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
        
        # ===== Speed Configuration =====
        self.max_speed = 0.18
        self.cruise_speed = 0.15
        self.curve_speed = 0.10
        self.sharp_curve_speed = 0.07
        self.recovery_speed = 0.06
        self.reverse_speed = -0.10  # ← جديد! سرعة الرجوع
        
        # ===== Detection Parameters =====
        self.yellow_lower = np.array([18, 70, 70])
        self.yellow_upper = np.array([45, 255, 255])
        self.white_lower = np.array([0, 0, 175])
        self.white_upper = np.array([180, 65, 255])
        
        # ===== State Machine =====
        self.current_state = DrivingState.LANE_FOLLOWING
        self.state_transition_time = time.time()
        
        # ===== Kalman Filter =====
        self.kalman_steering = KalmanFilter(
            process_variance=1e-3,
            measurement_variance=5e-2
        )
        
        # ===== Moving Average Buffers =====
        self.steering_history = deque(maxlen=4)
        self.error_history = deque(maxlen=3)
        self.speed_history = deque(maxlen=3)
        
        # ===== Lane Detection Regions =====
        self.near_region_weight = 0.65
        self.far_region_weight = 0.35
        
        # ===== Curve Detection Thresholds =====
        self.curve_threshold_medium = 0.35
        self.curve_threshold_sharp = 0.60
        
        # ===== White Line Safety - Enhanced! =====
        self.white_critical_threshold = 5000   # ← اصطدام! يجب الرجوع
        self.white_emergency_threshold = 4000  # طوارئ شديدة
        self.white_warning_threshold = 2200    # تحذير
        
        # ===== White Line Collision Recovery =====
        self.collision_detected = False
        self.collision_side = None  # 'left' or 'right'
        self.reverse_frames = 0
        self.max_reverse_frames = 20  # عدد frames للرجوع
        self.correction_frames = 0
        self.max_correction_frames = 15  # عدد frames للتصحيح
        
        # ===== Recovery System =====
        self.lost_frames = 0
        self.max_lost_frames = 12
        self.last_valid_error = 0.0
        self.last_valid_time = time.time()
        
        # ===== Performance Monitoring =====
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        
        # ===== Debug/Logging =====
        self.debug_mode = True  # للمراقبة
        
        self.get_logger().info('🏁 Competition Lane Follower V2 - With White Line Recovery')
        self.get_logger().info(f'📊 PID: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}')
        self.get_logger().info(f'⚡ Speed Range: {self.sharp_curve_speed}-{self.max_speed} m/s')
        self.get_logger().info('🔄 Reverse Recovery: ENABLED')
    
    def detect_lane_multi_region(self, frame):
        """Multi-region lane detection"""
        height, width = frame.shape[:2]
        
        near_start = int(height * 0.68)
        far_start = int(height * 0.48)
        far_end = int(height * 0.68)
        
        near_roi = frame[near_start:height, :]
        far_roi = frame[far_start:far_end, :]
        
        hsv_near = cv2.cvtColor(near_roi, cv2.COLOR_BGR2HSV)
        hsv_far = cv2.cvtColor(far_roi, cv2.COLOR_BGR2HSV)
        
        mask_near = cv2.inRange(hsv_near, self.yellow_lower, self.yellow_upper)
        mask_far = cv2.inRange(hsv_far, self.yellow_lower, self.yellow_upper)
        
        kernel = np.ones((4, 4), np.uint8)
        mask_near = cv2.morphologyEx(mask_near, cv2.MORPH_CLOSE, kernel)
        mask_near = cv2.erode(mask_near, kernel, iterations=1)
        mask_near = cv2.dilate(mask_near, kernel, iterations=2)
        
        kernel_small = np.ones((3, 3), np.uint8)
        mask_far = cv2.morphologyEx(mask_far, cv2.MORPH_CLOSE, kernel_small)
        
        return mask_near, mask_far, hsv_near
    
    def calculate_lane_center(self, mask_near, mask_far, width):
        """Calculate weighted lane center"""
        M_near = cv2.moments(mask_near)
        M_far = cv2.moments(mask_far)
        
        center_near = None
        center_far = None
        
        if M_near["m00"] > 150:
            center_near = int(M_near["m10"] / M_near["m00"])
        
        if M_far["m00"] > 100:
            center_far = int(M_far["m10"] / M_far["m00"])
        
        if center_near is not None and center_far is not None:
            weighted_center = (self.near_region_weight * center_near + 
                             self.far_region_weight * center_far)
            return weighted_center, True, True
        elif center_near is not None:
            return center_near, True, False
        elif center_far is not None:
            return center_far, True, False
        else:
            return None, False, False
    
    def compute_pid_control(self, error):
        """Enhanced PID controller"""
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt < 0.001:
            dt = 0.001
        
        P = error
        
        if abs(error) < 0.5:
            self.integral += error * dt
            self.integral = np.clip(self.integral, -0.7, 0.7)
        else:
            self.integral *= 0.95
        
        I = self.integral
        
        self.error_history.append(error)
        if len(self.error_history) >= 2:
            filtered_error = np.mean(list(self.error_history)[-2:])
            D = (filtered_error - self.prev_error) / dt
        else:
            D = 0
        
        output = (self.kp * P) + (self.ki * I) + (self.kd * D)
        
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def adaptive_speed_control(self, abs_error, has_far_vision):
        """Intelligent speed adaptation"""
        if abs_error > self.curve_threshold_sharp:
            speed = self.sharp_curve_speed
            self.current_state = DrivingState.CURVE_HANDLING
        elif abs_error > self.curve_threshold_medium:
            ratio = (abs_error - self.curve_threshold_medium) / \
                   (self.curve_threshold_sharp - self.curve_threshold_medium)
            speed = self.curve_speed + (self.sharp_curve_speed - self.curve_speed) * ratio
            self.current_state = DrivingState.CURVE_HANDLING
        else:
            if has_far_vision:
                speed = self.max_speed
            else:
                speed = self.cruise_speed
            
            if self.current_state != DrivingState.WHITE_LINE_COLLISION:
                self.current_state = DrivingState.LANE_FOLLOWING
        
        self.speed_history.append(speed)
        return np.mean(self.speed_history)
    
    def detect_white_boundaries(self, hsv_frame, width):
        """Enhanced white line detection"""
        mask_white = cv2.inRange(hsv_frame, self.white_lower, self.white_upper)
        
        mid = width // 2
        left_white = mask_white[:, :mid]
        right_white = mask_white[:, mid:]
        
        left_pixels = cv2.countNonZero(left_white)
        right_pixels = cv2.countNonZero(right_white)
        
        return left_pixels, right_pixels
    
    def handle_white_line_collision(self, left_white, right_white):
        """
        🆕 معالجة الاصطدام بالخط الأبيض
        Returns: (should_reverse, reverse_steering, collision_side)
        """
        # تحديد الجانب الذي حصل فيه الاصطدام
        if right_white > self.white_critical_threshold:
            # اصطدام بالخط الأيمن
            self.collision_detected = True
            self.collision_side = 'right'
            self.reverse_frames = 0
            return True, -0.7, 'right'  # رجوع + انعطاف يسار
            
        elif left_white > self.white_critical_threshold:
            # اصطدام بالخط الأيسر
            self.collision_detected = True
            self.collision_side = 'left'
            self.reverse_frames = 0
            return True, 0.7, 'left'  # رجوع + انعطاف يمين
        
        return False, 0.0, None
    
    def execute_reverse_recovery(self):
        """
        🆕 تنفيذ عملية الرجوع والتصحيح
        Returns: (throttle, steering)
        """
        self.reverse_frames += 1
        
        if self.reverse_frames <= self.max_reverse_frames:
            # المرحلة 1: الرجوع للخلف مع التصحيح
            if self.collision_side == 'right':
                # كنا على اليمين، نرجع ونتجه لليسار
                steering = -0.8  # انعطاف قوي لليسار
            else:
                # كنا على اليسار، نرجع ونتجه لليمين
                steering = 0.8   # انعطاف قوي لليمين
            
            throttle = self.reverse_speed
            
            self.get_logger().warn(
                f'🔄 REVERSING ({self.reverse_frames}/{self.max_reverse_frames}) - '
                f'Side: {self.collision_side} | Steering: {steering:.2f}'
            )
            
            return throttle, steering
        
        elif self.reverse_frames <= (self.max_reverse_frames + self.max_correction_frames):
            # المرحلة 2: التقدم للأمام مع تصحيح قوي
            self.correction_frames += 1
            
            if self.collision_side == 'right':
                steering = -0.6  # استمر في التصحيح لليسار
            else:
                steering = 0.6   # استمر في التصحيح لليمين
            
            throttle = self.recovery_speed
            
            self.get_logger().warn(
                f'⚠️  CORRECTING ({self.correction_frames}/{self.max_correction_frames}) - '
                f'Steering: {steering:.2f}'
            )
            
            return throttle, steering
        
        else:
            # انتهى Recovery - ارجع للوضع الطبيعي
            self.collision_detected = False
            self.collision_side = None
            self.reverse_frames = 0
            self.correction_frames = 0
            self.current_state = DrivingState.LANE_FOLLOWING
            
            self.get_logger().info('✅ White line recovery completed - Resuming normal operation')
            
            return None, None  # سيتم التعامل معه في المنطق الرئيسي
    
    def recovery_behavior(self):
        """Smart recovery when lane is lost"""
        self.lost_frames += 1
        
        if self.lost_frames <= 5:
            steering = self.last_valid_error * self.kp * 0.8
            throttle = self.recovery_speed
        elif self.lost_frames <= self.max_lost_frames:
            search_direction = np.sign(self.last_valid_error) if self.last_valid_error != 0 else 1
            steering = 0.7 * search_direction
            throttle = self.recovery_speed
        else:
            steering = 0.9 * np.sign(self.last_valid_error)
            throttle = self.recovery_speed * 0.8
        
        self.integral *= 0.8
        
        return steering, throttle
    
    def image_callback(self, msg):
        """Main processing callback"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if frame is None:
                return
            
            height, width = frame.shape[:2]
            
            # ===== Multi-Region Lane Detection =====
            mask_near, mask_far, hsv_near = self.detect_lane_multi_region(frame)
            
            # ===== White Line Detection (أولاً!) =====
            left_white, right_white = self.detect_white_boundaries(hsv_near, width)
            
            # ===== 🆕 فحص الاصطدام بالخط الأبيض =====
            should_reverse, reverse_steering, collision_side = self.handle_white_line_collision(
                left_white, right_white
            )
            
            # ===== إذا كان هناك اصطدام أو recovery جاري =====
            if self.collision_detected:
                self.current_state = DrivingState.WHITE_LINE_COLLISION
                
                throttle, steering = self.execute_reverse_recovery()
                
                if throttle is not None:  # لا نزال في recovery
                    self.drive_car(throttle, steering)
                    return
                # else: انتهى recovery، استمر في المنطق الطبيعي
            
            # ===== Calculate Lane Center =====
            lane_center, lane_found, has_far_vision = self.calculate_lane_center(
                mask_near, mask_far, width
            )
            
            steering_cmd = 0.0
            throttle = 0.0
            
            if lane_found:
                # ===== Lane Tracking Active =====
                self.lost_frames = 0
                
                target_center = width / 2
                error = (lane_center - target_center) / (width / 2)
                
                self.last_valid_error = error
                self.last_valid_time = time.time()
                
                # ===== PID Control =====
                pid_output = self.compute_pid_control(error)
                
                # ===== Kalman Filtering =====
                steering_cmd = self.kalman_steering.update(pid_output)
                
                # ===== Adaptive Speed =====
                throttle = self.adaptive_speed_control(abs(error), has_far_vision)
                
                # ===== White Line Safety (بعد PID) =====
                if right_white > self.white_emergency_threshold:
                    steering_cmd = min(steering_cmd - 0.55, -0.5)
                    throttle = self.sharp_curve_speed
                    self.get_logger().warn('🚨 EMERGENCY RIGHT - HARD LEFT')
                    
                elif left_white > self.white_emergency_threshold:
                    steering_cmd = max(steering_cmd + 0.55, 0.5)
                    throttle = self.sharp_curve_speed
                    self.get_logger().warn('🚨 EMERGENCY LEFT - HARD RIGHT')
                    
                elif right_white > self.white_warning_threshold:
                    steering_cmd -= 0.28
                    throttle = min(throttle, self.curve_speed)
                    
                elif left_white > self.white_warning_threshold:
                    steering_cmd += 0.28
                    throttle = min(throttle, self.curve_speed)
                
                steering_cmd = np.clip(steering_cmd, -1.0, 1.0)
                
                if self.debug_mode and (abs(error) > 0.4 or right_white > 2000 or left_white > 2000):
                    self.get_logger().info(
                        f'{self.current_state.name} | E:{error:.2f} | S:{steering_cmd:.2f} | '
                        f'V:{throttle:.2f} | L_W:{left_white} | R_W:{right_white}'
                    )
            
            else:
                # ===== Lane Lost - Recovery Mode =====
                self.current_state = DrivingState.RECOVERY
                steering_cmd, throttle = self.recovery_behavior()
                
                self.get_logger().warn(
                    f'❌ LOST ({self.lost_frames}/{self.max_lost_frames}) - Recovery'
                )
            
            # ===== Send Control Commands =====
            self.drive_car(throttle, steering_cmd)
            
            # ===== Performance Monitoring =====
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed
                self.get_logger().info(f'📊 Performance: {self.fps:.1f} FPS')
            
        except Exception as e:
            self.get_logger().error(f'💥 Processing error: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def drive_car(self, throttle, steering):
        """Send motor commands"""
        cmd = MotorCommands()
        cmd.motor_names = ['motor_throttle', 'steering_angle']
        cmd.values = [float(throttle), float(steering)]
        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = CompetitionLaneFollowerV2()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
