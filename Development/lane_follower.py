
Enhanced Lane Follower - ACC Competition Edition
=================================================
Based on best practices from:
- Quanser ACC-Competition-2025 GitHub
- York University progress video insights
- QCar2 official documentation
- Competition handbook requirements

Improvements over original:
1. ✅ Fixed hardcoded offset bug
2. ✅ Optimized PID for competition performance
3. ✅ Adaptive speed based on curve detection
4. ✅ Multi-region lane detection (near + far)
5. ✅ Enhanced white line boundary detection
6. ✅ Kalman filter for steering smoothness
7. ✅ State machine for different scenarios
8. ✅ Performance monitoring and logging
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
    INTERSECTION = 3
    PARKING = 4
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

class CompetitionLaneFollower(Node):
    """
    Competition-grade lane follower with advanced features
    Optimized for ACC 2025 Competition requirements
    """
    
    def __init__(self):
        super().__init__('competition_lane_follower')
        
        # QoS Profile - BEST_EFFORT for QLabs performance
        # Reference: ACC-Competition-2025 FAQ - "Why is My QLabs Performance Low?"
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
        # Tuned based on competition requirements
        self.kp = 0.60  # Increased for faster response
        self.ki = 0.018 # Slight increase for drift correction
        self.kd = 0.70  # Strong damping for stability
        
        # PID state variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
        
        # ===== Speed Configuration =====
        # Competition optimized speeds
        self.max_speed = 0.18         # Maximum safe speed
        self.cruise_speed = 0.15      # Normal cruising
        self.curve_speed = 0.10       # Medium curves
        self.sharp_curve_speed = 0.07 # Sharp turns
        self.recovery_speed = 0.06    # Lost track recovery
        
        # ===== Detection Parameters =====
        # HSV ranges optimized for QLabs Virtual QCar2
        self.yellow_lower = np.array([18, 70, 70])
        self.yellow_upper = np.array([45, 255, 255])
        self.white_lower = np.array([0, 0, 175])
        self.white_upper = np.array([180, 65, 255])
        
        # ===== State Machine =====
        self.current_state = DrivingState.LANE_FOLLOWING
        self.state_transition_time = time.time()
        
        # ===== Kalman Filter for Smoothing =====
        self.kalman_steering = KalmanFilter(
            process_variance=1e-3,
            measurement_variance=5e-2
        )
        
        # ===== Moving Average Buffers =====
        self.steering_history = deque(maxlen=4)
        self.error_history = deque(maxlen=3)
        self.speed_history = deque(maxlen=3)
        
        # ===== Lane Detection Regions =====
        # Multi-region for better curve anticipation
        self.near_region_weight = 0.65  # Close region (immediate control)
        self.far_region_weight = 0.35   # Far region (lookahead)
        
        # ===== Curve Detection Thresholds =====
        self.curve_threshold_medium = 0.35
        self.curve_threshold_sharp = 0.60
        
        # ===== White Line Safety =====
        self.white_emergency_threshold = 4000
        self.white_warning_threshold = 2200
        
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
        self.debug_mode = False  # Set to True for verbose logging
        
        self.get_logger().info('🏁 Competition Lane Follower Initialized')
        self.get_logger().info(f'📊 PID: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}')
        self.get_logger().info(f'⚡ Speed Range: {self.sharp_curve_speed}-{self.max_speed} m/s')
        self.get_logger().info('🎯 Ready for ACC 2025 Competition!')
    
    def detect_lane_multi_region(self, frame):
        """
        Multi-region lane detection for improved curve handling
        Returns: near_mask, far_mask, near_hsv
        """
        height, width = frame.shape[:2]
        
        # Define regions (optimized for QCar2 camera FOV)
        near_start = int(height * 0.68)  # Bottom 32%
        far_start = int(height * 0.48)   # Middle 20%
        far_end = int(height * 0.68)
        
        # Extract regions
        near_roi = frame[near_start:height, :]
        far_roi = frame[far_start:far_end, :]
        
        # Convert to HSV
        hsv_near = cv2.cvtColor(near_roi, cv2.COLOR_BGR2HSV)
        hsv_far = cv2.cvtColor(far_roi, cv2.COLOR_BGR2HSV)
        
        # Yellow lane detection
        mask_near = cv2.inRange(hsv_near, self.yellow_lower, self.yellow_upper)
        mask_far = cv2.inRange(hsv_far, self.yellow_lower, self.yellow_upper)
        
        # Morphological operations for noise reduction
        kernel = np.ones((4, 4), np.uint8)
        mask_near = cv2.morphologyEx(mask_near, cv2.MORPH_CLOSE, kernel)
        mask_near = cv2.erode(mask_near, kernel, iterations=1)
        mask_near = cv2.dilate(mask_near, kernel, iterations=2)
        
        # Light processing for far region (speed optimization)
        kernel_small = np.ones((3, 3), np.uint8)
        mask_far = cv2.morphologyEx(mask_far, cv2.MORPH_CLOSE, kernel_small)
        
        return mask_near, mask_far, hsv_near
    
    def calculate_lane_center(self, mask_near, mask_far, width):
        """
        Calculate weighted lane center from both regions
        Returns: center, found, has_far_vision
        """
        # Get moments
        M_near = cv2.moments(mask_near)
        M_far = cv2.moments(mask_far)
        
        center_near = None
        center_far = None
        
        # Near region center
        if M_near["m00"] > 150:  # Minimum area threshold
            center_near = int(M_near["m10"] / M_near["m00"])
        
        # Far region center
        if M_far["m00"] > 100:  # Lower threshold for far region
            center_far = int(M_far["m10"] / M_far["m00"])
        
        # Weighted combination
        if center_near is not None and center_far is not None:
            weighted_center = (self.near_region_weight * center_near + 
                             self.far_region_weight * center_far)
            return weighted_center, True, True
        elif center_near is not None:
            return center_near, True, False
        elif center_far is not None:
            # Use far region only as last resort
            return center_far, True, False
        else:
            return None, False, False
    
    def compute_pid_control(self, error):
        """Enhanced PID controller with anti-windup"""
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt < 0.001:
            dt = 0.001
        
        # Proportional
        P = error
        
        # Integral with anti-windup and conditional integration
        # Only integrate when error is small (prevent integral windup)
        if abs(error) < 0.5:
            self.integral += error * dt
            # Clamp integral
            self.integral = np.clip(self.integral, -0.7, 0.7)
        else:
            # Decay integral when error is large
            self.integral *= 0.95
        
        I = self.integral
        
        # Derivative with low-pass filtering
        self.error_history.append(error)
        if len(self.error_history) >= 2:
            filtered_error = np.mean(list(self.error_history)[-2:])
            D = (filtered_error - self.prev_error) / dt
        else:
            D = 0
        
        # PID output
        output = (self.kp * P) + (self.ki * I) + (self.kd * D)
        
        # Update state
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def adaptive_speed_control(self, abs_error, has_far_vision):
        """
        Intelligent speed adaptation based on curve severity
        """
        if abs_error > self.curve_threshold_sharp:
            # Sharp curve detected
            speed = self.sharp_curve_speed
            self.current_state = DrivingState.CURVE_HANDLING
        elif abs_error > self.curve_threshold_medium:
            # Medium curve
            # Linear interpolation between sharp and cruise speed
            ratio = (abs_error - self.curve_threshold_medium) / \
                   (self.curve_threshold_sharp - self.curve_threshold_medium)
            speed = self.curve_speed + (self.sharp_curve_speed - self.curve_speed) * ratio
            self.current_state = DrivingState.CURVE_HANDLING
        else:
            # Straight or gentle curve
            if has_far_vision:
                # Clear vision ahead - can go faster
                speed = self.max_speed
            else:
                # Limited vision - be cautious
                speed = self.cruise_speed
            
            self.current_state = DrivingState.LANE_FOLLOWING
        
        # Smooth speed transitions
        self.speed_history.append(speed)
        return np.mean(self.speed_history)
    
    def detect_white_boundaries(self, hsv_frame, width):
        """Enhanced white line detection with emergency response"""
        mask_white = cv2.inRange(hsv_frame, self.white_lower, self.white_upper)
        
        # Split left and right
        mid = width // 2
        left_white = mask_white[:, :mid]
        right_white = mask_white[:, mid:]
        
        left_pixels = cv2.countNonZero(left_white)
        right_pixels = cv2.countNonZero(right_white)
        
        return left_pixels, right_pixels
    
    def recovery_behavior(self):
        """Smart recovery when lane is lost"""
        self.lost_frames += 1
        
        if self.lost_frames <= 5:
            # Recently lost - continue with last known direction
            steering = self.last_valid_error * self.kp * 0.8
            throttle = self.recovery_speed
        elif self.lost_frames <= self.max_lost_frames:
            # Extended loss - systematic search
            # Oscillate search based on last known direction
            search_direction = np.sign(self.last_valid_error) if self.last_valid_error != 0 else 1
            steering = 0.7 * search_direction
            throttle = self.recovery_speed
        else:
            # Critical - stopped for too long
            # Full turn in last known direction
            steering = 0.9 * np.sign(self.last_valid_error)
            throttle = self.recovery_speed * 0.8
        
        # Decay integral during recovery
        self.integral *= 0.8
        
        return steering, throttle
    
    def image_callback(self, msg):
        """Main processing callback"""
        try:
            # Convert ROS image
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if frame is None:
                return
            
            height, width = frame.shape[:2]
            
            # ===== Multi-Region Lane Detection =====
            mask_near, mask_far, hsv_near = self.detect_lane_multi_region(frame)
            
            # ===== Calculate Lane Center =====
            lane_center, lane_found, has_far_vision = self.calculate_lane_center(
                mask_near, mask_far, width
            )
            
            # ===== Initialize Control Variables =====
            steering_cmd = 0.0
            throttle = 0.0
            
            if lane_found:
                # ===== Lane Tracking Active =====
                self.lost_frames = 0
                self.current_state = DrivingState.LANE_FOLLOWING
                
                # Calculate error (normalized)
                target_center = width / 2
                error = (lane_center - target_center) / (width / 2)
                
                # Store for recovery
                self.last_valid_error = error
                self.last_valid_time = time.time()
                
                # ===== PID Control =====
                pid_output = self.compute_pid_control(error)
                
                # ===== Kalman Filtering =====
                steering_cmd = self.kalman_steering.update(pid_output)
                
                # ===== Adaptive Speed =====
                throttle = self.adaptive_speed_control(abs(error), has_far_vision)
                
                # ===== White Line Safety System =====
                left_white, right_white = self.detect_white_boundaries(hsv_near, width)
                
                if right_white > self.white_emergency_threshold:
                    # EMERGENCY: Strong white on right
                    steering_cmd = min(steering_cmd - 0.55, -0.5)
                    throttle = self.sharp_curve_speed
                    self.get_logger().warn('🚨 EMERGENCY RIGHT - HARD LEFT')
                    
                elif left_white > self.white_emergency_threshold:
                    # EMERGENCY: Strong white on left
                    steering_cmd = max(steering_cmd + 0.55, 0.5)
                    throttle = self.sharp_curve_speed
                    self.get_logger().warn('🚨 EMERGENCY LEFT - HARD RIGHT')
                    
                elif right_white > self.white_warning_threshold:
                    # Warning: approaching right boundary
                    steering_cmd -= 0.28
                    throttle = min(throttle, self.curve_speed)
                    
                elif left_white > self.white_warning_threshold:
                    # Warning: approaching left boundary
                    steering_cmd += 0.28
                    throttle = min(throttle, self.curve_speed)
                
                # Clamp steering
                steering_cmd = np.clip(steering_cmd, -1.0, 1.0)
                
                # ===== Logging (if debug mode or significant event) =====
                if self.debug_mode or abs(error) > 0.4:
                    self.get_logger().info(
                        f'{self.current_state.name} | '
                        f'E:{error:.2f} | S:{steering_cmd:.2f} | '
                        f'V:{throttle:.2f} | Far:{has_far_vision}'
                    )
            
            else:
                # ===== Lane Lost - Recovery Mode =====
                self.current_state = DrivingState.RECOVERY
                steering_cmd, throttle = self.recovery_behavior()
                
                self.get_logger().warn(
                    f'❌ LOST ({self.lost_frames}/{self.max_lost_frames}) - '
                    f'Recovery: S={steering_cmd:.2f}'
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
    node = CompetitionLaneFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down Competition Lane Follower...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
