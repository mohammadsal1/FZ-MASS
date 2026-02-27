
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from qcar2_interfaces.msg import MotorCommands
import cv2
from cv_bridge import CvBridge
import numpy as np
import threading
import os
from collections import deque
from pal.utilities.vision import Camera2D

# ------------------------------------------------------------
#  PID Controller class
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

        # Parameters
        self.declare_parameter('speed', 0.4)
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

        # Camera parameters for myCam4 (lane following)
        self.declare_parameter('camera_id', '3@tcpip://localhost:18964')
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('frame_rate', 30)

        # YOLO model paths (update with your actual paths)
        self.declare_parameter('yolo_cfg', '/path/to/yolov4-tiny.cfg')
        self.declare_parameter('yolo_weights', '/path/to/yolov4-tiny.weights')
        self.declare_parameter('yolo_names', '/path/to/coco.names')
        self.declare_parameter('yolo_confidence', 0.5)
        self.declare_parameter('yolo_nms', 0.4)

        # Traffic light specific parameters
        self.declare_parameter('traffic_light_area_threshold', 5000)
        self.declare_parameter('traffic_light_color_delay', 3.0)

        # Stop sign wait duration (wait 2 seconds before stopping)
        self.declare_parameter('stop_sign_wait_duration', 2.0)
        self.declare_parameter('stop_duration', 5.0)            # how long to stay stopped
        self.declare_parameter('yield_max_wait', 5.0)

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

        self.tl_area_thresh = self.get_parameter('traffic_light_area_threshold').value
        self.tl_color_delay = self.get_parameter('traffic_light_color_delay').value

        self.stop_sign_wait_duration = self.get_parameter('stop_sign_wait_duration').value
        self.stop_duration = self.get_parameter('stop_duration').value
        self.yield_max_wait = self.get_parameter('yield_max_wait').value

        camera_id = self.get_parameter('camera_id').value
        frame_width = self.get_parameter('frame_width').value
        frame_height = self.get_parameter('frame_height').value
        frame_rate = self.get_parameter('frame_rate').value

        # Publishers & Subscribers
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pub_cmd = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)

        # Subscription to the second camera (for traffic signs and lights)
        self.sub_sign_cam = self.create_subscription(
            Image, '/camera/color_image', self.sign_image_callback, 10)
        self.bridge = CvBridge()

        # Initialize myCam4 (for lane following)
        self.myCam4 = Camera2D(
            cameraId=camera_id,
            frameWidth=frame_width,
            frameHeight=frame_height,
            frameRate=frame_rate
        )
        self.get_logger().info(f'📷 Camera myCam4 initialized with ID: {camera_id}')

        # Initialize additional cameras for obstacle checking (myCam1, myCam3)
        self.myCam1 = None
        self.myCam3 = None
        self.cam1_active = False
        self.cam3_active = False

        # Background subtractors for obstacle detection (fallback)
        self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2()
        self.bg_subtractor3 = cv2.createBackgroundSubtractorMOG2()

        # -------------------- YOLO initialization --------------------
        self.yolo_net = None
        self.yolo_classes = []
        self.yolo_confidence = self.get_parameter('yolo_confidence').value
        self.yolo_nms = self.get_parameter('yolo_nms').value
        self.load_yolo_model()
        # -------------------------------------------------------------

        # PID for lane error
        self.pid_steering = PIDController(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            integral_limit=self.integral_limit,
            output_limit=self.output_limit
        )

        # State machine
        self.state = "DRIVING"
        self.state_start_time = 0.0
        self.last_stop_time = 0.0
        self.stop_cooldown = 5.0                      # فترة تجاهل الإشارات بعد التوقف

        # Approach variables
        self.approach_speed = 0.275
        self.sign_area_threshold = 2000                # عتبة مساحة الإشارة
        self.white_line_threshold = 1000                # عدد البيكسلات البيضاء في أسفل الصورة للكشف عن الخط
        self.approach_sign = None
        self.obstacle_persistent_start = None
        self.obstacle_persistent_threshold = 5.0

        # متغير جديد: الوقت الذي يجب بعده تجاهل الإشارات
        self.ignore_signs_until = 0.0

        # Lidar distances with filtering
        self.raw_front_dist = 100.0
        self.front_dist = 100.0
        self.front_dist_history = deque(maxlen=5)

        # Control variables
        self.lock = threading.Lock()
        self.current_steering = 0.0
        self.current_speed = 0.0

        # Detected sign (shared)
        self.detected_sign = None
        self.detected_sign_area = 0.0
        self.last_sign_frame = None
        self.sign_lock = threading.Lock()

        # -------------------- Traffic light variables (from Code1) --------------------
        self.traffic_light_bbox = None
        self.traffic_light_area = 0.0
        self.traffic_light_color = None
        self.traffic_light_first_detected_time = 0.0
        self.traffic_light_color_determined = False
        self.traffic_light_last_seen = 0.0
        self.tl_center_tolerance = 100
        self.prev_tl_bbox = None
        self.tl_passed = False
        self.tl_pass_confirmed_time = 0.0
        # -----------------------------------------------------------------

        # Obstacle detection results from auxiliary cameras
        self.obstacle_detected_cam1 = False
        self.obstacle_detected_cam3 = False
        self.aux_lock = threading.Lock()

        # Camera status
        self.camera_ok = False

        # Timer
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.create_timer(2.0, self.check_health)

        self.get_logger().info('🚗 Approach speed 0.275, decision based on sign area OR white line. Speed=0.4')
        self.get_logger().info('🚦 Traffic light detection enabled (red/green only).')
        self.get_logger().info(f'⏱ Stop sign wait duration: {self.stop_sign_wait_duration} seconds')

    # --------------------------------------------------------
    # Load YOLO model (from Code1)
    # --------------------------------------------------------
    def load_yolo_model(self):
        cfg = self.get_parameter('yolo_cfg').value
        weights = self.get_parameter('yolo_weights').value
        names_file = self.get_parameter('yolo_names').value

        if not os.path.exists(cfg) or not os.path.exists(weights):
            self.get_logger().warn('YOLO configuration or weights not found. Using shape-based detection.')
            return

        try:
            self.yolo_net = cv2.dnn.readNet(weights, cfg)
            self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            self.yolo_net = None
            return

        if os.path.exists(names_file):
            with open(names_file, 'r') as f:
                self.yolo_classes = [line.strip() for line in f.readlines()]
        else:
            self.get_logger().warn('Class names file not found. Using default COCO classes.')
            self.yolo_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                                 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                                 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                                 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                                 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                                 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                                 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                                 'toothbrush']

        self.get_logger().info('✅ YOLO model loaded successfully.')

    # --------------------------------------------------------
    # YOLO inference (from Code1)
    # --------------------------------------------------------
    def detect_with_yolo(self, frame):
        if self.yolo_net is None:
            return None

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)

        layer_names = self.yolo_net.getUnconnectedOutLayersNames()
        try:
            outputs = self.yolo_net.forward(layer_names)
        except:
            return None

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.yolo_confidence:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.yolo_confidence, self.yolo_nms)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'box': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self.yolo_classes[class_ids[i]] if class_ids[i] < len(self.yolo_classes) else 'unknown'
                })
        return detections

    # --------------------------------------------------------
    # Estimate shape from bounding box (from Code1, for traffic lights and signs)
    # --------------------------------------------------------
    def estimate_shape_from_bbox(self, bbox, frame, color_range=None):
        x, y, w, h = bbox
        height, width = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        if w <= 0 or h <= 0:
            return None

        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None

        if color_range is not None:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower, upper = color_range
            mask = cv2.inRange(hsv, lower, upper)
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            mask = edges

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 100:
            return None

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.03 * peri, True)
        num_vertices = len(approx)
        return num_vertices

    # --------------------------------------------------------
    # Traffic light detection using YOLO + shape fallback (from Code1)
    # --------------------------------------------------------
    def detect_traffic_light_shape_based(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area = 0
        best_bbox = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            if 0.4 < aspect_ratio < 2.5:
                if area > best_area:
                    best_area = area
                    best_bbox = (x, y, w, h)

        if best_bbox is not None:
            return best_bbox, best_area
        return None, 0

    def detect_traffic_light(self, frame):
        if self.yolo_net is not None:
            detections = self.detect_with_yolo(frame)
            if detections:
                tl_detections = [d for d in detections if d['class_id'] == 9]
                if tl_detections:
                    largest = max(tl_detections, key=lambda d: d['box'][2]*d['box'][3])
                    bbox = largest['box']
                    area = bbox[2] * bbox[3]
                    return bbox, area
        return self.detect_traffic_light_shape_based(frame)

    # --------------------------------------------------------
    # Analyze traffic light color (red/green only) - طريقة موثوقة (from Code1)
    # --------------------------------------------------------
    def analyze_traffic_light_color(self, bbox, frame):
        x, y, w, h = bbox
        height, width = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        if w <= 0 or h <= 0:
            return None

        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None

        # تقسيم ROI إلى نصفين: علوي (للأحمر) وسفلي (للأخضر)
        half = h // 2
        roi_top = roi[0:half, :]
        roi_bottom = roi[half:h, :]

        # تحويل إلى HSV
        hsv_top = cv2.cvtColor(roi_top, cv2.COLOR_BGR2HSV)
        hsv_bottom = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2HSV)

        # نطاقات دقيقة للأحمر (في النصف العلوي)
        lower_red1 = np.array([0, 150, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 150])
        upper_red2 = np.array([180, 255, 255])

        # نطاق دقيق للأخضر (في النصف السفلي)
        lower_green = np.array([40, 120, 120])
        upper_green = np.array([80, 255, 255])

        # حساب الأحمر في النصف العلوي
        mask_red_top1 = cv2.inRange(hsv_top, lower_red1, upper_red1)
        mask_red_top2 = cv2.inRange(hsv_top, lower_red2, upper_red2)
        mask_red_top = cv2.bitwise_or(mask_red_top1, mask_red_top2)
        red_top = cv2.countNonZero(mask_red_top)

        # حساب الأخضر في النصف السفلي
        mask_green_bottom = cv2.inRange(hsv_bottom, lower_green, upper_green)
        green_bottom = cv2.countNonZero(mask_green_bottom)

        self.get_logger().info(f'🔴 Red top: {red_top}, 🟢 Green bottom: {green_bottom}')

        # عتبة صغيرة جداً لضمان الكشف حتى عن الإشارات البعيدة
        threshold = 5
        if red_top > green_bottom and red_top > threshold:
            return 'red'
        elif green_bottom > red_top and green_bottom > threshold:
            return 'green'
        # إذا كان أحد اللونين فقط موجوداً
        elif red_top > threshold and green_bottom <= threshold:
            return 'red'
        elif green_bottom > threshold and red_top <= threshold:
            return 'green'
        else:
            return None

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------
    def check_health(self):
        if not self.camera_ok:
            self.get_logger().warn('⏳ Waiting for myCam4 to provide images...')
        if self.last_sign_frame is None:
            self.get_logger().warn('⏳ Waiting for /camera/color_image topic...')

    # --------------------------------------------------------
    # Timer callback (معدل)
    # --------------------------------------------------------
    def timer_callback(self):
        flag = self.myCam4.read()
        if flag:
            self.camera_ok = True
            frame = self.myCam4.imageData
            with self.sign_lock:
                current_sign = self.detected_sign
                current_sign_area = self.detected_sign_area
                tl_bbox = self.traffic_light_bbox
                tl_area = self.traffic_light_area
                tl_last_seen = self.traffic_light_last_seen
                sign_frame_copy = self.last_sign_frame.copy() if self.last_sign_frame is not None else None
            steering, speed, lane_frame = self.process_lane_and_sign(
                frame, current_sign, current_sign_area,
                tl_bbox, tl_area, tl_last_seen, sign_frame_copy)
            with self.lock:
                self.current_steering = steering
                self.current_speed = speed
            if self.show_debug and lane_frame is not None:
                cv2.imshow("Black Road Following", lane_frame)
                cv2.waitKey(1)
        else:
            self.camera_ok = False
            self.get_logger().warn('⚠️ Failed to read image from myCam4', throttle_duration_sec=2.0)

        # عرض صورة كشف الإشارات فقط إذا كانت موجودة
        with self.sign_lock:
            if self.last_sign_frame is not None:
                sign_disp = self.last_sign_frame.copy()
                if self.detected_sign:
                    cv2.putText(sign_disp, f"Sign: {self.detected_sign} ({self.detected_sign_area:.0f})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if self.traffic_light_bbox is not None:
                    x,y,w,h = self.traffic_light_bbox
                    cv2.rectangle(sign_disp, (x,y), (x+w, y+h), (255,0,0), 2)
                    color_text = self.traffic_light_color if self.traffic_light_color else '?'
                    cv2.putText(sign_disp, f"TL {color_text}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                cv2.imshow("Traffic Sign Detection", sign_disp)
                cv2.waitKey(1)

        # عرض الكاميرات المساعدة فقط إذا كانت نشطة
        if self.cam1_active and self.myCam1 is not None:
            if self.myCam1.read():
                frame1 = self.myCam1.imageData
                obs, frame1 = self.detect_obstacles_with_yolo(frame1, cam_id=1)
                with self.aux_lock:
                    self.obstacle_detected_cam1 = obs
                cv2.imshow("Cam1 - Obstacle Check", frame1)
                cv2.waitKey(1)

        if self.cam3_active and self.myCam3 is not None:
            if self.myCam3.read():
                frame3 = self.myCam3.imageData
                obs, frame3 = self.detect_obstacles_with_yolo(frame3, cam_id=3)
                with self.aux_lock:
                    self.obstacle_detected_cam3 = obs
                cv2.imshow("Cam3 - Obstacle Check", frame3)
                cv2.waitKey(1)

        # نشر أوامر المحرك
        with self.lock:
            steering = self.current_steering
            speed = self.current_speed
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(steering), float(speed)]
        self.pub_cmd.publish(msg)

    # --------------------------------------------------------
    # Process lane and sign (merged from Code1 and Code2) - مع تعديل STOP_SIGN_WAIT
    # --------------------------------------------------------
    def process_lane_and_sign(self, frame, sign, sign_area,
                              tl_bbox, tl_area, tl_last_seen, sign_frame):
        lane_error, road_mask, white_mask = self.detect_black_road_error(frame)
        current_time = self.get_clock().now().nanoseconds / 1e9
        pid_output = self.pid_steering.update(lane_error, current_time)
        steering_lane = -pid_output

        steering_avoid = 0.0
        if hasattr(self, 'left_dist') and self.left_dist < self.safe_distance:
            steering_avoid += (self.safe_distance - self.left_dist) * self.wall_gain
        if hasattr(self, 'right_dist') and self.right_dist < self.safe_distance:
            steering_avoid -= (self.safe_distance - self.right_dist) * self.wall_gain
        steering_raw = steering_lane + steering_avoid
        steering_raw = np.clip(steering_raw, -self.max_angle, self.max_angle)

        speed = self.base_speed
        steering = steering_raw

        # Traffic light handling (from Code1, simplified)
        traffic_light_present = (tl_bbox is not None and
                                 (current_time - tl_last_seen) < 1.0 and
                                 sign_frame is not None)

        if traffic_light_present:
            # If traffic light is in lower half, consider it behind and ignore
            height = sign_frame.shape[0]
            x, y, w, h = tl_bbox
            center_y = y + h//2
            if center_y > height // 2:
                self.get_logger().info("🚦 Traffic light is behind (lower half), ignoring")
                if self.state in ["TRAFFIC_DETECTED", "TRAFFIC_RED", "TRAFFIC_GREEN"]:
                    self.state = "DRIVING"
                    self.traffic_light_color_determined = False
                    self.tl_passed = True
                    self.tl_pass_confirmed_time = current_time
            elif self.tl_passed:
                if current_time - self.tl_pass_confirmed_time > 5.0:
                    self.tl_passed = False
                else:
                    self.get_logger().info("🚦 Traffic light was passed recently, ignoring")
            else:
                # Handle traffic light color
                raw_color = self.analyze_traffic_light_color(tl_bbox, sign_frame)
                if raw_color == 'red':
                    self.state = "TRAFFIC_RED"
                    self.traffic_light_color = 'red'
                    self.get_logger().info("🔴 Traffic light RED - STOP IMMEDIATELY")
                    speed = 0.0
                elif raw_color == 'green':
                    self.state = "TRAFFIC_GREEN"
                    self.traffic_light_color = 'green'
                    self.get_logger().info("🟢 Traffic light GREEN")
                    # Continue normal driving
                else:
                    self.traffic_light_color = None
                self.prev_tl_bbox = tl_bbox
        else:
            if self.state in ["TRAFFIC_DETECTED", "TRAFFIC_RED", "TRAFFIC_GREEN"]:
                self.state = "DRIVING"
                self.traffic_light_color_determined = False
                self.traffic_light_bbox = None
                self.get_logger().info("Traffic light out of view, returning to DRIVING")
            if self.tl_passed and current_time - self.tl_pass_confirmed_time > 5.0:
                self.tl_passed = False

        # --- Other signs (stop, yield) ---
        height, width = frame.shape[:2]
        bottom_roi = white_mask[int(height*0.8):, :]
        white_pixels_bottom = cv2.countNonZero(bottom_roi)
        reached_white_line = white_pixels_bottom > self.white_line_threshold

        # Get obstacle status from auxiliary cameras
        with self.aux_lock:
            vision_obstacle = self.obstacle_detected_cam1 or self.obstacle_detected_cam3

        # State machine for stop/yield (from Code2, adapted with STOP_SIGN_WAIT)
        if self.state == "DRIVING":
            if hasattr(self, 'left_dist') and (self.left_dist < self.danger_distance or self.right_dist < self.danger_distance):
                speed = 0.2

            if sign is not None and self.approach_sign is None:
                if current_time < self.ignore_signs_until:
                    self.get_logger().info(f'⏸ Ignoring {sign} sign (cooldown until {self.ignore_signs_until:.1f})')
                else:
                    self.approach_sign = sign
                    self.state = "APPROACHING"
                    self.state_start_time = current_time
                    self.get_logger().info(f'🔍 Approaching {sign} sign, reducing speed to {self.approach_speed}')
                    speed = self.approach_speed

        elif self.state == "APPROACHING":
            speed = self.approach_speed
            steering = steering_raw

            if sign_area > self.sign_area_threshold or reached_white_line:
                self.get_logger().info(f'🎯 Reached sign (area={sign_area:.0f}, white={white_pixels_bottom})')
                if self.approach_sign == 'stop':
                    # التعديل هنا: الانتقال إلى STOP_SIGN_WAIT بدلاً من STOPPED مباشرة
                    self.state = "STOP_SIGN_WAIT"
                    self.state_start_time = current_time
                    # نستمر بنفس السرعة approach_speed خلال فترة الانتظار
                    speed = self.approach_speed
                    steering = steering_raw
                    self.get_logger().info(f'⏳ Stop sign: waiting {self.stop_sign_wait_duration} seconds before stopping')
                elif self.approach_sign == 'yield':
                    self.state = "YIELD_WAITING"
                    self.state_start_time = current_time
                    speed = 0.1
                    steering = steering_raw
                    self.pid_steering.reset()
                    self.activate_aux_cameras()
                    self.obstacle_persistent_start = None
                self.approach_sign = None

        elif self.state == "STOP_SIGN_WAIT":
            # في حالة الانتظار قبل التوقف، نستمر بنفس السرعة approach_speed
            speed = self.approach_speed
            steering = steering_raw
            elapsed = current_time - self.state_start_time
            if elapsed > self.stop_sign_wait_duration:
                # بعد انتهاء فترة الانتظار، ننتقل إلى التوقف الفعلي (STOPPED)
                if current_time - self.last_stop_time > self.stop_cooldown:
                    self.state = "STOPPED"
                    self.state_start_time = current_time
                    self.last_stop_time = current_time
                    speed = 0.0
                    steering = 0.0
                    self.pid_steering.reset()
                    self.activate_aux_cameras()
                    self.get_logger().info('🛑 STOPPED for 5 seconds, auxiliary cameras ON')
                else:
                    self.get_logger().info('🛑 Stop sign ignored due to cooldown')
                    self.state = "DRIVING"
                    speed = self.base_speed

        elif self.state == "STOPPED":
            speed = 0.0
            steering = 0.0
            if current_time - self.state_start_time > self.stop_duration:
                self.get_logger().info('✅ Stop completed, resuming normal driving (ignoring signs for 5s)')
                self.state = "DRIVING"
                self.pid_steering.reset()
                self.deactivate_aux_cameras()
                self.ignore_signs_until = current_time + 5.0
                self.get_logger().info(f'⏸ Ignoring signs until {self.ignore_signs_until:.1f}')

        elif self.state == "YIELD_WAITING":
            speed = 0.1
            steering = steering_raw
            front_clear = self.front_dist > 2.0 or self.front_dist < 0.3
            if front_clear and not vision_obstacle:
                self.get_logger().info('✅ No obstacle, return to normal')
                self.state = "DRIVING"
                speed = self.base_speed
                self.pid_steering.reset()
                self.deactivate_aux_cameras()
                self.obstacle_persistent_start = None
            else:
                if self.obstacle_persistent_start is None:
                    self.obstacle_persistent_start = current_time
                    self.get_logger().info('⏱ Obstacle persistent timer started')
                elif current_time - self.obstacle_persistent_start > self.obstacle_persistent_threshold:
                    self.get_logger().warn('⏰ Obstacle persisted >5s, overriding and continuing')
                    self.state = "DRIVING"
                    speed = self.base_speed * 0.5
                    self.pid_steering.reset()
                    self.deactivate_aux_cameras()
                    self.obstacle_persistent_start = None
                else:
                    if not front_clear:
                        self.get_logger().info(f'🚧 Lidar {self.front_dist:.2f}m')
                    if vision_obstacle:
                        self.get_logger().info('🚧 Vision obstacle')
                    if current_time - self.state_start_time > self.yield_max_wait:
                        self.get_logger().warn('⏰ Yield timeout, proceed')
                        self.state = "DRIVING"
                        speed = self.base_speed * 0.5
                        self.pid_steering.reset()
                        self.deactivate_aux_cameras()
                        self.obstacle_persistent_start = None

        lane_frame = None
        if self.show_debug:
            sign_display = sign if not traffic_light_present else f"TL:{self.traffic_light_color}"
            lane_frame = self.create_lane_debug_frame(frame, road_mask, white_mask, lane_error, steering, speed,
                                                      sign_display, self.state, white_pixels_bottom)
        return steering, speed, lane_frame

    # --------------------------------------------------------
    # Activate auxiliary cameras (from Code2)
    # --------------------------------------------------------
    def activate_aux_cameras(self):
        if not self.cam1_active:
            try:
                self.myCam1 = Camera2D(cameraId="0@tcpip://localhost:18961",
                                       frameWidth=640, frameHeight=480, frameRate=30)
                self.cam1_active = True
                self.get_logger().info('📷 Cam1 activated')
            except Exception as e:
                self.get_logger().error(f'Failed to activate Cam1: {e}')

        if not self.cam3_active:
            try:
                self.myCam3 = Camera2D(cameraId="2@tcpip://localhost:18963",
                                       frameWidth=640, frameHeight=480, frameRate=30)
                self.cam3_active = True
                self.get_logger().info('📷 Cam3 activated')
            except Exception as e:
                self.get_logger().error(f'Failed to activate Cam3: {e}')

    # --------------------------------------------------------
    # Deactivate auxiliary cameras (from Code2)
    # --------------------------------------------------------
    def deactivate_aux_cameras(self):
        if self.cam1_active:
            self.myCam1.terminate()
            self.myCam1 = None
            self.cam1_active = False
            self.get_logger().info('📷 Cam1 deactivated')
            try:
                cv2.destroyWindow("Cam1 - Obstacle Check")
            except cv2.error:
                pass
        if self.cam3_active:
            self.myCam3.terminate()
            self.myCam3 = None
            self.cam3_active = False
            self.get_logger().info('📷 Cam3 deactivated')
            try:
                cv2.destroyWindow("Cam3 - Obstacle Check")
            except cv2.error:
                pass
        with self.aux_lock:
            self.obstacle_detected_cam1 = False
            self.obstacle_detected_cam3 = False

    # --------------------------------------------------------
    # Detect obstacles using YOLO (auxiliary) (from Code2)
    # --------------------------------------------------------
    def detect_obstacles_with_yolo(self, frame, cam_id=1):
        if self.yolo_net is None:
            return self.detect_obstacles_fallback(frame, cam_id)
        detections = self.detect_with_yolo(frame)
        obstacle = False
        if detections:
            for d in detections:
                relevant = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
                if d['class_name'] in relevant:
                    obstacle = True
                    x, y, w, h = d['box']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f"{d['class_name']} {d['confidence']:.2f}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return obstacle, frame

    def detect_obstacles_fallback(self, frame, cam_id=1):
        bg = self.bg_subtractor1 if cam_id == 1 else self.bg_subtractor3
        fg = bg.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacle = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                obstacle = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Obstacle", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return obstacle, frame

    # --------------------------------------------------------
    # Sign image callback (from Code1, with traffic light detection)
    # --------------------------------------------------------
    def sign_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn('Received empty image in sign_image_callback', throttle_duration_sec=1.0)
                return

            sign, area = self.detect_traffic_sign_with_area(cv_image)
            tl_bbox, tl_area = self.detect_traffic_light(cv_image)

            with self.sign_lock:
                self.detected_sign = sign
                self.detected_sign_area = area
                self.last_sign_frame = cv_image.copy()
                if tl_bbox is not None:
                    self.traffic_light_bbox = tl_bbox
                    self.traffic_light_area = tl_area
                    self.traffic_light_last_seen = self.get_clock().now().nanoseconds / 1e9
        except Exception as e:
            self.get_logger().error(f'Error in sign_image_callback: {e}')

    # --------------------------------------------------------
    # Combined sign detection (stop/yield) returning (sign, area) (from Code1)
    # --------------------------------------------------------
    def detect_traffic_sign_with_area(self, frame):
        if self.yolo_net is not None:
            detections = self.detect_with_yolo(frame)
            if detections:
                sign_vertices = {'stop': 8, 'yield': 3}
                for d in detections:
                    class_name = d['class_name'].lower()
                    bbox = d['box']
                    area = bbox[2] * bbox[3]
                    if 'stop' in class_name:
                        num_vertices = self.estimate_shape_from_bbox(bbox, frame)
                        if num_vertices and abs(num_vertices - sign_vertices['stop']) <= 2:
                            return 'stop', area
                    elif 'yield' in class_name or 'give way' in class_name:
                        num_vertices = self.estimate_shape_from_bbox(bbox, frame)
                        if num_vertices and abs(num_vertices - sign_vertices['yield']) <= 1:
                            return 'yield', area
                    elif 'stop' in class_name:
                        return 'stop', area
                    elif 'yield' in class_name:
                        return 'yield', area

        return self.detect_traffic_sign_by_shape_with_area(frame)

    # --------------------------------------------------------
    # Shape-based sign detection (stop/yield) (from Code1)
    # --------------------------------------------------------
    def detect_traffic_sign_by_shape_with_area(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]

        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        roi = red_mask[0:int(height*0.5), :]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_sign = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            num_vertices = len(approx)

            if 7 <= num_vertices <= 9:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 1.2:
                    if area > best_area:
                        best_area = area
                        best_sign = 'stop'
            elif num_vertices == 3:
                if area > best_area:
                    best_area = area
                    best_sign = 'yield'

        if best_sign:
            return best_sign, best_area
        else:
            return self.detect_traffic_sign_simple_with_area(frame)

    # --------------------------------------------------------
    # Simple color-based detection (fallback) (from Code1)
    # --------------------------------------------------------
    def detect_traffic_sign_simple_with_area(self, frame):
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
            return 'stop', red_pixels
        elif red_pixels > 200 and white_pixels < 200:
            return 'yield', red_pixels
        return None, 0

    # --------------------------------------------------------
    # Lidar callback (from Code2)
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
        raw_front = np.min(ranges[front_indices])
        self.front_dist_history.append(raw_front)
        self.front_dist = np.median(self.front_dist_history)

    # --------------------------------------------------------
    # Black road detection (from Code2, similar to Code1)
    # --------------------------------------------------------
    def detect_black_road_error(self, image):
        height, width = image.shape[:2]
        target_x = width // 2
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        road_mask = cv2.bitwise_and(black_mask, cv2.bitwise_not(white_mask))
        kernel = np.ones((5,5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        roi_height = int(height * 0.4)
        roi_y_start = height - roi_height
        slices = 5
        slice_height = roi_height // slices
        points = []
        for i in range(slices):
            y_start = roi_y_start + i * slice_height
            y_end = y_start + slice_height
            slice_mask = road_mask[y_start:y_end, :]
            black_pixels = np.where(slice_mask > 0)
            if len(black_pixels[1]) > 100:
                avg_x = np.mean(black_pixels[1])
                avg_y = (y_start + y_end) // 2
                points.append((avg_x, avg_y))

        if len(points) > 0:
            closest_point = max(points, key=lambda p: p[1])
            road_center_x = closest_point[0]
            error = road_center_x - target_x
        else:
            error = 0.0
        return error, road_mask, white_mask

    # --------------------------------------------------------
    # Debug frame (from Code2)
    # --------------------------------------------------------
    def create_lane_debug_frame(self, image, road_mask, white_mask, error, steering, speed, sign, state, white_pixels):
        try:
            height, width = image.shape[:2]
            target_x = width // 2
            road_vis = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
            white_vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
            road_vis[road_mask > 0] = [0, 255, 0]
            white_vis[white_mask > 0] = [255, 0, 0]
            combined = cv2.addWeighted(image, 0.7, road_vis, 0.3, 0)
            combined = cv2.addWeighted(combined, 1, white_vis, 0.3, 0)

            cv2.line(combined, (target_x, 0), (target_x, height), (255, 255, 0), 1)
            if abs(error) > 1:
                cv2.line(combined, (target_x, height-20),
                         (int(target_x + error), height-20-50), (0, 255, 255), 2)

            info = [
                f"Steering: {steering:.2f} rad",
                f"Error: {error:.1f} px",
                f"Speed: {speed:.2f} m/s",
                f"State: {state}",
                f"Sign: {sign if sign else 'None'}",
                f"White Pixels: {white_pixels}"
            ]
            for i, txt in enumerate(info):
                cv2.putText(combined, txt, (10, 30 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Green = Road (Black)", (10, height-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(combined, "Red = Sidewalk (White)", (10, height-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            return combined
        except Exception as e:
            self.get_logger().error(f'Error in debug display: {e}')
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
        node.get_logger().info('👋 Shutting down...')
    finally:
        stop_msg = MotorCommands()
        stop_msg.motor_names = ['steering_angle', 'motor_throttle']
        stop_msg.values = [0.0, 0.0]
        node.pub_cmd.publish(stop_msg)

        node.myCam4.terminate()
        if node.cam1_active:
            node.myCam1.terminate()
        if node.cam3_active:
            node.myCam3.terminate()

        cv2.destroyAllWindows()

        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()