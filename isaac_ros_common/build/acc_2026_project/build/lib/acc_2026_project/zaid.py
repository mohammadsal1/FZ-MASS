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
#  دوال المسار (Hough Transform)
# ------------------------------------------------------------
def region_selection(image):
    try:
        mask = np.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        rows, cols = image.shape[:2]
        bottom_left  = [cols * 0.1, rows * 0.95]
        top_left     = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right    = [cols * 0.6, rows * 0.6]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    except Exception as e:
        # في حالة الخطأ نعيد صورة سوداء بنفس الأبعاد
        return np.zeros_like(image)

def hough_transform(image):
    try:
        rho = 1
        theta = np.pi/180
        threshold = 20
        minLineLength = 20
        maxLineGap = 500
        return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                               minLineLength=minLineLength, maxLineGap=maxLineGap)
    except:
        return None

def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append(length)
    except:
        pass
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    try:
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return ((x1, int(y1)), (x2, int(y2)))
    except:
        return None

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    try:
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line, color, thickness)
        return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
    except:
        return image

# ------------------------------------------------------------
#  Main Node
# ------------------------------------------------------------
class TrafficAwareLaneFollower(Node):
    def __init__(self):
        super().__init__('traffic_aware_lane_follower')

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

        # Camera parameters (only myCam4 now)
        self.declare_parameter('camera_id', '3@tcpip://localhost:18964')
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('frame_rate', 30)

        # YOLO model paths
        self.declare_parameter('yolo_cfg', '/path/to/yolov4-tiny.cfg')
        self.declare_parameter('yolo_weights', '/path/to/yolov4-tiny.weights')
        self.declare_parameter('yolo_names', '/path/to/coco.names')
        self.declare_parameter('yolo_confidence', 0.5)
        self.declare_parameter('yolo_nms', 0.4)

        # Traffic light specific parameters
        self.declare_parameter('traffic_light_area_threshold', 5000)
        self.declare_parameter('traffic_light_color_delay', 3.0)

        # Stop sign wait duration
        self.declare_parameter('stop_sign_wait_duration', 2.0)
        self.declare_parameter('stop_duration', 5.0)
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

        self.bridge = CvBridge()

        # Initialize myCam4 (the only camera for lane following and sign detection)
        try:
            self.myCam4 = Camera2D(
                cameraId=camera_id,
                frameWidth=frame_width,
                frameHeight=frame_height,
                frameRate=frame_rate
            )
            self.get_logger().info(f'📷 Camera myCam4 initialized with ID: {camera_id}')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize myCam4: {e}')
            self.myCam4 = None

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

        # State machine variables
        self.state = "DRIVING"
        self.state_start_time = 0.0
        self.last_stop_time = 0.0
        self.stop_cooldown = 5.0
        self.ignore_signs_until = 0.0

        # Approach variables
        self.approach_speed = 0.275
        self.sign_area_threshold = 2000
        self.white_line_threshold = 1000
        self.approach_sign = None
        self.obstacle_persistent_start = None
        self.obstacle_persistent_threshold = 5.0

        # Lidar distances with filtering
        self.raw_front_dist = 100.0
        self.front_dist = 100.0
        self.front_dist_history = deque(maxlen=5)
        self.left_dist = 100.0
        self.right_dist = 100.0

        # Control variables
        self.lock = threading.Lock()
        self.current_steering = 0.0
        self.current_speed = 0.0

        # Traffic light state variables
        self.traffic_light_color = None
        self.traffic_light_color_determined = False
        self.traffic_light_last_seen = 0.0
        self.prev_tl_bbox = None
        self.tl_passed = False
        self.tl_pass_confirmed_time = 0.0

        # Obstacle detection results from auxiliary cameras
        self.obstacle_detected_cam1 = False
        self.obstacle_detected_cam3 = False
        self.aux_lock = threading.Lock()

        # Camera status
        self.camera_ok = False

        # Timer
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.create_timer(2.0, self.check_health)

        self.get_logger().info('🚗 Using Hough lane detection + YOLO signs on myCam4 only.')
        self.get_logger().info('🚦 Traffic light detection enabled (red/green only).')
        self.get_logger().info(f'⏱ Stop sign wait duration: {self.stop_sign_wait_duration} seconds')

    # --------------------------------------------------------
    # Load YOLO model
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
    # YOLO inference
    # --------------------------------------------------------
    def detect_with_yolo(self, frame):
        if self.yolo_net is None or frame is None:
            return None
        try:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.yolo_net.setInput(blob)

            layer_names = self.yolo_net.getUnconnectedOutLayersNames()
            outputs = self.yolo_net.forward(layer_names)

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
        except Exception as e:
            self.get_logger().error(f'Error in YOLO detection: {e}')
            return None

    # --------------------------------------------------------
    # Estimate shape from bounding box
    # --------------------------------------------------------
    def estimate_shape_from_bbox(self, bbox, frame, color_range=None):
        if bbox is None or frame is None:
            return None
        try:
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
        except Exception as e:
            self.get_logger().error(f'Error in shape estimation: {e}')
            return None

    # --------------------------------------------------------
    # Traffic light detection using YOLO + shape fallback
    # --------------------------------------------------------
    def detect_traffic_light_shape_based(self, frame):
        if frame is None:
            return None, 0
        try:
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

            return best_bbox, best_area
        except Exception as e:
            self.get_logger().error(f'Error in shape-based traffic light detection: {e}')
            return None, 0

    def detect_traffic_light(self, frame):
        if frame is None:
            return None, 0
        try:
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
        except Exception as e:
            self.get_logger().error(f'Error in traffic light detection: {e}')
            return None, 0

    # --------------------------------------------------------
    # Analyze traffic light color (red/green only)
    # --------------------------------------------------------
    def analyze_traffic_light_color(self, bbox, frame):
        if bbox is None or frame is None:
            return None
        try:
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

            half = h // 2
            roi_top = roi[0:half, :]
            roi_bottom = roi[half:h, :]

            hsv_top = cv2.cvtColor(roi_top, cv2.COLOR_BGR2HSV)
            hsv_bottom = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 150, 150])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 150, 150])
            upper_red2 = np.array([180, 255, 255])

            lower_green = np.array([40, 120, 120])
            upper_green = np.array([80, 255, 255])

            mask_red_top1 = cv2.inRange(hsv_top, lower_red1, upper_red1)
            mask_red_top2 = cv2.inRange(hsv_top, lower_red2, upper_red2)
            mask_red_top = cv2.bitwise_or(mask_red_top1, mask_red_top2)
            red_top = cv2.countNonZero(mask_red_top)

            mask_green_bottom = cv2.inRange(hsv_bottom, lower_green, upper_green)
            green_bottom = cv2.countNonZero(mask_green_bottom)

            self.get_logger().info(f'🔴 Red top: {red_top}, 🟢 Green bottom: {green_bottom}')

            threshold = 5
            if red_top > green_bottom and red_top > threshold:
                return 'red'
            elif green_bottom > red_top and green_bottom > threshold:
                return 'green'
            elif red_top > threshold and green_bottom <= threshold:
                return 'red'
            elif green_bottom > threshold and red_top <= threshold:
                return 'green'
            else:
                return None
        except Exception as e:
            self.get_logger().error(f'Error in color analysis: {e}')
            return None

    # --------------------------------------------------------
    # Combined sign detection (stop/yield) returning (sign, area)
    # --------------------------------------------------------
    def detect_traffic_sign_with_area(self, frame):
        if frame is None:
            return None, 0
        try:
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
        except Exception as e:
            self.get_logger().error(f'Error in sign detection: {e}')
            return None, 0

    # --------------------------------------------------------
    # Shape-based sign detection (stop/yield)
    # --------------------------------------------------------
    def detect_traffic_sign_by_shape_with_area(self, frame):
        if frame is None:
            return None, 0
        try:
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
        except Exception as e:
            self.get_logger().error(f'Error in shape-based sign detection: {e}')
            return None, 0

    # --------------------------------------------------------
    # Simple color-based detection (fallback)
    # --------------------------------------------------------
    def detect_traffic_sign_simple_with_area(self, frame):
        if frame is None:
            return None, 0
        try:
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
        except Exception as e:
            self.get_logger().error(f'Error in simple sign detection: {e}')
            return None, 0

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------
    def check_health(self):
        if not self.camera_ok:
            self.get_logger().warn('⏳ Waiting for myCam4 to provide images...')

    # --------------------------------------------------------
    # Timer callback (main processing loop)
    # --------------------------------------------------------
    def timer_callback(self):
        try:
            if self.myCam4 is None:
                return

            flag = self.myCam4.read()
            if flag:
                self.camera_ok = True
                frame = self.myCam4.imageData
                steering, speed, lane_frame = self.process_frame(frame)
                with self.lock:
                    self.current_steering = steering
                    self.current_speed = speed
                if self.show_debug and lane_frame is not None:
                    cv2.imshow("Lane Following with Signs", lane_frame)
                    cv2.waitKey(1)
            else:
                self.camera_ok = False
                self.get_logger().warn('⚠️ Failed to read image from myCam4', throttle_duration_sec=2.0)

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
        except Exception as e:
            self.get_logger().error(f'Unhandled exception in timer_callback: {e}', throttle_duration_sec=5.0)

    # --------------------------------------------------------
    # Process a single frame: lane detection + sign detection + state machine
    # --------------------------------------------------------
    def process_frame(self, frame):
        # قيم افتراضية
        steering = 0.0
        speed = self.base_speed
        lane_error = 0.0
        left_line = right_line = None
        edges = region = None
        sign = None
        sign_area = 0.0
        tl_bbox = None
        tl_color = None
        white_pixels_bottom = 0
        white_mask = None

        try:
            # 1. حساب خطأ المسار باستخدام Hough Transform
            lane_error, left_line, right_line, edges, region, lines = self.detect_lane_error(frame)
        except Exception as e:
            self.get_logger().error(f'Error in lane detection: {e}')

        current_time = self.get_clock().now().nanoseconds / 1e9
        pid_output = self.pid_steering.update(lane_error, current_time)
        steering_lane = -pid_output

        # 2. تجنب العوائق بالـ Lidar
        steering_avoid = 0.0
        if hasattr(self, 'left_dist') and self.left_dist < self.safe_distance:
            steering_avoid += (self.safe_distance - self.left_dist) * self.wall_gain
        if hasattr(self, 'right_dist') and self.right_dist < self.safe_distance:
            steering_avoid -= (self.safe_distance - self.right_dist) * self.wall_gain
        steering_raw = steering_lane + steering_avoid
        steering_raw = np.clip(steering_raw, -self.max_angle, self.max_angle)
        steering = steering_raw

        # 3. كشف الإشارات
        try:
            sign, sign_area = self.detect_traffic_sign_with_area(frame)
        except Exception as e:
            self.get_logger().error(f'Error in sign detection: {e}')

        try:
            tl_bbox, tl_area = self.detect_traffic_light(frame)
        except Exception as e:
            self.get_logger().error(f'Error in traffic light detection: {e}')

        traffic_light_present = (tl_bbox is not None)

        if traffic_light_present:
            self.traffic_light_last_seen = current_time

        if traffic_light_present:
            try:
                tl_color = self.analyze_traffic_light_color(tl_bbox, frame)
                self.traffic_light_color = tl_color
            except Exception as e:
                self.get_logger().error(f'Error in traffic light color analysis: {e}')

        # 4. معالجة حالة traffic light
        try:
            if traffic_light_present:
                height = frame.shape[0]
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
                    if tl_color == 'red':
                        self.state = "TRAFFIC_RED"
                        self.get_logger().info("🔴 Traffic light RED - STOP IMMEDIATELY")
                        speed = 0.0
                    elif tl_color == 'green':
                        self.state = "TRAFFIC_GREEN"
                        self.get_logger().info("🟢 Traffic light GREEN")
                    else:
                        self.traffic_light_color = None
                    self.prev_tl_bbox = tl_bbox
            else:
                if self.state in ["TRAFFIC_DETECTED", "TRAFFIC_RED", "TRAFFIC_GREEN"]:
                    self.state = "DRIVING"
                    self.traffic_light_color_determined = False
                    self.get_logger().info("Traffic light out of view, returning to DRIVING")
                if self.tl_passed and current_time - self.tl_pass_confirmed_time > 5.0:
                    self.tl_passed = False
        except Exception as e:
            self.get_logger().error(f'Error in traffic light state handling: {e}')

        # 5. كشف الخط الأبيض
        try:
            white_mask = self.get_white_mask(frame)
            height, width = frame.shape[:2]
            bottom_roi = white_mask[int(height*0.8):, :]
            white_pixels_bottom = cv2.countNonZero(bottom_roi)
            reached_white_line = white_pixels_bottom > self.white_line_threshold
        except Exception as e:
            self.get_logger().error(f'Error in white line detection: {e}')
            white_pixels_bottom = 0
            reached_white_line = False

        # الحصول على حالة العوائق من الكاميرات المساعدة
        with self.aux_lock:
            vision_obstacle = self.obstacle_detected_cam1 or self.obstacle_detected_cam3

        # 6. آلة الحالات (stop, yield)
        try:
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
                        self.state = "STOP_SIGN_WAIT"
                        self.state_start_time = current_time
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
                speed = self.approach_speed
                steering = steering_raw
                elapsed = current_time - self.state_start_time
                if elapsed > self.stop_sign_wait_duration:
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
        except Exception as e:
            self.get_logger().error(f'Error in state machine: {e}')
            self.state = "DRIVING"
            speed = self.base_speed
            steering = steering_raw

        # 7. إنشاء إطار debug
        lane_frame = None
        if self.show_debug:
            try:
                sign_display = sign if sign else "None"
                if traffic_light_present and tl_color:
                    sign_display = f"TL:{tl_color}"
                lane_frame = self.create_lane_debug_frame(frame, left_line, right_line, edges, region,
                                                          lane_error, steering, speed,
                                                          sign_display, self.state, white_pixels_bottom,
                                                          tl_bbox, tl_color, white_mask)
            except Exception as e:
                self.get_logger().error(f'Error creating debug frame: {e}')

        return steering, speed, lane_frame

    # --------------------------------------------------------
    # Detect lane error using Hough lines
    # --------------------------------------------------------
    def detect_lane_error(self, image):
        if image is None or image.size == 0:
            return 0.0, None, None, None, None, None
        try:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel_size = 5
            blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
            low_t = 50
            high_t = 150
            edges = cv2.Canny(blur, low_t, high_t)
            region = region_selection(edges)
            lines = hough_transform(region)

            if lines is None or len(lines) == 0:
                return 0.0, None, None, edges, region, lines

            left_line, right_line = lane_lines(image, lines)

            height, width = image.shape[:2]
            target_x = width // 2

            if left_line is not None and right_line is not None:
                (x1_left, y1_left), (x2_left, y2_left) = left_line
                (x1_right, y1_right), (x2_right, y2_right) = right_line
                lane_center_x = (x1_left + x1_right) // 2
                error = lane_center_x - target_x
            elif left_line is not None:
                (x1_left, y1_left), (x2_left, y2_left) = left_line
                lane_center_x = x1_left + 150
                error = lane_center_x - target_x
            elif right_line is not None:
                (x1_right, y1_right), (x2_right, y2_right) = right_line
                lane_center_x = x1_right - 150
                error = lane_center_x - target_x
            else:
                error = 0.0

            return error, left_line, right_line, edges, region, lines
        except Exception as e:
            self.get_logger().error(f'Exception in detect_lane_error: {e}')
            return 0.0, None, None, None, None, None

    # --------------------------------------------------------
    # Get white mask for stop line detection
    # --------------------------------------------------------
    def get_white_mask(self, frame):
        if frame is None:
            return np.zeros((10,10), dtype=np.uint8)
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            return white_mask
        except Exception as e:
            self.get_logger().error(f'Error in get_white_mask: {e}')
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    # --------------------------------------------------------
    # Activate auxiliary cameras
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
    # Deactivate auxiliary cameras
    # --------------------------------------------------------
    def deactivate_aux_cameras(self):
        if self.cam1_active:
            try:
                self.myCam1.terminate()
                self.myCam1 = None
                self.cam1_active = False
                self.get_logger().info('📷 Cam1 deactivated')
            except Exception as e:
                self.get_logger().error(f'Error deactivating Cam1: {e}')
            try:
                cv2.destroyWindow("Cam1 - Obstacle Check")
            except cv2.error:
                pass
        if self.cam3_active:
            try:
                self.myCam3.terminate()
                self.myCam3 = None
                self.cam3_active = False
                self.get_logger().info('📷 Cam3 deactivated')
            except Exception as e:
                self.get_logger().error(f'Error deactivating Cam3: {e}')
            try:
                cv2.destroyWindow("Cam3 - Obstacle Check")
            except cv2.error:
                pass
        with self.aux_lock:
            self.obstacle_detected_cam1 = False
            self.obstacle_detected_cam3 = False

    # --------------------------------------------------------
    # Detect obstacles using YOLO (auxiliary)
    # --------------------------------------------------------
    def detect_obstacles_with_yolo(self, frame, cam_id=1):
        if frame is None:
            return False, frame
        try:
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
        except Exception as e:
            self.get_logger().error(f'Error in detect_obstacles_with_yolo: {e}')
            return False, frame

    def detect_obstacles_fallback(self, frame, cam_id=1):
        try:
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
        except Exception as e:
            self.get_logger().error(f'Error in fallback obstacle detection: {e}')
            return False, frame

    # --------------------------------------------------------
    # Lidar callback
    # --------------------------------------------------------
    def scan_callback(self, msg):
        try:
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
        except Exception as e:
            self.get_logger().error(f'Error in scan_callback: {e}')

    # --------------------------------------------------------
    # Create debug frame showing Hough lines and info
    # --------------------------------------------------------
    def create_lane_debug_frame(self, image, left_line, right_line, edges, region,
                                error, steering, speed, sign, state, white_pixels,
                                tl_bbox=None, tl_color=None, white_mask=None):
        if image is None:
            return None
        try:
            # رسم الخطوط على الصورة
            if left_line is not None or right_line is not None:
                lines = (left_line, right_line)
                display = draw_lane_lines(image, lines, color=[0, 0, 255], thickness=8)
            else:
                display = image.copy()

            # إضافة معلومات على الشاشة
            height, width = image.shape[:2]
            target_x = width // 2
            cv2.line(display, (target_x, 0), (target_x, height), (255, 255, 0), 1)

            info = [
                f"Steering: {steering:.2f} rad",
                f"Error: {error:.1f} px",
                f"Speed: {speed:.2f} m/s",
                f"State: {state}",
                f"Sign: {sign}",
                f"White Pixels: {white_pixels}"
            ]
            for i, txt in enumerate(info):
                cv2.putText(display, txt, (10, 30 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # إذا وجدت إشارة ضوئية، ارسم مستطيلاً حولها
            if tl_bbox is not None:
                x, y, w, h = tl_bbox
                color_tl = (0, 0, 255) if tl_color == 'red' else (0, 255, 0) if tl_color == 'green' else (255, 0, 0)
                cv2.rectangle(display, (x, y), (x+w, y+h), color_tl, 2)
                cv2.putText(display, f"TL {tl_color if tl_color else '?'}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tl, 2)

            # عرض صورة edges صغيرة في الزاوية إذا كانت موجودة
            if edges is not None and edges.size > 0:
                small_edges = cv2.resize(edges, (160, 120))
                small_edges_color = cv2.cvtColor(small_edges, cv2.COLOR_GRAY2BGR)
                display[10:130, -170:-10] = small_edges_color

            return display
        except Exception as e:
            self.get_logger().error(f'Error in create_lane_debug_frame: {e}')
            return None

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TrafficAwareLaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('👋 Shutting down...')
    finally:
        stop_msg = MotorCommands()
        stop_msg.motor_names = ['steering_angle', 'motor_throttle']
        stop_msg.values = [0.0, 0.0]
        try:
            node.pub_cmd.publish(stop_msg)
        except:
            pass

        try:
            node.myCam4.terminate()
        except:
            pass
        if node.cam1_active:
            try:
                node.myCam1.terminate()
            except:
                pass
        if node.cam3_active:
            try:
                node.myCam3.terminate()
            except:
                pass

        cv2.destroyAllWindows()

        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()