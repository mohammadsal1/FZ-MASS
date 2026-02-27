def __init__(self):
    super().__init__('traffic_aware_black_road_follower')

    # ------------------- المعاملات القابلة للضبط (Parameters) -------------------
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

    # Parameters للكاميرا الرئيسية (myCam4)
    self.declare_parameter('camera_id', '3@tcpip://localhost:18964')
    self.declare_parameter('frame_width', 640)
    self.declare_parameter('frame_height', 480)
    self.declare_parameter('frame_rate', 30)

    # YOLO paths
    self.declare_parameter('yolo_cfg', '/path/to/yolov4-tiny.cfg')
    self.declare_parameter('yolo_weights', '/path/to/yolov4-tiny.weights')
    self.declare_parameter('yolo_names', '/path/to/coco.names')
    self.declare_parameter('yolo_confidence', 0.5)
    self.declare_parameter('yolo_nms', 0.4)

    # Traffic light parameters
    self.declare_parameter('traffic_light_area_threshold', 5000)
    self.declare_parameter('traffic_light_color_delay', 3.0)
    self.declare_parameter('traffic_light_yellow_stop_delay', 1.0)

    # Stop sign wait
    self.declare_parameter('stop_sign_wait_duration', 2.0)

    # قراءة القيم
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
    self.tl_yellow_stop_delay = self.get_parameter('traffic_light_yellow_stop_delay').value
    self.stop_sign_wait_duration = self.get_parameter('stop_sign_wait_duration').value

    camera_id = self.get_parameter('camera_id').value
    frame_width = self.get_parameter('frame_width').value
    frame_height = self.get_parameter('frame_height').value
    frame_rate = self.get_parameter('frame_rate').value

    # ------------------- Publishers & Subscribers -------------------
    self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
    self.pub_cmd = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)

    # Subscription للكاميرا الثانية (للكشف عن الإشارات)
    self.sub_sign_cam = self.create_subscription(
        Image, '/camera/color_image', self.sign_image_callback, 10)
    self.bridge = CvBridge()

    # ------------------- التهيئة للملاحة (Navigation) -------------------
    # نقاط الطريق حسب السيناريو (تعديل الإحداثيات حسب بيئتك)
    self.waypoints = [
        {'x': 0.125, 'y': 4.395, 'action': 'pickup', 'led': 'blue'},
        {'x': -0.905, 'y': 0.800, 'action': 'dropoff', 'led': 'orange'},
        {'x': 0.0, 'y': 0.0, 'action': 'hub', 'led': 'magenta'}
    ]
    self.current_wp_index = 0          # الفهرس الحالي
    self.navigation_mode = False       # هل نحن في وضع التنقل بين النقاط؟
    self.arrival_threshold = 0.15      # نصف قطر الوصول بالمتر
    self.nav_gain = 0.6                # معامل تصحيح التوجيه (يحتاج ضبط)

    # متغيرات لتخزين موقع السيارة (يتم تحديثها من callback أو timer)
    self.current_x = 0.0
    self.current_y = 0.0
    self.current_yaw = 0.0

    # ------------------- قراءة الموقع (أضف إحدى الطريقتين) -------------------
    # الطريقة الأولى: إذا كان هناك topic للموقع (مثل /qcar2/pose) لاحقاً
    # self.sub_pose = self.create_subscription(Pose2D, '/qcar2/pose', self.pose_callback, 10)

    # الطريقة الثانية: استخدام QCar لقراءة الموقع مباشرة (قم بإلغاء التعليق إذا أردت)
    # from pal.products.qcar import QCar
    # self.car = QCar()  # قد تحتاج إلى وضع mode مناسب للمحاكاة
    # self.create_timer(0.1, self.update_pose_from_qcar)

    # ------------------- كاميرا myCam4 لمتابعة الخط -------------------
    from pal.utilities.vision import Camera2D
    self.myCam4 = Camera2D(
        cameraId=camera_id,
        frameWidth=frame_width,
        frameHeight=frame_height,
        frameRate=frame_rate
    )
    self.get_logger().info(f'📷 Camera myCam4 initialized with ID: {camera_id}')

    # كاميرات مساعدة (للعوائق)
    self.myCam1 = None
    self.myCam3 = None
    self.cam1_active = False
    self.cam3_active = False

    # Background subtractors
    self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2()
    self.bg_subtractor3 = cv2.createBackgroundSubtractorMOG2()

    # ------------------- YOLO initialization -------------------
    self.yolo_net = None
    self.yolo_classes = []
    self.yolo_confidence = self.get_parameter('yolo_confidence').value
    self.yolo_nms = self.get_parameter('yolo_nms').value
    self.load_yolo_model()

    # ------------------- PID controller -------------------
    self.pid_steering = PIDController(
        kp=self.kp,
        ki=self.ki,
        kd=self.kd,
        integral_limit=self.integral_limit,
        output_limit=self.output_limit
    )

    # ------------------- State machine -------------------
    self.state = "DRIVING"
    self.state_start_time = 0.0
    self.stop_duration = 5.0
    self.yield_max_wait = 5.0
    self.last_stop_time = 0.0
    self.stop_cooldown = 5.0

    # Approach variables
    self.approach_speed = 0.275
    self.sign_area_threshold = 2000
    self.white_line_threshold = 1000
    self.approach_sign = None
    self.obstacle_persistent_start = None
    self.obstacle_persistent_threshold = 5.0
    self.ignore_signs_until = 0.0

    # Lidar distances
    self.raw_front_dist = 100.0
    self.front_dist = 100.0
    from collections import deque
    self.front_dist_history = deque(maxlen=5)

    # Control variables
    import threading
    self.lock = threading.Lock()
    self.current_steering = 0.0
    self.current_speed = 0.0

    # Detected sign
    self.detected_sign = None
    self.detected_sign_area = 0.0
    self.last_sign_frame = None
    self.sign_lock = threading.Lock()

    # Traffic light variables
    self.traffic_light_bbox = None
    self.traffic_light_area = 0.0
    self.traffic_light_color = None
    self.traffic_light_first_detected_time = 0.0
    self.traffic_light_color_determined = False
    self.traffic_light_last_seen = 0.0
    self.traffic_light_yellow_stop_time = None
    self.tl_color_history = deque(maxlen=10)
    self.tl_raw_color = None
    self.tl_center_tolerance = 100

    # Obstacle detection from auxiliary cameras
    self.obstacle_detected_cam1 = False
    self.obstacle_detected_cam3 = False
    self.aux_lock = threading.Lock()

    # Camera status
    self.camera_ok = False

    # Timers
    self.timer = self.create_timer(0.1, self.timer_callback)
    self.create_timer(2.0, self.check_health)

    # بدء التنقل تلقائياً بعد ثانيتين (اختياري)
    self.create_timer(2.0, self.start_navigation)

    self.get_logger().info('🚗 Approach speed 0.275, decision based on sign area OR white line. Speed=0.4')
    self.get_logger().info('🚦 Traffic light detection enabled (improved).')
    self.get_logger().info(f'⏱ Stop sign wait duration: {self.stop_sign_wait_duration} seconds')
    self.get_logger().info('🧭 Navigation waypoints loaded. Waiting for pose data...')
