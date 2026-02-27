import rclpy
from rclpy.node import Node
from qcar2_interfaces.msg import MotorCommands # استدعاء نوع الرسالة الخاص

class SimpleDriver(Node):
    def __init__(self):
        super().__init__('simple_driver_node')
        # تعريف الناشر (Publisher) على قناة الماتور
        self.publisher_ = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)
        
        # مؤقت لرسال الأمر كل 0.1 ثانية (10Hz)
        timer_period = 0.1  
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Driver Node Started! 🚀')

    def timer_callback(self):
        msg = MotorCommands()
        # تحديد الأجزاء التي نريد تحريكها
        msg.motor_names = ['motor_throttle', 'steering_angle']
        # تحديد القيم: سرعة 1.0 للأمام، زاوية 0.0 (دغري)
        msg.values = [1.0, 0.0]
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    driver = SimpleDriver()
    
    try:
        rclpy.spin(driver)
    except KeyboardInterrupt:
        pass
    finally:
        # إيقاف السيارة عند الخروج
        stop_msg = MotorCommands()
        stop_msg.motor_names = ['motor_throttle', 'steering_angle']
        stop_msg.values = [0.0, 0.0]
        driver.publisher_.publish(stop_msg)
        
        driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
