import os
import sys

def check_files():
    print("🔍 جاري فحص بيئة العمل وملفات Quanser...\n")
    
    # 1. فحص ملف الإعدادات
    if os.path.exists('qcar_config.json'):
        print("✅ ملف qcar_config.json موجود.")
    else:
        print("❌ تحذير: ملف qcar_config.json مفقود! قد لا تعمل السيارة بدونه.")

    # 2. فحص مكتبة PAL (Product Abstraction Layer)
    try:
        import pal.utilities.vision
        print("✅ مكتبة PAL (الكاميرا) تم تحميلها بنجاح.")
    except ImportError as e:
        print(f"❌ فشل استيراد PAL: {e}")
        print("   -> تأكد من وجود مجلد 'pal' بجوار هذا السكريبت.")

    try:
        import pal.utilities.lidar
        print("✅ مكتبة PAL (اللايدار) تم تحميلها بنجاح.")
    except ImportError:
        print("❌ فشل استيراد اللايدار من PAL.")

    # 3. فحص مكتبة QCar
    try:
        import qcar
        print("✅ ملف qcar.py تم تحميله بنجاح.")
    except ImportError as e:
        print(f"❌ فشل استيراد qcar.py: {e}")

    # 4. فحص مكتبات ROS
    try:
        import rclpy
        print("✅ مكتبات ROS 2 (rclpy) جاهزة.")
    except ImportError:
        print("❌ بيئة ROS غير مفعلة (تأكد من sourcing).")

    print("\n------------------------------------------------")
    print("النتيجة النهائية:")
    if 'pal' in sys.modules and 'qcar' in sys.modules:
        print("🚀 كل الأنظمة جاهزة! يمكنك تشغيل lane_follower الآن.")
    else:
        print("⚠️ هناك ملفات ناقصة، يرجى مراجعة القائمة أعلاه.")

if __name__ == "__main__":
    check_files()
