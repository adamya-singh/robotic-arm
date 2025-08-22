#!/usr/bin/env python3
# File: exercise_servos_writeonly.py
# pip install xarm
import time
import xarm

# Small, safe motion
DELTA_DEG = 10.0
DURATION_MS = 800
PAUSE_SEC = (DURATION_MS / 1000.0) + 0.2

def read_all_arm_data(arm):
    """Read and print all available data from the arm"""
    print("=== Reading all arm data ===")
    
    # Read battery voltage
    try:
        battery_voltage = arm.getBatteryVoltage()
        print(f"Battery voltage: {battery_voltage}V")
    except Exception as e:
        print(f"Battery voltage read failed: {e}")
    
    # Read position of each servo
    for sid in range(1, 7):
        try:
            current_pos = arm.getPosition(sid, degrees=True)
            print(f"Servo {sid} position: {current_pos}°")
        except Exception as e:
            print(f"Servo {sid} position read failed: {e}")
    
    print("=== End of arm data ===\n")

def main():
    # Open via USB HID (VID:PID 0483:5750)
    arm = xarm.Controller('USB')

    # Read all available data from the arm
    read_all_arm_data(arm)

    # Move each servo 1..6 a little forward, then back to 0°
    for sid in range(1, 7):
        try:
            # Go +DELTA
            arm.setPosition(sid, DELTA_DEG, duration=DURATION_MS, wait=False)
            time.sleep(PAUSE_SEC)

            # Return to 0°
            arm.setPosition(sid, 0.0, duration=DURATION_MS, wait=False)
            time.sleep(PAUSE_SEC)

            print(f"Servo {sid} moved +{DELTA_DEG}° and back.")
        except Exception as e:
            print(f"[S{sid}] Write failed: {e}")

    # Optional: cut torque (some boards accept this without needing a read)
    try:
        arm.servoOff()
    except Exception:
        pass

if __name__ == "__main__":
    main()