#!/usr/bin/env python3
# File: exercise_servos_writeonly.py
# pip install xarm
import time
import hid_compat
import xarm

# Small, safe motion
DELTA_DEG = 40.0
DURATION_MS = 800
PAUSE_SEC = (DURATION_MS / 1000.0) + 0.2

# Dual servo testing constants
DELTA_DEG_DUAL = 40.0
PAUSE_SEC_DUAL = (DURATION_MS / 1000.0) + 0.3

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

def move_two_servos_simultaneously(arm, sid1, pos1, sid2, pos2, duration_ms, wait=False):
    """Move two servos to specified positions simultaneously"""
    # Use the multi-servo API for true simultaneous movement
    arm.setPosition([[sid1, float(pos1)], [sid2, float(pos2)]], 
                    duration=duration_ms, wait=wait)

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
            
            # Read arm data after moving to +DELTA
            print(f"\n--- After moving servo {sid} to +{DELTA_DEG}° ---")
            read_all_arm_data(arm)

            # Return to 0°
            arm.setPosition(sid, 0.0, duration=DURATION_MS, wait=False)
            time.sleep(PAUSE_SEC)
            
            # Read arm data after returning to 0°
            print(f"\n--- After returning servo {sid} to 0° ---")
            read_all_arm_data(arm)

            print(f"Servo {sid} moved +{DELTA_DEG}° and back.")
        except Exception as e:
            print(f"[S{sid}] Write failed: {e}")

    # Test dual servo movements
    print("\n=== Testing dual servo movements ===")
    servo_pairs = [(1, 2), (3, 4), (5, 6)]
    for sid1, sid2 in servo_pairs:
        try:
            # Move both servos to +DELTA_DEG_DUAL
            move_two_servos_simultaneously(arm, sid1, DELTA_DEG_DUAL, sid2, DELTA_DEG_DUAL, DURATION_MS)
            time.sleep(PAUSE_SEC_DUAL)
            
            # Read data after moving to +DELTA_DEG_DUAL
            print(f"\n--- After moving servos {sid1} and {sid2} to +{DELTA_DEG_DUAL}° ---")
            read_all_arm_data(arm)
            
            # Move both back to 0°
            move_two_servos_simultaneously(arm, sid1, 0.0, sid2, 0.0, DURATION_MS)
            time.sleep(PAUSE_SEC_DUAL)
            
            # Read data after returning to 0°
            print(f"\n--- After returning servos {sid1} and {sid2} to 0° ---")
            read_all_arm_data(arm)
            
            print(f"Servos {sid1} and {sid2} moved +{DELTA_DEG_DUAL}° and back.")
        except Exception as e:
            print(f"[S{sid1},{sid2}] Dual movement failed: {e}")

    # Optional: cut torque (some boards accept this without needing a read)
    try:
        arm.servoOff()
    except Exception:
        pass

if __name__ == "__main__":
    main()
