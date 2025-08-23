#!/usr/bin/env python3
# File: continuous_arm_monitor.py
# Simple script to continuously monitor XArm servo positions and status
# pip install xarm

import time
import sys
import xarm

def read_all_arm_data(arm):
    """Get battery voltage and each servo's position as a single formatted line."""
    parts = []
    
    # Get battery voltage
    try:
        v = arm.getBatteryVoltage()
        parts.append(f"v:{v}V")
    except Exception as e:
        parts.append(f"v:ERR")
    
    # Get servo positions
    for sid in range(1, 7):
        try:
            pos = arm.getPosition(sid, degrees=True)
            parts.append(f"{sid}:{pos}°")
        except Exception as e:
            parts.append(f"{sid}:ERR")
    
    return "  ".join(parts)

def main():
    print("Opening XArm controller over USB...")
    try:
        arm = xarm.Controller('USB')
    except Exception as e:
        print(f"Failed to open controller: {e}")
        sys.exit(1)

    print("Starting continuous monitoring... Press Ctrl+C to stop.")
    print("Reading arm data every 1 second...")
    print("Format: v:voltage  1:pos1°  2:pos2°  3:pos3°  4:pos4°  5:pos5°  6:pos6°")
    
    try:
        while True:
            data_line = read_all_arm_data(arm)
            print(data_line)
            time.sleep(1.0)  # Wait 1 second between readings
            
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    
    # Optional: cut torque on exit
    try:
        arm.servoOff()
    except Exception:
        pass
    
    print("Monitoring stopped. Goodbye.")

if __name__ == "__main__":
    main()
