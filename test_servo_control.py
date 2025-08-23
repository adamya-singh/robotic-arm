#!/usr/bin/env python3
# File: interactive_servo_control.py
# pip install xarm

import time
import sys
import xarm

DURATION_MS = 800  # movement duration in milliseconds
WAIT_BETWEEN_MS = 200  # small extra pause after each move

def read_all_arm_data(arm):
    """Print battery voltage and each servo's position (in degrees)."""
    print("\n=== Arm status ===")
    try:
        v = arm.getBatteryVoltage()
        print(f"Battery voltage: {v}V")
    except Exception as e:
        print(f"Battery voltage read failed: {e}")
    for sid in range(1, 7):
        try:
            pos = arm.getPosition(sid, degrees=True)
            print(f"Servo {sid} position: {pos}°")
        except Exception as e:
            print(f"Servo {sid} position read failed: {e}")
    print("==================\n")

def prompt_servo_id():
    """Prompt for a servo id (1-6) or 'q' to quit. Returns int or None if quitting."""
    while True:
        s = input("Servo ID (1-6) or 'q' to quit: ").strip().lower()
        if s in ("q", "quit", "exit"):
            return None
        try:
            sid = int(s)
            if 1 <= sid <= 6:
                return sid
            print("Please enter a number from 1 to 6.")
        except ValueError:
            print("Invalid input. Enter 1-6, or 'q' to quit.")

def prompt_position_deg():
    """Prompt for a target position in degrees (float)."""
    while True:
        s = input("Target position (degrees, e.g. 0, 10, -30): ").strip().lower()
        if s in ("q", "quit", "exit"):
            return None
        try:
            deg = float(s)
            return deg
        except ValueError:
            print("Invalid number. Try again (or 'q' to quit).")

def main():
    print("Opening XArm controller over USB…")
    try:
        arm = xarm.Controller('USB')
    except Exception as e:
        print(f"Failed to open controller: {e}")
        sys.exit(1)

    # Show initial status
    read_all_arm_data(arm)

    print("Interactive mode. Press Ctrl+C or type 'q' at a prompt to exit.")
    try:
        while True:
            sid = prompt_servo_id()
            if sid is None:
                break

            pos_deg = prompt_position_deg()
            if pos_deg is None:
                break

            try:
                # Move the selected servo
                arm.setPosition(sid, pos_deg, duration=DURATION_MS, wait=True)
                time.sleep(WAIT_BETWEEN_MS / 1000.0)

                # Read back the position for confirmation
                try:
                    new_pos = arm.getPosition(sid, degrees=True)
                    print(f"Moved servo {sid} to ~{new_pos}° (requested {pos_deg}°).")
                except Exception as e_read:
                    print(f"Move done, but readback failed: {e_read}")

            except Exception as e_move:
                print(f"Move failed for servo {sid}: {e_move}")

    except KeyboardInterrupt:
        print("\nExiting on user interrupt…")

    # Optional: cut torque on exit
    try:
        arm.servoOff()
    except Exception:
        pass

    print("Goodbye.")

if __name__ == "__main__":
    main()