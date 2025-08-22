#!/usr/bin/env python3
import hid

TARGET_VID = 0x0483
TARGET_PID = 0x5750

# Enumerate first
devinfo = None
for d in hid.enumerate():
    if d['vendor_id'] == TARGET_VID and d['product_id'] == TARGET_PID:
        devinfo = d
        break

if not devinfo:
    raise RuntimeError("Device not found")

print("Opening HID device at path:", devinfo['path'])

h = hid.device()
h.open_path(devinfo['path'])   # open by path (works with Linux's bytes path)
h.set_nonblocking(1)

print("Opened successfully!")

# Try to read a packet
data = h.read(64, timeout_ms=1000)   # 64-byte buffer, 1s timeout
print("Read:", data)

h.close()
