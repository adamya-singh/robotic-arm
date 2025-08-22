#!/usr/bin/env python3
import sys, time, struct
import hid

# Adjust if lsusb shows different values:
VENDOR_ID  = 0x0483
PRODUCT_ID = 0x5750

# LewanSoul/Hiwonder-style frame:
# 0x55 0x55 | ID | LEN | CMD | PARAMS... | CHECK
# CHECK = (~(ID + LEN + CMD + sum(PARAMS))) & 0xFF
def checksum(payload_bytes):
    s = sum(payload_bytes) & 0xFF
    return (~s) & 0xFF

def build_frame(servo_id, cmd, params=b""):
    # LEN = number of bytes from CMD to end (CMD + params + CHECK)
    length = 1 + len(params) + 1
    body = bytes([servo_id, length, cmd]) + params
    chk = checksum(body)
    return b"\x55\x55" + body + bytes([chk])

def open_device(vendor=VENDOR_ID, product=PRODUCT_ID):
    dev = hid.device()
    dev.open(vendor, product)
    # Nonblocking makes reads return quickly
    dev.set_nonblocking(1)
    return dev

def hid_write(dev, data):
    # HID raw on Linux typically expects a leading Report ID byte.
    # If there is no report ID, use 0x00 as a prefix.
    # Many boards accept 64-byte reports; pad to 64.
    report_id = b"\x00"
    packet = report_id + data
    if len(packet) < 64:
        packet = packet + b"\x00"*(64 - len(packet))
    return dev.write(packet)

def hid_read(dev, timeout_ms=200):
    # Read a 64-byte report (adjust if your device uses a different size)
    t0 = time.time()
    while True:
        out = dev.read(64)
        if out:
            return bytes(out)
        if (time.time() - t0) * 1000 > timeout_ms:
            return b""

def pretty(b):
    return " ".join(f"{x:02X}" for x in b)

def main():
    try:
        dev = open_device()
        print("Opened HID device.")
    except Exception as e:
        print("Failed to open HID device:", e)
        sys.exit(1)

    # ---- Test 1: Ping servo ID=1 (common) ----
    # Many firmwares accept a PING-like command; if your board doesn’t,
    # we'll try a ReadPosition next which is widely supported.
    # Using CMD=0x1C as ReadPosition per common LewanSoul docs.
    # If ping isn't defined on your board, skip it without error.
    servo_ids = [1]  # expand if you know more IDs
    READ_POS_CMD = 0x1C

    for sid in servo_ids:
        # Build "ReadPosition" frame for ID=sid
        frame = build_frame(sid, READ_POS_CMD, b"")
        print(f"TX (ID={sid} ReadPosition):", pretty(frame))
        hid_write(dev, frame)

        # Expect a reply frame back (header 0x55 0x55 ...)
        resp = hid_read(dev, timeout_ms=300)
        if not resp:
            print(f"NO REPLY from ID={sid} (try other IDs or check power/chain).")
            continue

        print("RX raw:", pretty(resp))

        # Many HID firmwares wrap the bus frame inside the report.
        # Find the header 0x55 0x55 inside the 64-byte report:
        try:
            idx = resp.index(0x55)
            if idx+1 < len(resp) and resp[idx+1] == 0x55:
                pkt = resp[idx:]
            else:
                # fallback: try to find full header sequence
                hdr = bytes([0x55, 0x55])
                i2 = bytes(resp).find(hdr)
                if i2 >= 0:
                    pkt = resp[i2:]
                else:
                    pkt = resp
        except ValueError:
            pkt = resp

        # Minimal decode: [55 55 | ID | LEN | CMD | ...]
        if len(pkt) >= 6 and pkt[0] == 0x55 and pkt[1] == 0x55:
            rid, rlen, rcmd = pkt[2], pkt[3], pkt[4]
            params = pkt[5:5 + (rlen - 2)] if rlen >= 2 else b""
            print(f"Parsed: ID={rid} LEN={rlen} CMD=0x{rcmd:02X} PARAMS={pretty(params)}")
            # For ReadPosition, PARAMS is usually POS_L POS_H (ticks)
            if rcmd == READ_POS_CMD and len(params) >= 2:
                pos_ticks = params[0] | (params[1] << 8)
                print(f"Position (ticks): {pos_ticks}")
        else:
            print("Could not locate bus frame header in report. (Still useful: we proved HID I/O works.)")

    dev.close()
    print("Done.")

if __name__ == "__main__":
    main()
