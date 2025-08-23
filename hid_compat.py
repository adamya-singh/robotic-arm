#!/usr/bin/env python3
"""
hid_compat.py
A backward-compat shim so libraries expecting the legacy `hid.device()` API
(e.g., calling `hid.device(); dev.open(vid, pid); dev.write([...])`)
work with newer `hidapi` builds that expose `hid.Device` and stricter method
signatures.
"""

import hid as _hid


def _to_bytes(data):
    """Coerce various buffer-like inputs to `bytes` for strict hidapi builds."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, (list, tuple)):
        return bytes(int(x) & 0xFF for x in data)
    raise TypeError(f"hid_compat.write expects bytes-like or list/tuple of ints, got {type(data)}")


# Install the shim only if needed (no-op if a proper legacy `hid.device` exists)
if not hasattr(_hid, "device") or _hid.device is _hid.Device:

    class _CompatDevice:
        """
        Emulates the old `hid.device` object:
          d = hid.device()
          d.open(vid, pid[, serial])
          d.open_path(path)
          d.set_nonblocking(flag)
          d.write(bytes_or_list)
          d.read(length[, timeout_ms])
          d.close()
        """
        __slots__ = ("_d",)

        def __init__(self):
            self._d = None

        # Legacy: open(vid, pid, serial_number=None)
        def open(self, vid, pid, serial_number=None):
            """
            Try multiple constructor signatures used across hidapi variants.
            """
            tried = []
            for ctor in (
                # Common keyword form (hidapi >= 0.14.* python wheel)
                lambda: _hid.Device(vendor_id=vid, product_id=pid, serial=serial_number),
                lambda: _hid.Device(vendor_id=vid, product_id=pid),
                # Alternative kw names seen in some builds
                lambda: _hid.Device(vid=vid, pid=pid, serial=serial_number),
                lambda: _hid.Device(vid=vid, pid=pid),
                # Positional (older cython-hidapi)
                lambda: _hid.Device(vid, pid) if serial_number is None else _hid.Device(vid, pid, serial_number),
            ):
                try:
                    self._d = ctor()
                    return
                except TypeError as e:
                    tried.append(str(e))
            raise TypeError(
                "hid_compat: Could not construct hid.Device with any known signature. "
                "Constructor TypeErrors: " + " | ".join(tried)
            )

        # Legacy: open_path(path)
        def open_path(self, path):
            try:
                self._d = _hid.Device(path=path)
            except TypeError:
                # Older builds accept positional path
                self._d = _hid.Device(path)

        # Legacy: set_nonblocking(flag)
        def set_nonblocking(self, flag):
            # Some builds expose a method, others a property
            try:
                self._d.set_nonblocking(int(bool(flag)))
            except AttributeError:
                try:
                    self._d.nonblocking = bool(flag)
                except Exception:
                    # Fallback: ignore; reads will be blocking
                    pass

        # Legacy: write(data)
        def write(self, data):
            data = _to_bytes(data)
            return self._d.write(data)

        # Legacy: read(length, timeout_ms=0)
        def read(self, length, timeout_ms=0):
            # Prefer (len, timeout); fall back to (len)
            try:
                return self._d.read(int(length), int(timeout_ms))
            except TypeError:
                return self._d.read(int(length))

        # Legacy: close()
        def close(self):
            if self._d:
                try:
                    self._d.close()
                finally:
                    self._d = None

        # Nice-to-have context manager support
        def __enter__(self):
            if self._d is None:
                raise RuntimeError("hid_compat: device not opened")
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    # Install alias so `xarm` (and others) can use the legacy entrypoint.
    _hid.device = _CompatDevice
