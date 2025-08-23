#!/usr/bin/env python3
"""
xArm RPC Bridge (single-file FastAPI)

Endpoints
- GET    /health
- POST   /rpc/invoke          (allow-listed xarm methods)
- POST   /rpc/step            (one control-period round trip)
- GET    /telemetry/snapshot
- WS     /telemetry/stream
- POST   /safety/limits
- GET    /safety/limits
- POST   /safety/estop

Security
- Optional bearer token: set env PI_RPC_TOKEN="yourtoken"
- Optional IP allowlist (comma-separated): PI_RPC_IP_ALLOWLIST="192.168.1.10,192.168.1.11"

Run
- pip install fastapi uvicorn pydantic "xarm"  # xarm: your arm’s Python lib
- python bridge.py --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator
import uvicorn

# --- Hardware lib ---
try:
    import xarm  # replace with your actual device lib
except Exception as e:  # graceful error if library is missing
    raise SystemExit(f"Failed to import xarm library: {e}")

# --------- Config / Globals ----------

API_VERSION = 1

# Hard limits (mechanical) for safety envelope
SERVO_LIMITS_DEG = {
    1: (-50.0, 50.0),
    2: (-120.0, 120.0),
    3: (-100.0, 100.0),
    4: (-120.0, 70.0),
    5: (-90.0, 90.0),
    6: (-120.0, 120.0),
}

# Default soft limits & rate caps
DEFAULT_MAX_RATE_DEG_S = 60.0
DEFAULT_TOLERANCE_DEG = 2.5

LIMITS_PATH = Path(__file__).with_name("limits.json")

# Security
PI_RPC_TOKEN = os.getenv("PI_RPC_TOKEN", "").strip()
IP_ALLOW = [ip.strip() for ip in os.getenv("PI_RPC_IP_ALLOWLIST", "").split(",") if ip.strip()]

# Concurrency (serialize hardware access)
xarm_lock = asyncio.Lock()

# Initialize controller (single instance)
try:
    ARM = xarm.Controller("USB")
except Exception as e:
    raise SystemExit(f"Could not initialize xarm.Controller('USB'): {e}")

# Track last clipping event (for telemetry)
LAST_CLIP: Optional[Dict[str, Any]] = None


# ---------- Safety / Limits persistence ----------

class ServoLimit(BaseModel):
    min_deg: float
    max_deg: float
    max_rate_deg_s: float = Field(default=DEFAULT_MAX_RATE_DEG_S, gt=0)
    tolerance_deg: float = Field(default=DEFAULT_TOLERANCE_DEG, ge=0)

    @validator("max_deg")
    def _order(cls, v, values):
        if "min_deg" in values and v <= values["min_deg"]:
            raise ValueError("max_deg must be greater than min_deg")
        return v


def _default_limits() -> Dict[str, ServoLimit]:
    out = {}
    for sid, (mn, mx) in SERVO_LIMITS_DEG.items():
        out[str(sid)] = ServoLimit(min_deg=mn, max_deg=mx)
    return out


def load_limits() -> Dict[str, ServoLimit]:
    if LIMITS_PATH.exists():
        with LIMITS_PATH.open("r") as f:
            raw = json.load(f)
        return {k: ServoLimit(**v) for k, v in raw.items()}
    return _default_limits()


def save_limits(lims: Dict[str, ServoLimit]) -> None:
    with LIMITS_PATH.open("w") as f:
        json.dump({k: v.dict() for k, v in lims.items()}, f, indent=2)


LIMITS: Dict[str, ServoLimit] = load_limits()


# ---------- Security dependency ----------

async def auth_guard(request: Request):
    # IP allowlist (optional)
    if IP_ALLOW:
        client_ip = request.client.host if request.client else ""
        if client_ip not in IP_ALLOW:
            raise HTTPException(status_code=403, detail="IP not allowed")

    # Bearer token (optional)
    if PI_RPC_TOKEN:
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = auth.split(" ", 1)[1].strip()
        if token != PI_RPC_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid token")


# ---------- Models ----------

class HealthResp(BaseModel):
    ok: bool
    api_version: int
    t_pi: int  # unix ns


class InvokeReq(BaseModel):
    method: str
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    seq: Optional[Union[int, str]] = None


class InvokeResp(BaseModel):
    ok: bool
    result: Any = None
    t_pi: int
    seq: Optional[Union[int, str]] = None


class StepReq(BaseModel):
    servo_id: int
    delta_deg: float
    duration_ms: int = Field(ge=1)
    readback: bool = True


class StepResp(BaseModel):
    angle_deg: float
    clipped: bool
    t_pi: int


class LimitsUpdateReq(BaseModel):
    limits: Dict[str, ServoLimit]


class LimitsResp(BaseModel):
    limits: Dict[str, ServoLimit]


class TelemetrySnapshot(BaseModel):
    positions: Dict[str, Optional[float]]
    battery_v: Optional[float]
    temps_c: Dict[str, Optional[float]]


# ---------- Helpers ----------

def now_ns() -> int:
    return time.time_ns()


async def _read_angle_deg(servo_id: int) -> float:
    async with xarm_lock:
        return await run_in_threadpool(ARM.getPosition, int(servo_id), True)


def _safe_bounds(lim: ServoLimit) -> (float, float):
    # shrink by tolerance; if collapsed, fall back to hard bounds
    low = lim.min_deg + lim.tolerance_deg
    high = lim.max_deg - lim.tolerance_deg
    if low >= high:
        low, high = lim.min_deg, lim.max_deg
    return low, high


async def _enforce_motion_limits(
    servo_id: int, target_deg: float, duration_ms: int
) -> (float, bool):
    """
    Clamp target to soft bounds and max rate.
    Returns (safe_target, clipped_flag).
    """
    global LAST_CLIP
    sid = str(int(servo_id))
    lim = LIMITS.get(sid)
    if lim is None:
        raise HTTPException(status_code=400, detail=f"No limits for servo {servo_id}")

    # Current position
    curr = await _read_angle_deg(servo_id)

    clipped = False
    # Bounds
    low, high = _safe_bounds(lim)
    bounded = max(min(target_deg, high), low)
    if bounded != target_deg:
        clipped = True
        LAST_CLIP = {"servo_id": servo_id, "reason": "bounds", "t_pi": now_ns()}
    target_deg = bounded

    # Rate limiting
    duration_s = max(duration_ms, 1) / 1000.0
    max_delta = lim.max_rate_deg_s * duration_s
    desired = target_deg - curr
    if abs(desired) > max_delta:
        target_deg = curr + (max_delta if desired > 0 else -max_delta)
        clipped = True
        LAST_CLIP = {"servo_id": servo_id, "reason": "rate", "t_pi": now_ns()}

    return target_deg, clipped


async def _set_position_safe(servo_id: int, target_deg: float, duration_ms: int) -> None:
    safe_target, _ = await _enforce_motion_limits(servo_id, target_deg, duration_ms)
    async with xarm_lock:
        await run_in_threadpool(ARM.setPosition, int(servo_id), float(safe_target), duration_ms, False)


async def _snapshot() -> TelemetrySnapshot:
    positions: Dict[str, Optional[float]] = {}
    temps: Dict[str, Optional[float]] = {}
    for sid in sorted(SERVO_LIMITS_DEG.keys()):
        sid_str = str(sid)
        try:
            async with xarm_lock:
                pos = await run_in_threadpool(ARM.getPosition, sid, True)
        except Exception:
            pos = None
        positions[sid_str] = pos

        # temp may not be supported; best-effort
        try:
            async with xarm_lock:
                temp = await run_in_threadpool(ARM.getTemp, sid)  # type: ignore[attr-defined]
        except Exception:
            temp = None
        temps[sid_str] = temp

    try:
        async with xarm_lock:
            batt = await run_in_threadpool(ARM.getBatteryVoltage)
    except Exception:
        batt = None

    return TelemetrySnapshot(positions=positions, temps_c=temps, battery_v=batt)


# ---------- Allowlist dispatch for /rpc/invoke ----------

# Map method name -> ("type", callable or handler)
# For setPosition we intercept to enforce limits.
ALLOWED_METHODS = {
    "getPosition": "passthrough",
    "setPosition": "intercept",  # safety clamp here
    "servoOn": "passthrough",
    "servoOff": "passthrough",
    "getBatteryVoltage": "passthrough",
    "getVolt": "passthrough",   # if your lib exposes this
    "getTemp": "passthrough",   # per-servo requires arg
}


async def _invoke_allowlisted(req: InvokeReq) -> Any:
    kind = ALLOWED_METHODS.get(req.method)
    if not kind:
        raise HTTPException(status_code=400, detail=f"Method not allowed: {req.method}")

    # Intercept setPosition for safety clamp
    if req.method == "setPosition":
        # Expect: (servo_id, deg, duration=..., wait=False)
        if len(req.args) < 2 and "servo_id" not in req.kwargs:
            raise HTTPException(status_code=400, detail="setPosition requires servo_id and deg")
        # Extract args with defaults
        servo_id = int(req.kwargs.pop("servo_id")) if "servo_id" in req.kwargs else int(req.args[0])
        deg = float(req.kwargs.pop("deg")) if "deg" in req.kwargs else float(req.args[1])
        duration_ms = int(req.kwargs.pop("duration", req.kwargs.pop("duration_ms", 200)))
        # ignore wait, always False (non-blocking move)
        await _set_position_safe(servo_id, deg, duration_ms)
        return "ok"

    # Passthrough (but still serialize access)
    if kind == "passthrough":
        # Small convenience: default degrees=True for getPosition if not set
        if req.method == "getPosition":
            if len(req.args) >= 1 and "degrees" not in req.kwargs:
                req.kwargs["degrees"] = True

        async with xarm_lock:
            func = getattr(ARM, req.method)
            return await run_in_threadpool(func, *req.args, **req.kwargs)

    raise HTTPException(status_code=400, detail="Unhandled dispatch path")


# ---------- FastAPI app ----------

app = FastAPI(title="xArm RPC Bridge", version=str(API_VERSION))


@app.get("/health", response_model=HealthResp)
async def health():
    return HealthResp(ok=True, api_version=API_VERSION, t_pi=now_ns())


@app.post("/rpc/invoke", response_model=InvokeResp, dependencies=[Depends(auth_guard)])
async def rpc_invoke(req: InvokeReq):
    try:
        result = await _invoke_allowlisted(req)
        return InvokeResp(ok=True, result=result, t_pi=now_ns(), seq=req.seq)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invoke error: {e}")


@app.post("/rpc/step", response_model=StepResp, dependencies=[Depends(auth_guard)])
async def rpc_step(req: StepReq):
    """
    One control-period round trip:
      - read current
      - compute target = current + delta
      - clamp (bounds + rate)
      - setPosition(wait=False)
      - sleep duration_ms (server-side)
      - readback (optional)
    """
    try:
        curr = await _read_angle_deg(req.servo_id)
        target = curr + float(req.delta_deg)
        safe_target, clipped = await _enforce_motion_limits(req.servo_id, target, req.duration_ms)

        async with xarm_lock:
            await run_in_threadpool(ARM.setPosition, int(req.servo_id), float(safe_target), int(req.duration_ms), False)

        # Server-owned timing for lower jitter
        await asyncio.sleep(max(req.duration_ms, 1) / 1000.0)

        if req.readback:
            angle = await _read_angle_deg(req.servo_id)
        else:
            angle = safe_target

        return StepResp(angle_deg=float(angle), clipped=bool(clipped), t_pi=now_ns())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step error: {e}")


@app.get("/telemetry/snapshot", response_model=TelemetrySnapshot, dependencies=[Depends(auth_guard)])
async def telemetry_snapshot():
    try:
        return await _snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snapshot error: {e}")


@app.websocket("/telemetry/stream")
async def telemetry_stream(ws: WebSocket):
    # Lightweight auth for WS: require ?token=... if PI_RPC_TOKEN is set; IP allowlist still applies below.
    # Note: WebSocket has no headers dependency path like HTTP; implement inline checks.
    client_ip = ws.client.host if ws.client else ""
    if IP_ALLOW and client_ip not in IP_ALLOW:
        await ws.close(code=4403)  # policy violation
        return
    if PI_RPC_TOKEN:
        token = ws.query_params.get("token", "")
        if token != PI_RPC_TOKEN:
            await ws.close(code=4401)  # unauthorized
            return

    await ws.accept()
    try:
        hz = float(ws.query_params.get("hz", "5"))
        hz = max(0.5, min(hz, 20.0))
        period = 1.0 / hz
        while ws.application_state == WebSocketState.CONNECTED:
            snap = await _snapshot()
            msg = {
                "t_pi": now_ns(),
                "positions": snap.positions,
                "battery_v": snap.battery_v,
                "temps_c": snap.temps_c,
                "last_clip": LAST_CLIP,
            }
            await ws.send_json(msg)
            await asyncio.sleep(period)
    except WebSocketDisconnect:
        return
    except Exception:
        # Best-effort close on error
        try:
            await ws.close()
        except Exception:
            pass


@app.post("/safety/limits", response_model=Dict[str, str], dependencies=[Depends(auth_guard)])
async def safety_limits_set(req: LimitsUpdateReq):
    # Merge+persist
    try:
        for sid, lim in req.limits.items():
            # Ensure servo exists in hard table
            sid_int = int(sid)
            if sid_int not in SERVO_LIMITS_DEG:
                raise HTTPException(status_code=400, detail=f"Unknown servo {sid}")
            # Also ensure soft limits are within hard limits
            hard_min, hard_max = SERVO_LIMITS_DEG[sid_int]
            if lim.min_deg < hard_min or lim.max_deg > hard_max:
                raise HTTPException(
                    status_code=400,
                    detail=f"Soft limits for servo {sid} must be within hard bounds {hard_min}..{hard_max}",
                )
            LIMITS[str(sid_int)] = lim
        save_limits(LIMITS)
        return {"ok": "true"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Limits update error: {e}")


@app.get("/safety/limits", response_model=LimitsResp, dependencies=[Depends(auth_guard)])
async def safety_limits_get():
    return LimitsResp(limits=LIMITS)


@app.post("/safety/estop", response_model=Dict[str, str], dependencies=[Depends(auth_guard)])
async def safety_estop():
    try:
        async with xarm_lock:
            await run_in_threadpool(ARM.servoOff)
        return {"ok": "true"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ESTOP error: {e}")


# ---------- CLI ----------

def _parse_cli():
    import argparse
    p = argparse.ArgumentParser(description="xArm RPC Bridge")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    # Bind to LAN; rely on firewall + token/IP allow for access control
    uvicorn.run("bridge:app", host=args.host, port=args.port, reload=False)