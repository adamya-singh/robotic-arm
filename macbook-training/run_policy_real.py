import argparse
import time
import numpy as np
import requests
from stable_baselines3 import PPO


def _headers(token: str | None) -> dict:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def rpc_invoke(base_url: str, method: str, args=None, kwargs=None, token: str | None = None, timeout=5.0):
    payload = {
        "method": method,
        "args": args or [],
        "kwargs": kwargs or {},
    }
    r = requests.post(f"{base_url}/rpc/invoke", json=payload, headers=_headers(token), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok", False):
        raise RuntimeError(f"RPC invoke failed: {data}")
    return data.get("result")


def rpc_step(base_url: str, servo_id: int, delta_deg: float, duration_ms: int, readback: bool, token: str | None = None, timeout=10.0):
    payload = {
        "servo_id": int(servo_id),
        "delta_deg": float(delta_deg),
        "duration_ms": int(duration_ms),
        "readback": bool(readback),
    }
    r = requests.post(f"{base_url}/rpc/step", json=payload, headers=_headers(token), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # data: {angle_deg, clipped, t_pi}
    return float(data["angle_deg"]), bool(data["clipped"])


def rpc_multi_step(base_url: str, servo_deltas, duration_ms: int, readback: bool, token: str | None = None, timeout=10.0):
    payload = {
        "servo_deltas": [[int(sid), float(delta)] for sid, delta in servo_deltas],
        "duration_ms": int(duration_ms),
        "readback": bool(readback),
    }
    r = requests.post(f"{base_url}/rpc/multi_step", json=payload, headers=_headers(token), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # data: {angles_deg: {"sid": angle}, clipped: {"sid": bool}, t_pi: ...}
    angles = {int(k): float(v) for k, v in data["angles_deg"].items()}
    clipped = {int(k): bool(v) for k, v in data["clipped"].items()}
    return angles, clipped


def main():
    ap = argparse.ArgumentParser(description="Run PPO policy on real arm via RPC bridge.")
    ap.add_argument("--policy", choices=["one", "two"], default=None, help="Which policy to run: one or two servo")
    ap.add_argument("--model", default=None, help="Optional override path to PPO .zip")
    ap.add_argument("--host", required=True, help="Raspberry Pi host/IP")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--servo-id", type=int, default=3)
    ap.add_argument("--goal", type=float, default=30.0, help="Target angle (deg)")
    ap.add_argument("--servo-id-2", type=int, default=4, help="Second servo ID (two-servo policy)")
    ap.add_argument("--goal-2", type=float, default=30.0, help="Target angle for second servo (deg)")
    ap.add_argument("--max-delta-2", type=float, default=None, help="Scale for action→delta_deg for second servo (defaults to --max-delta)")
    ap.add_argument("--hz", type=float, default=20.0, help="Control loop frequency")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--tolerance", type=float, default=2.5, help="Terminate when within tolerance (deg)")
    ap.add_argument("--max-delta", type=float, default=5.0, help="Scale for action→delta_deg (matches training)")
    ap.add_argument("--token", default=None, help="Bearer token if PI enforces auth")
    ap.add_argument("--servo-on", action="store_true", help="Send servoOn before running")
    ap.add_argument("--servo-off", action="store_true", help="Send servoOff after completion")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions")
    ap.add_argument("--print-every", type=int, default=10, help="How often to log iterations")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Determine policy selection
    policy_choice = args.policy
    if policy_choice is None:
        try:
            sel = input("Run which policy? [1] one-servo, [2] two-servo (default 1): ").strip()
        except EOFError:
            sel = "1"
        policy_choice = "two" if sel == "2" else "one"

    # Choose model path (allow override via --model)
    if args.model:
        model_path = args.model
    else:
        if policy_choice == "two":
            model_path = "macbook-training/ppo_xarm_sim_2j.zip"
        else:
            model_path = "macbook-training/ppo_xarm_sim.zip"

    # Load PPO policy
    model = PPO.load(model_path, device="cpu")

    # Optionally turn servos on
    if args.servo_on:
        try:
            rpc_invoke(base_url, "servoOn", token=args.token)
        except Exception as e:
            print(f"Warning: servoOn failed: {e}")

    dt = 1.0 / max(args.hz, 0.1)
    duration_ms = max(1, int(round(dt * 1000.0)))

    if policy_choice == "two":
        # Two-servo control using /rpc/multi_step
        sid1 = int(args.servo_id)
        sid2 = int(args.servo_id_2)
        goal1 = float(args.goal)
        goal2 = float(args.goal_2)
        max_delta_1 = float(args.max_delta)
        max_delta_2 = float(args.max_delta_2) if args.max_delta_2 is not None else max_delta_1

        angle1 = rpc_invoke(base_url, "getPosition", args=[sid1], kwargs={"degrees": True}, token=args.token)
        angle2 = rpc_invoke(base_url, "getPosition", args=[sid2], kwargs={"degrees": True}, token=args.token)
        angle1 = float(angle1)
        angle2 = float(angle2)

        obs = np.array([angle1, goal1 - angle1, angle2, goal2 - angle2], dtype=np.float32)
        print(f"Start s1={angle1:.2f}°→{goal1:.2f}°, s2={angle2:.2f}°→{goal2:.2f}°, hz={args.hz}, duration_ms={duration_ms}")

        for t in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            a = np.array(action).reshape(-1)
            if a.size < 2:
                raise RuntimeError(f"Two-servo policy expects action dim=2, got shape {a.shape}")
            a = np.clip(a[:2], -1.0, 1.0)
            delta1 = float(a[0]) * max_delta_1
            delta2 = float(a[1]) * max_delta_2

            angles, clipped = rpc_multi_step(
                base_url,
                servo_deltas=[[sid1, delta1], [sid2, delta2]],
                duration_ms=duration_ms,
                readback=True,
                token=args.token,
            )

            angle1 = float(angles.get(sid1, angle1))
            angle2 = float(angles.get(sid2, angle2))

            dist1 = abs(angle1 - goal1)
            dist2 = abs(angle2 - goal2)
            if (t % args.print_every) == 0:
                print(
                    f"{t:03d} s1={angle1:7.2f}° d1={dist1:5.2f}° s2={angle2:7.2f}° d2={dist2:5.2f}° a=({a[0]:+.3f},{a[1]:+.3f}) d=({delta1:+.2f},{delta2:+.2f}) clip=({clipped.get(sid1)}, {clipped.get(sid2)})"
                )

            if dist1 < args.tolerance and dist2 < args.tolerance:
                print(f"Reached both goals within tolerance at t={t}")
                break

            obs = np.array([angle1, goal1 - angle1, angle2, goal2 - angle2], dtype=np.float32)
    else:
        # Single-servo control using /rpc/step (existing behavior)
        angle = rpc_invoke(base_url, "getPosition", args=[args.servo_id], kwargs={"degrees": True}, token=args.token)
        angle = float(angle)
        goal = float(args.goal)

        obs = np.array([angle, goal - angle], dtype=np.float32)
        print(f"Start angle={angle:.2f}°, goal={goal:.2f}°, hz={args.hz}, duration_ms={duration_ms}")

        for t in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            # Ensure scalar
            a = float(np.clip(np.array(action).reshape(-1)[0], -1.0, 1.0))
            delta_deg = a * args.max_delta

            # One control-period step executed on the Pi (server sleeps for duration_ms)
            angle, clipped = rpc_step(
                base_url,
                servo_id=args.servo_id,
                delta_deg=delta_deg,
                duration_ms=duration_ms,
                readback=True,
                token=args.token,
            )

            dist = abs(angle - goal)
            if (t % args.print_every) == 0:
                print(f"{t:03d} angle={angle:7.2f}°  dist={dist:5.2f}°  a={a:+.3f}  d={delta_deg:+.2f}°  clip={clipped}")

            # Termination like training
            if dist < args.tolerance:
                print(f"Reached goal within tolerance ({dist:.2f} < {args.tolerance} deg) at t={t}")
                break

            # Next observation
            obs = np.array([angle, goal - angle], dtype=np.float32)

    # Optionally power off
    if args.servo_off:
        try:
            rpc_invoke(base_url, "servoOff", token=args.token)
        except Exception as e:
            print(f"Warning: servoOff failed: {e}")


if __name__ == "__main__":
    main()