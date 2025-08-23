# file: xarm_env_minimal.py
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import xarm

# Per-servo mechanical limits in degrees (hard bounds)
SERVO_LIMITS_DEG = {
    1: (-50.0, 50.0),    # gripper open/close
    2: (-120.0, 120.0),  # gripper rotation
    3: (-100.0, 100.0),  # forearm (top joint)
    4: (-120.0, 70.0),   # arm (middle joint)
    5: (-90.0, 90.0),    # arm (bottom joint)
    6: (-120.0, 120.0),  # base rotation
}

class OneJointReachEnv(gym.Env):
    """
    Minimal real-robot env:
      - Controls servo 1 only
      - Observation: [current_angle_deg]
      - Action: delta angle in [-1, 1] scaled to +/-2 deg
      - Reward: -abs(current_angle - goal_deg)
    """
    metadata = {"render_modes": []}

    def __init__(self, hz=10, max_delta_deg=2.0, goal_deg=30.0,
                 servo_id=1, tolerance_deg=2.5,
                 joint_min_deg=None, joint_max_deg=None):
        super().__init__()
        self.arm = xarm.Controller('USB')
        self.sid = int(servo_id)
        self.hz = hz
        self.dt = 1.0 / hz
        self.max_delta = float(max_delta_deg)
        self.goal = float(goal_deg)
        self.tolerance = float(tolerance_deg)

        # Resolve joint limits: prefer explicit args, else from table
        if joint_min_deg is not None and joint_max_deg is not None:
            jmin, jmax = float(joint_min_deg), float(joint_max_deg)
        else:
            if self.sid not in SERVO_LIMITS_DEG:
                raise ValueError(f"No limits configured for servo {self.sid}")
            jmin, jmax = SERVO_LIMITS_DEG[self.sid]
        # Ensure min < max
        if jmin >= jmax:
            raise ValueError("joint_min_deg must be less than joint_max_deg")
        self.jmin, self.jmax = float(jmin), float(jmax)

        # action: scalar in [-1, 1] → delta deg
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # obs: current angle (deg). Box keeps it simple.
        self.observation_space = spaces.Box(
            low=np.array([self.jmin], dtype=np.float32),
            high=np.array([self.jmax], dtype=np.float32),
            dtype=np.float32
        )

    def _read_deg(self):
        return float(self.arm.getPosition(self.sid, degrees=True))

    def _cmd_deg(self, deg):
        # Clamp to safe bounds with tolerance margin to avoid overshoot
        safety_min = self.jmin + self.tolerance
        safety_max = self.jmax - self.tolerance
        if safety_min > safety_max:
            # Fallback to hard limits if tolerance collapses the range
            safety_min, safety_max = self.jmin, self.jmax
        deg = float(np.clip(deg, safety_min, safety_max))
        # Duration matches control period
        duration_ms = int(self.dt * 1000)
        self.arm.setPosition(self.sid, deg, duration=duration_ms, wait=False)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Move to 0 deg as "home"
        self._cmd_deg(0.0)
        time.sleep(0.5)
        angle = self._read_deg()
        obs = np.array([angle], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        # Map action [-1,1] → delta degrees
        delta = float(np.clip(action[0], -1.0, 1.0)) * self.max_delta

        # Current angle → command next angle
        curr = self._read_deg()
        target = curr + delta
        self._cmd_deg(target)

        # Wait for the control period to elapse
        time.sleep(self.dt)

        # New observation
        new_angle = self._read_deg()
        obs = np.array([new_angle], dtype=np.float32)

        # Reward: closer to goal is better
        dist = abs(new_angle - self.goal)
        reward = -dist

        # Episode termination: succeed if within tolerance
        terminated = dist < self.tolerance
        truncated = False
        info = {"dist_deg": dist}

        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self.arm.servoOff()
        except:
            pass


if __name__ == "__main__":
    import numpy as np, time
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise SystemExit("stable-baselines3 is required for PPO. Install with: pip install stable-baselines3") from e

    # Environment configuration (slow control + modest step size for real hardware)
    env = OneJointReachEnv(hz=5, max_delta_deg=5.0, goal_deg=30.0, servo_id=3)

    # Train a small PPO policy (keep timesteps low to avoid long real-world training)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200)

    # Evaluate the learned policy deterministically
    obs, _ = env.reset()
    print(f"start angle: {obs[0]:.2f}°  → goal: {env.goal:.2f}°")

    try:
        for t in range(60):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            print(f"{t:02d} angle={obs[0]:6.2f}°  dist={info['dist_deg']:5.2f}°  r={r:7.3f}")
            if terminated or truncated:
                print("Done.")
                break
        time.sleep(0.2)
    finally:
        env.close()
