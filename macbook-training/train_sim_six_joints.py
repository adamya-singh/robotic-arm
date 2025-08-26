# train_sim_six_joints.py
# pip install gymnasium stable-baselines3 torch numpy
import os, math, time
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

"""
SixJointReachEnvSim: fast, no-sleep simulator with simple first-order actuator dynamics
for six independent joints. Matches your real env's interface and extends it to 6 DoF:
  - obs: interleaved per-joint [angle_i, goal_i_minus_angle_i] × 6 (12-dim)
  - act: 6D deltas in [-1, 1]^6 scaled by max_delta_deg per joint
  - reward: negative L1 distance to goals across all 6 joints
Terminate when all joints are within tolerance OR step budget is reached.
"""

class SixJointReachEnvSim(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        hz=20,
        max_delta_deg=5.0,
        goal_deg=30.0,
        tolerance_deg=2.5,
        jmin=-100.0,
        jmax=100.0,
        settle_ratio=0.35,     # fraction of remaining error applied per step (0..1)
        pos_noise_std=0.15,    # sensor/pose noise (deg)
        max_steps=200,
        randomize_domain=True,
        random_goal_range=(-40.0, 40.0),
        seed=None,
    ):
        super().__init__()
        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        # Per-joint parameters (6 DoF)
        self.max_delta = np.full(6, float(max_delta_deg), dtype=float)
        self.tolerance = float(tolerance_deg)
        self.jmin = float(jmin)
        self.jmax = float(jmax)
        self.max_steps = int(max_steps)
        self.randomize_domain = bool(randomize_domain)
        self.random_goal_range = tuple(random_goal_range)

        # dynamics params (can be randomized each reset)
        self.default_settle_ratio = np.full(6, float(settle_ratio), dtype=float)
        self.default_noise = np.full(6, float(pos_noise_std), dtype=float)

        # spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # Observation per joint: [angle, goal_minus_angle] × 6
        obs_low = np.array([self.jmin, -360.0] * 6, dtype=np.float32)
        obs_high = np.array([self.jmax, 360.0] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # state
        self.angle = np.zeros(6, dtype=float)
        self.target = np.zeros(6, dtype=float)
        self.goal = np.full(6, float(goal_deg), dtype=float)
        self.steps = 0
        self.settle_ratio = self.default_settle_ratio.copy()
        self.pos_noise_std = self.default_noise.copy()

        self._seed = seed

    def _maybe_domain_randomize(self):
        if not self.randomize_domain:
            self.goal = np.clip(self.goal, self.jmin + 5.0, self.jmax - 5.0)
            self.settle_ratio = self.default_settle_ratio.copy()
            self.pos_noise_std = self.default_noise.copy()
            return
        # Randomize goal and simple dynamics for sim2real robustness
        self.goal = np.random.uniform(
            self.random_goal_range[0], self.random_goal_range[1], size=6
        ).astype(float)
        self.goal = np.clip(self.goal, self.jmin + 5.0, self.jmax - 5.0)
        self.settle_ratio = np.random.uniform(0.20, 0.45, size=6).astype(float)  # how “snappy”
        self.pos_noise_std = np.random.uniform(0.05, 0.35, size=6).astype(float)  # sensor noise

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)

    def _read_deg(self):
        # Sensor read per joint = true angle + noise
        return (self.angle + np.random.normal(0.0, self.pos_noise_std)).astype(float)

    def _cmd_deg(self, deg):
        # Hardware would clamp; do the same here (vectorized)
        safety_min = self.jmin + self.tolerance
        safety_max = self.jmax - self.tolerance
        self.target = np.clip(deg, safety_min, safety_max).astype(float)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self._maybe_domain_randomize()

        # Start near zero with a bit of noise for all joints
        self.angle = np.random.uniform(-2.0, 2.0, size=6).astype(float)
        self.target = self.angle.copy()
        self.steps = 0

        angle_read = self._read_deg()
        delta_goal = self.goal - angle_read
        # Interleave [angle_i, goal_minus_angle_i] for all 6 joints
        obs = np.empty(12, dtype=np.float32)
        obs[0::2] = angle_read.astype(np.float32)
        obs[1::2] = delta_goal.astype(np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1

        # Action per joint: delta degrees from [-1,1] * max_delta
        action = np.asarray(action, dtype=float).reshape(-1)
        delta = np.clip(action, -1.0, 1.0) * self.max_delta

        # Command next targets (like your real env)
        curr = self._read_deg()
        self._cmd_deg(curr + delta)

        # First-order actuator “settling” toward target (vectorized)
        # angle <- angle + settle_ratio * (target - angle)
        self.angle += self.settle_ratio * (self.target - self.angle)

        # New observation
        new_angle = self._read_deg()
        delta_goal = self.goal - new_angle
        obs = np.empty(12, dtype=np.float32)
        obs[0::2] = new_angle.astype(np.float32)
        obs[1::2] = delta_goal.astype(np.float32)

        # Reward and termination
        dist = np.abs(new_angle - self.goal)
        reward = -float(dist.sum())
        terminated = bool(np.all(dist < self.tolerance))
        truncated = bool(self.steps >= self.max_steps)

        info = {"dist_deg": dist.tolist(), "goal_deg": self.goal.tolist()}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


def main():
    # Use Apple GPU if present
    #device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"Using device: {device}")

    # Factory so each parallel env has slightly different dynamics/goals
    def make_env():
        return SixJointReachEnvSim(
            hz=20,
            max_delta_deg=5.0,
            tolerance_deg=2.5,
            goal_deg=30.0,             # base goal; will be randomized if randomize_domain=True
            jmin=-100.0, jmax=100.0,
            settle_ratio=0.35,
            pos_noise_std=0.15,
            max_steps=200,
            randomize_domain=True,     # domain randomization ON for sim2real
            random_goal_range=(-40, 40),
        )

    # Vectorized envs speed PPO up massively
    n_envs = 8
    env = make_vec_env(make_env, n_envs=n_envs)

    # Check if saved model exists and load it, otherwise create new
    model_path = "ppo_xarm_sim_6j"
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing model from {model_path}.zip")
        model = PPO.load(model_path, env=env, device=device, verbose=1)
    else:
        print("Creating new model")
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=1,
            # A solid starting config; tweak as desired
            n_steps=1024,        # per env → 1024 * n_envs transitions per rollout
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[128, 128]),
        )

    total_timesteps = 2_000_000
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print("Saved: ppo_xarm_sim_2j.zip")

    # Quick deterministic roll to sanity-check
    test_env = make_env()
    obs, _ = test_env.reset()
    start_angles = ", ".join(f"{a:.2f}" for a in obs[0::2])
    goal_angles = ", ".join(f"{g:.2f}" for g in test_env.goal)
    print(f"start=[{start_angles}]°, goal=[{goal_angles}]°")
    for t in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = test_env.step(action)
        if (t % 10) == 0:
            angles_str = ",".join(f"{a:6.2f}" for a in obs[0::2])
            dists_str = ",".join(f"{d:5.2f}" for d in info['dist_deg'])
            print(f"{t:03d} angle=[{angles_str}]°  dist=[{dists_str}]°  r={r:7.3f}")
        if term or trunc:
            break
    test_env.close()

if __name__ == "__main__":
    main()