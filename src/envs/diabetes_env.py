import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DiabetesEnv(gym.Env):
    """More realistic diabetes RL environment"""

    def __init__(self):
        super(DiabetesEnv, self).__init__()

        # Action space: rest, walk, run, eat, insulin
        self.action_space = spaces.Discrete(5)

        # State: [glucose, steps, insulin_left, hours_since_meal]
        self.observation_space = spaces.Box(
            low=np.array([40, 0, 0, 0], dtype=np.float32),
            high=np.array([400, 20000, 10, 24], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([150.0, 0.0, 5.0, 0.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        glucose, steps, insulin_left, hours = self.state

        # Natural drift
        glucose += 1.0
        hours += 1.0

        if action == 0:  # rest
            glucose += 2.0
        elif action == 1:  # walk
            glucose -= 3.0
            steps += 2000
        elif action == 2:  # run
            glucose -= 7.0
            steps += 4000
        elif action == 3:  # eat meal
            glucose += 40.0
            hours = 0
        elif action == 4:  # insulin
            if insulin_left > 0:
                glucose -= 20.0
                insulin_left -= 1

        # Reward
        reward = -abs(glucose - 100) / 50.0
        if glucose < 70:
            reward -= 2.0
        elif glucose > 180:
            reward -= 1.0

        self.state = np.array([glucose, steps, insulin_left, hours], dtype=np.float32)

        terminated = steps >= 20000 or hours >= 24
        truncated = False

        return self.state, reward, terminated, truncated, {}
