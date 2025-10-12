import math
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiabetesExerciseEnv(gym.Env):
    """Gymnasium environment modelling daily exercise choices for a person living with diabetes.

    The agent chooses one of four activity intensities every simulation hour. Glucose, heart rate,
    fatigue and adherence are updated by combining simplified physiological dynamics with noisy
    responses. Episodes correspond to one simulated day unless a safety threshold is crossed.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        *,
        day_length: int = 24,
        step_target: int = 9000,
        max_daily_steps: int = 18000,
        glucose_target: float = 110.0,
        glucose_bounds: Tuple[float, float] = (60.0, 350.0),
        heart_rate_bounds: Tuple[float, float] = (50.0, 190.0),
    ) -> None:
        super().__init__()

        if day_length <= 0:
            raise ValueError("day_length must be > 0")
        if step_target <= 0:
            raise ValueError("step_target must be > 0")
        if max_daily_steps < step_target:
            raise ValueError("max_daily_steps must be >= step_target")

        self.day_length = day_length
        self.time_step_hours = 1.0
        self.step_target = float(step_target)
        self.max_daily_steps = float(max_daily_steps)
        self.glucose_target = float(glucose_target)
        self.glucose_bounds = glucose_bounds
        self.heart_rate_bounds = heart_rate_bounds
        self.resting_hr = 72.0
        self.meal_schedule: Dict[int, float] = {8: 42.0, 13: 46.0, 19: 38.0}

        self._action_catalog: Dict[int, Dict[str, float]] = {
            0: {
                "name": "rest",
                "step_gain": 0.0,
                "glucose_effect": 2.5,
                "hr_effect": -8.0,
                "fatigue_delta": -0.18,
                "adherence_delta": 0.05,
            },
            1: {
                "name": "light_walk",
                "step_gain": 1600.0,
                "glucose_effect": -6.5,
                "hr_effect": 12.0,
                "fatigue_delta": 0.04,
                "adherence_delta": 0.02,
            },
            2: {
                "name": "moderate_jog",
                "step_gain": 2900.0,
                "glucose_effect": -12.0,
                "hr_effect": 24.0,
                "fatigue_delta": 0.09,
                "adherence_delta": -0.03,
            },
            3: {
                "name": "high_intensity",
                "step_gain": 3600.0,
                "glucose_effect": -18.0,
                "hr_effect": 36.0,
                "fatigue_delta": 0.16,
                "adherence_delta": -0.07,
            },
        }

        self.action_space = spaces.Discrete(len(self._action_catalog))

        obs_low = np.array(
            [
                self.glucose_bounds[0],
                self.heart_rate_bounds[0],
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                self.glucose_bounds[1],
                self.heart_rate_bounds[1],
                1.0,
                1.0,
                float(self.day_length - 1),
                self.max_daily_steps,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.state: np.ndarray | None = None
        self._step_count = 0
        self._last_action = 0

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._last_action = 0

        glucose = float(
            np.clip(self.np_random.normal(loc=self.glucose_target + 15.0, scale=18.0), *self.glucose_bounds)
        )
        heart_rate = float(
            np.clip(self.np_random.normal(loc=self.resting_hr + 3.0, scale=5.0), *self.heart_rate_bounds)
        )
        fatigue = float(np.clip(self.np_random.uniform(0.15, 0.35), 0.0, 1.0))
        adherence = float(np.clip(self.np_random.uniform(0.65, 0.9), 0.0, 1.0))
        time_of_day = float(self.np_random.integers(low=6, high=9))
        steps = 0.0

        self.state = np.array(
            [glucose, heart_rate, fatigue, adherence, time_of_day, steps],
            dtype=np.float32,
        )

        info = {
            "glucose_target": self.glucose_target,
            "step_target": self.step_target,
        }
        return self._get_obs(), info

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not valid for the current action space.")

        glucose, heart_rate, fatigue, adherence, time_of_day, steps = map(float, self.state)
        profile = self._action_catalog[int(action)]

        self._step_count += 1
        self._last_action = int(action)

        steps = min(steps + profile["step_gain"], self.max_daily_steps)

        fatigue = self._update_fatigue(fatigue, profile)
        adherence = self._update_adherence(adherence, fatigue, profile)

        glucose = self._update_glucose(glucose, profile, time_of_day)
        heart_rate = self._update_heart_rate(heart_rate, glucose, fatigue, profile)

        time_of_day = (time_of_day + self.time_step_hours) % self.day_length

        self.state = np.array(
            [glucose, heart_rate, fatigue, adherence, time_of_day, steps],
            dtype=np.float32,
        )

        reward = self._compute_reward(glucose, heart_rate, fatigue, adherence, steps, time_of_day, profile)
        terminated, termination_reason = self._check_termination(glucose, heart_rate, adherence)
        truncated = self._step_count >= self.day_length and not terminated
        if truncated:
            termination_reason = termination_reason or "end_of_day"

        info = {
            "step": self._step_count,
            "action_name": profile["name"],
            "metrics": {
                "glucose": glucose,
                "heart_rate": heart_rate,
                "fatigue": fatigue,
                "adherence": adherence,
                "steps": steps,
                "time_of_day": time_of_day,
            },
            "termination_reason": termination_reason,
        }

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.state is None:
            print("Environment not reset.")
            return

        glucose, heart_rate, fatigue, adherence, time_of_day, steps = self.state
        action_name = self._action_catalog[self._last_action]["name"]
        print(
            f"[t={self._step_count:02d}] {action_name:<15} | "
            f"BG={glucose:6.1f} mg/dL | HR={heart_rate:6.1f} bpm | "
            f"fatigue={fatigue:4.2f} | adherence={adherence:4.2f} | "
            f"steps={steps:6.0f} | hour={time_of_day:4.1f}"
        )

    def close(self):
        self.state = None

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("State has not been initialised. Call reset first.")
        return self.state.copy()

    def _update_fatigue(self, fatigue: float, profile: Dict[str, float]) -> float:
        fatigue += profile["fatigue_delta"]
        if profile["step_gain"] <= 0.0:
            fatigue -= 0.05 * max(0.0, fatigue - 0.25)
        fatigue += 0.015 * max(0.0, profile["step_gain"] / self.step_target)
        return float(np.clip(fatigue, 0.0, 1.0))

    def _update_adherence(self, adherence: float, fatigue: float, profile: Dict[str, float]) -> float:
        adherence += profile["adherence_delta"]
        adherence -= 0.12 * max(0.0, fatigue - 0.78)
        adherence += 0.03 * math.tanh(0.6 - fatigue)
        return float(np.clip(adherence, 0.0, 1.0))

    def _update_glucose(self, glucose: float, profile: Dict[str, float], time_of_day: float) -> float:
        drift = 0.06 * (self.glucose_target - glucose)
        noise = self.np_random.normal(0.0, 3.0)

        next_hour = int((time_of_day + self.time_step_hours) % self.day_length)
        meal_delta = 0.0
        if next_hour in self.meal_schedule:
            meal_delta = self.np_random.normal(self.meal_schedule[next_hour], 6.0)

        glucose += drift + profile["glucose_effect"] + meal_delta + noise
        return float(np.clip(glucose, *self.glucose_bounds))

    def _update_heart_rate(
        self,
        heart_rate: float,
        glucose: float,
        fatigue: float,
        profile: Dict[str, float],
    ) -> float:
        baseline_pull = 0.25 * (self.resting_hr + 10.0 * fatigue - heart_rate)
        glucose_load = 0.04 * (glucose - self.glucose_target)
        variability = self.np_random.normal(0.0, 2.5)
        heart_rate += baseline_pull + profile["hr_effect"] + glucose_load + variability
        return float(np.clip(heart_rate, *self.heart_rate_bounds))

    def _compute_reward(
        self,
        glucose: float,
        heart_rate: float,
        fatigue: float,
        adherence: float,
        steps: float,
        time_of_day: float,
        profile: Dict[str, float],
    ) -> float:
        glucose_component = np.clip(1.0 - abs(glucose - self.glucose_target) / 40.0, -1.0, 1.0)
        activity_ratio = min(steps / self.step_target, 1.5)
        activity_component = 0.5 * np.tanh(activity_ratio - 0.8)

        adherence_component = 0.6 * (adherence - 0.5)
        fatigue_penalty = -0.5 * np.tanh(max(0.0, fatigue - 0.35) * 2.0)
        heart_rate_penalty = -0.4 * max(0.0, (heart_rate - 165.0) / 35.0)

        late_day_penalty = 0.0
        if time_of_day > (self.day_length * 0.6) and steps < 0.45 * self.step_target:
            late_day_penalty = -0.2
        if profile["step_gain"] == 0.0 and steps < 0.2 * self.step_target:
            late_day_penalty += -0.05

        reward = (
            glucose_component
            + activity_component
            + adherence_component
            + fatigue_penalty
            + heart_rate_penalty
            + late_day_penalty
        )

        if glucose < 80.0:
            reward -= 0.3
        if glucose > 200.0:
            reward -= 0.2

        return float(reward)

    def _check_termination(self, glucose: float, heart_rate: float, adherence: float) -> Tuple[bool, str | None]:
        if glucose <= self.glucose_bounds[0] + 1e-6:
            return True, "severe_hypoglycemia"
        if glucose >= self.glucose_bounds[1] - 1e-6:
            return True, "severe_hyperglycemia"
        if heart_rate >= self.heart_rate_bounds[1] - 1e-6:
            return True, "dangerous_heart_rate"
        if adherence <= 0.05:
            return True, "adherence_failure"
        return False, None


# Backwards-compatible alias
DiabetesEnv = DiabetesExerciseEnv

__all__ = ["DiabetesExerciseEnv", "DiabetesEnv"]
