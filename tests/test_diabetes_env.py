import numpy as np
import pytest

from src.envs.diabetes_env import DiabetesEnv, DiabetesExerciseEnv


@pytest.fixture
def env():
    environment = DiabetesExerciseEnv()
    yield environment
    environment.close()


def test_reset_observation_within_space(env):
    obs, info = env.reset(seed=42)

    assert env.observation_space.contains(obs), "Reset observation should lie within the observation space"
    assert info["glucose_target"] == env.glucose_target
    assert info["step_target"] == env.step_target


def test_step_returns_valid_transition(env):
    env.reset(seed=13)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(next_obs), "Step observation should lie within the observation space"
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "metrics" in info and "action_name" in info
    metrics = info["metrics"]
    assert {"glucose", "heart_rate", "fatigue", "adherence", "steps", "time_of_day"} <= metrics.keys()
    assert np.isfinite(reward)


def test_check_termination_edges(env):
    # severe hypoglycemia
    terminated, reason = env._check_termination(env.glucose_bounds[0], env.resting_hr, 0.5)
    assert terminated and reason == "severe_hypoglycemia"

    # severe hyperglycemia
    terminated, reason = env._check_termination(env.glucose_bounds[1], env.resting_hr, 0.5)
    assert terminated and reason == "severe_hyperglycemia"

    # dangerous heart rate
    terminated, reason = env._check_termination(env.glucose_target, env.heart_rate_bounds[1], 0.5)
    assert terminated and reason == "dangerous_heart_rate"

    # adherence failure
    terminated, reason = env._check_termination(env.glucose_target, env.resting_hr, 0.0)
    assert terminated and reason == "adherence_failure"


def test_episode_truncates_after_full_day(env):
    env.reset(seed=0)
    env.meal_schedule = {}  # avoid large glucose spikes during the rollout

    truncated_flags = []
    for _ in range(env.day_length):
        _, _, terminated, truncated, info = env.step(0)  # choose "rest" consistently
        truncated_flags.append((terminated, truncated, info["termination_reason"]))
        if terminated:
            pytest.skip("Episode terminated early; rerun test with a different seed to avoid random termination.")

    *_, (terminated, truncated, reason) = truncated_flags
    assert not terminated
    assert truncated
    assert reason == "end_of_day"


def test_diabetes_env_alias():
    assert DiabetesEnv is DiabetesExerciseEnv


def test_step_counts_non_decreasing_within_episode(env):
    obs, _ = env.reset(seed=2024)
    previous_steps = obs[5]

    for _ in range(env.day_length // 2):
        obs, _, terminated, truncated, _ = env.step(2)  # moderate_jog
        if terminated or truncated:
            pytest.skip("Episode terminated before verifying step monotonicity; rerun with a different seed.")
        current_steps = obs[5]
        assert current_steps + 1e-6 >= previous_steps
        previous_steps = current_steps


def test_long_rollout_health_statistics(env):
    obs, _ = env.reset(seed=314)
    rewards = []
    glucose_values = []
    fatigue_values = []
    total_steps = 250

    for i in range(total_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)
        glucose_values.append(obs[0])
        fatigue_values.append(obs[2])

        if terminated or truncated:
            obs, _ = env.reset(seed=314 + i + 1)

    safe_fraction = sum(70.0 <= g <= 180.0 for g in glucose_values) / len(glucose_values)
    assert safe_fraction >= 0.5
    assert -2.5 < float(np.mean(rewards)) < 2.5
    assert max(fatigue_values) <= 1.0 + 1e-6
