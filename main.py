from stable_baselines3 import PPO
from src.envs.diabetes_env import DiabetesEnv

def main():
    env = DiabetesEnv()

    # Train PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Test agent
    obs, _ = env.reset()
    glucose_history = []
    reward_history = []

    for _ in range(50):   # simulate 50 steps
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        glucose = obs[0]   # first element of state
        glucose_history.append(glucose)
        reward_history.append(reward)

        print(f"Action: {action}, Glucose: {glucose}, Reward: {reward}")

        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
