import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="rgb_array")
model = PPO.load("models/cartpole_ppo")

print("Watching trained bot play...")
episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    print(f"Episode {ep+1}: Score = {total_reward}")

env.close()
print("Done watching!")