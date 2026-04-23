import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

def run_episodes(env, model=None, n=20):
    scores = []
    for _ in range(n):
        obs, _ = env.reset()
        total = 0
        done = False
        truncated = False
        while not done and not truncated:
            if model:
                action, _ = model.predict(obs)
            else:
                action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            total += reward
        scores.append(total)
    return scores

env = TimeLimit(gym.make("CartPole-v1", render_mode="rgb_array"), max_episode_steps=500)
model = PPO.load("models/cartpole_ppo")

random_scores = run_episodes(env, model=None)
trained_scores = run_episodes(env, model=model)
env.close()

plt.figure(figsize=(10, 5))
plt.plot(random_scores, color='red', linewidth=2, marker='o', label='Random agent')
plt.plot(trained_scores, color='steelblue', linewidth=2, marker='o', label='Trained PPO bot')
plt.title("Random Agent vs Trained PPO Bot")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("comparison.png")
plt.show()
print("Saved as comparison.png!")