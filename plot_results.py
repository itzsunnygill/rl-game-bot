import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = TimeLimit(env, max_episode_steps=500)
model = PPO.load("models/cartpole_ppo")

scores = []
for ep in range(20):
    obs, _ = env.reset()
    total = 0
    done = False
    truncated = False
    while not done and not truncated:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total += reward
    scores.append(total)
    print(f"Episode {ep+1}: {total}")

env.close()

plt.figure(figsize=(10, 5))
plt.plot(scores, color='steelblue', linewidth=2, marker='o')
plt.axhline(y=200, color='red', linestyle='--', label='Untrained baseline (200)')
plt.title("Trained PPO Bot — CartPole Scores")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("results.png")
plt.show()
print("Graph saved as results.png!")