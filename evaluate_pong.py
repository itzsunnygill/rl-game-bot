import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import matplotlib.pyplot as plt

env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42)
env = VecFrameStack(env, n_stack=4)

model = PPO.load("models/pong_ppo", env=env)

print("Evaluating Pong bot...")
scores = []
for ep in range(10):
    obs = env.reset()
    total = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total += reward[0]
        if done:
            break
    scores.append(total)
    print(f"Episode {ep+1}: Score = {total}")

env.close()

plt.figure(figsize=(10, 5))
plt.plot(scores, color='steelblue', linewidth=2, marker='o')
plt.axhline(y=-21, color='red', linestyle='--', label='Worst possible (-21)')
plt.axhline(y=0, color='green', linestyle='--', label='Break even (0)')
plt.title("Pong PPO Bot — Evaluation Scores")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("pong_results.png")
plt.show()
print("Saved as pong_results.png!")