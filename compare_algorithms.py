import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from agent.agent import DQNAgent

env = gym.make("CartPole-v1")
env = TimeLimit(env, max_episode_steps=500)

def run_random(n=50):
    scores = []
    for _ in range(n):
        obs, _ = env.reset()
        total = 0
        done = truncated = False
        while not done and not truncated:
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            total += 1
        scores.append(total)
    return scores

def run_ppo(n=50):
    model = PPO.load("models/cartpole_ppo")
    scores = []
    for _ in range(n):
        obs, _ = env.reset()
        total = 0
        done = truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total += 1
        scores.append(total)
    return scores

def run_dqn(n=50):
    agent = DQNAgent(4, 2)
    agent.policy_net.load_state_dict(torch.load("models/dqn_best.pth"))
    agent.epsilon = 0.0
    scores = []
    for _ in range(n):
        obs, _ = env.reset()
        total = 0
        done = truncated = False
        while not done and not truncated:
            action = agent.act(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total += 1
        scores.append(total)
    return scores

print("Running Random agent...")
random_scores = run_random()
print(f"Random avg: {np.mean(random_scores):.0f}")

print("Running PPO agent...")
ppo_scores = run_ppo()
print(f"PPO avg: {np.mean(ppo_scores):.0f}")

print("Running DQN agent...")
dqn_scores = run_dqn()
print(f"DQN avg: {np.mean(dqn_scores):.0f}")

env.close()

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left — episode by episode
ax1.plot(random_scores, color='red', alpha=0.7, label=f'Random (avg: {np.mean(random_scores):.0f})')
ax1.plot(dqn_scores, color='orange', alpha=0.7, label=f'DQN scratch (avg: {np.mean(dqn_scores):.0f})')
ax1.plot(ppo_scores, color='steelblue', alpha=0.7, label=f'PPO (avg: {np.mean(ppo_scores):.0f})')
ax1.axhline(y=200, color='black', linestyle='--', alpha=0.5, label='Solved (200)')
ax1.set_title("Episode Scores — All Agents")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Score")
ax1.legend()

# Right — bar chart comparison
agents = ['Random', 'DQN\n(scratch)', 'PPO\n(SB3)']
avgs = [np.mean(random_scores), np.mean(dqn_scores), np.mean(ppo_scores)]
colors = ['red', 'orange', 'steelblue']
bars = ax2.bar(agents, avgs, color=colors, alpha=0.8, edgecolor='white')
ax2.axhline(y=200, color='black', linestyle='--', alpha=0.5, label='Solved (200)')
for bar, avg in zip(bars, avgs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{avg:.0f}', ha='center', va='bottom', fontweight='bold')
ax2.set_title("Average Score Comparison")
ax2.set_ylabel("Average Score")
ax2.legend()

plt.suptitle("RL Algorithm Comparison — CartPole", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("algorithm_comparison.png")
plt.show()
print("Saved as algorithm_comparison.png!")