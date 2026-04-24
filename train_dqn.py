import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from gymnasium.wrappers import TimeLimit
from agent.agent import DQNAgent

os.makedirs("models", exist_ok=True)

# Cap episodes at 500 steps max
env = gym.make("CartPole-v1")
env = TimeLimit(env, max_episode_steps=500)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

scores = []
best_avg = 0
episodes = 500

print("Training DQN from scratch...")
for ep in range(episodes):
    state, _ = env.reset()
    total = 0
    done = False
    truncated = False

    while not done and not truncated:
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)

        # Reward shaping — punish for failing
        if done and total < 499:
            reward = -10

        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        total += 1

    if ep % 10 == 0:
        agent.update_target()

    scores.append(total)
    avg = np.mean(scores[-50:])

    if avg > best_avg:
        best_avg = avg
        torch.save(agent.policy_net.state_dict(), "models/dqn_best.pth")

    if ep % 20 == 0:
        print(f"Episode {ep} | Score: {total} | Avg(50): {avg:.0f} | Best: {best_avg:.0f} | Epsilon: {agent.epsilon:.3f}")

torch.save(agent.policy_net.state_dict(), "models/dqn_cartpole.pth")
print(f"Training done! Best avg: {best_avg:.0f}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(scores, alpha=0.3, color='steelblue', label='Score')
plt.plot(np.convolve(scores, np.ones(50)/50, mode='valid'),
         color='steelblue', linewidth=2, label='Moving avg (50)')
plt.axhline(y=475, color='green', linestyle='--', label='Excellent (475)')
plt.axhline(y=200, color='red', linestyle='--', label='Solved (200)')
plt.title("DQN from Scratch — CartPole Training")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("dqn_results.png")
plt.show()
print("Graph saved!")