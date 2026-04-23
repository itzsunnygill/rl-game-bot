import gymnasium as gym
env = gym.make("CartPole-v1")
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    if done: obs, _ = env.reset()
print("Setup works!")
