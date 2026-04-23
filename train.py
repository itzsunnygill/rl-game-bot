import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = gym.make("CartPole-v1")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/"
)

print("Training started! Watch the reward go up...")
model.learn(total_timesteps=50000)

model.save("models/cartpole_ppo")
env.close()
print("Done! Model saved to models/cartpole_ppo.zip")