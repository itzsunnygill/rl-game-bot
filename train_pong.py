# NOTE: Run this on Kaggle with GPU enabled for best results
# Kaggle notebook: https://www.kaggle.com

import ale_py
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)
os.makedirs("models", exist_ok=True)

env = make_atari_env("ALE/Pong-v5", n_envs=4, seed=42)
env = VecFrameStack(env, n_stack=4)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    device="cuda"
)

model.learn(total_timesteps=1_000_000)
model.save("models/pong_ppo")
env.close()
print("Pong training complete!")