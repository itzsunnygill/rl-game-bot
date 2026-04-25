import streamlit as st
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from agent.agent import DQNAgent
import time

st.set_page_config(
    page_title="RL Game Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RL Game Bot — Live Demo")
st.markdown("A Reinforcement Learning agent trained to play CartPole using PPO and DQN algorithms.")

# Sidebar
st.sidebar.title("⚙️ Settings")
algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    ["PPO (Stable-Baselines3)", "DQN (From Scratch)", "Random Agent"]
)
episodes = st.sidebar.slider("Number of Episodes", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This project was built as a 2nd year ML college project.")
st.sidebar.markdown("**Algorithms:** PPO, DQN, Random")
st.sidebar.markdown("**Game:** CartPole-v1")

# Run button
if st.button("▶ Run Agent", type="primary"):
    env = gym.make("CartPole-v1")
    env = TimeLimit(env, max_episode_steps=500)

    # Load model
    if algorithm == "PPO (Stable-Baselines3)":
        model = PPO.load("models/cartpole_ppo")
    elif algorithm == "DQN (From Scratch)":
        agent = DQNAgent(4, 2)
        agent.policy_net.load_state_dict(
            torch.load("models/dqn_best.pth", map_location=torch.device('cpu'))
        )
        agent.epsilon = 0.0

    scores = []
    progress = st.progress(0)
    status = st.empty()

    for ep in range(episodes):
        obs, _ = env.reset()
        total = 0
        done = truncated = False

        while not done and not truncated:
            if algorithm == "PPO (Stable-Baselines3)":
                action, _ = model.predict(obs)
            elif algorithm == "DQN (From Scratch)":
                action = agent.act(obs)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, _ = env.step(action)
            total += 1

        scores.append(total)
        progress.progress((ep + 1) / episodes)
        status.text(f"Episode {ep+1}/{episodes} — Score: {total}")

    env.close()

    # Results
    st.success(f"Done! Average Score: {np.mean(scores):.0f} / 500")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Score", f"{np.mean(scores):.0f}")
    col2.metric("Best Score", f"{max(scores)}")
    col3.metric("Worst Score", f"{min(scores)}")

    # Graph
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(scores)+1), scores, color='steelblue', alpha=0.8)
    ax.axhline(y=200, color='red', linestyle='--', label='Solved threshold (200)')
    ax.axhline(y=np.mean(scores), color='green', linestyle='--', label=f'Average ({np.mean(scores):.0f})')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title(f"{algorithm} — Episode Scores")
    ax.legend()
    st.pyplot(fig)

    # Score table
    st.markdown("### Episode Breakdown")
    for i, score in enumerate(scores):
        col1, col2 = st.columns([1, 4])
        col1.write(f"Episode {i+1}")
        col2.progress(score/500)

st.markdown("---")
st.markdown("### 📊 Pre-trained Results")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**CartPole — Trained vs Random**")
    st.image("comparison.png")
with col2:
    st.markdown("**Algorithm Comparison**")
    st.image("algorithm_comparison.png")

st.markdown("---")
st.markdown("Built with ❤️ using PyTorch, Stable-Baselines3, Gymnasium and Streamlit")