# verify.py — Run this to confirm your RL environment is set up correctly
# Usage: python verify.py

import sys

print("=" * 50)
print(" RL Bot — Environment Verification")
print("=" * 50)

# 1. Check Python version
assert sys.version_info >= (3, 10), (
    f"Python 3.10+ is required. You have {sys.version_info.major}.{sys.version_info.minor}. "
    "Download the latest from https://python.org"
)
print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# 2. Check PyTorch
try:
    import torch
    cuda_info = f"CUDA available ({'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'})"
    print(f"✅ PyTorch {torch.__version__} | {cuda_info}")
except ImportError:
    print("❌ PyTorch not found — run: pip install torch torchvision torchaudio")
    sys.exit(1)

# 3. Check Gymnasium + CartPole
try:
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    env.close()
    print(f"✅ Gymnasium {gym.__version__} — CartPole-v1 loaded")
except ImportError:
    print("❌ Gymnasium not found — run: pip install gymnasium")
    sys.exit(1)

# 4. Check Stable-Baselines3
try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    print(f"✅ Stable-Baselines3 {sb3.__version__} — PPO imported")
except ImportError:
    print("❌ Stable-Baselines3 not found — run: pip install stable-baselines3[extra]")
    sys.exit(1)

# 5. Quick agent training test
print("\n⏳ Running quick PPO training test on CartPole (1000 steps)...")
env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1000)
env.close()
print("✅ PPO trained on CartPole-v1 for 1000 steps")

# 6. Optional: Check Atari support
print("\n⏳ Checking Atari / Pong support (optional)...")
try:
    import ale_py
    env = gym.make("ALE/Pong-v5")
    env.reset()
    env.close()
    print("✅ Atari / Pong-v5 is ready")
except Exception as e:
    print(f"⚠️  Atari not available (optional): {e}")
    print("   To enable: pip install ale-py gymnasium[atari] autorom[accept-rom-license]")
    print("   Then run:  AutoROM --accept-license")

# 7. Optional: Check NumPy, Matplotlib, TensorBoard
print("\n⏳ Checking optional extras...")
for pkg, import_name in [("numpy", "numpy"), ("matplotlib", "matplotlib"), ("cv2", "opencv-python"), ("tensorboard", "tensorboard")]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"✅ {import_name} {ver}")
    except ImportError:
        print(f"⚠️  {import_name} not installed (optional)")

print("\n" + "=" * 50)
print("🎉 Setup works! You're ready to train RL agents.")
print("=" * 50)
