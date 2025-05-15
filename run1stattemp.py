import os
import numpy as np
import cv2  # OpenCV for video writing
from stable_baselines3 import PPO
from myosuite.utils import gym
from tqdm import tqdm

# Create the environment
env = gym.make('MyoHandTHandleLift-v0')

# Train the model
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=1000)

# Evaluate the policy over 20 episodes
all_rewards = []
for _ in tqdm(range(20)):  # 20 test episodes
    ep_rewards = []
    done = False
    obs, _ = env.reset()  # Only use the observation, discard info
    while not done:
        # Get the next action from the policy
        action, _ = model.predict(obs)
        # Take an action based on the current observation
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_rewards.append(reward)
    all_rewards.append(np.sum(ep_rewards))

# Print average reward
print(f"Average reward: {np.mean(all_rewards)} over 20 episodes")

# Save your trained policy
save_path = os.path.join('MySubmission2023', 'agent', 'policies', 'MyMyoChallengePolicy.pkl')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)

# Render trained policy and save frames
frames = []
for _ in tqdm(range(5)):  # 5 random test episodes
    obs, _ = env.reset()  # Reset the environment and get the initial observation
    done = False
    while not done:
        # Capture the frame using the simulator renderer
        frame = env.sim.renderer.render_offscreen(width=400, height=400, camera_id=1)
        frames.append(frame)

        # Get the next action from the policy
        action, _ = model.predict(obs)

        # Take an action based on the current observation
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()

# Save video using OpenCV
os.makedirs('videos', exist_ok=True)
height, width, layers = frames[0].shape  # Get the frame dimensions
video_writer = cv2.VideoWriter('videos/test_policy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Write all frames to the video file
for frame in frames:
    video_writer.write(frame)

video_writer.release()

print("Video saved at 'videos/test_policy.mp4'.")
print("Average episode reward:", np.mean(all_rewards))

env.reset()

for _ in range(1000):
    env.unwrapped.mj_render()  # ✅ access the base env
    env.step(env.action_space.sample())

env.close()