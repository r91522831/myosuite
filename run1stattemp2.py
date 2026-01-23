from myosuite.utils import gym
from stable_baselines3 import PPO

# env = gym.make('myoElbowPose1D6MRandom-v0')

env = gym.make('MyoHandTHandleLift-v0')
# env = gym.make('MyoHandCupDrink-v0')
# env = gym.make('MyoHandAirplaneFly-v0')

model = PPO('MlpPolicy', env, verbose=0)
model.train(1000)

env.reset()

for _ in range(1000):
    env.unwrapped.mj_render()  # ✅ access the base env
    env.step(env.action_space.sample())

env.close()