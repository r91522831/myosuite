from myosuite.utils import gym

# env = gym.make('MyoHandTHandleLift-v0')
env = gym.make('MyoHandCupDrink-v0')
# env = gym.make('MyoHandCubelargePass-v0')


env.reset()

for _ in range(1000):
    env.unwrapped.mj_render()  # ✅ access the base env
    env.step(env.action_space.sample())

env.close()