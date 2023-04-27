import gymnasium as gym
import time
from gymnasium.envs.registration import register
import argparse

def main():
    register(
        id='empty-v0',
        entry_point='marl_grid.envs:EmptyEnvTest',
    )
    env = gym.make('empty-v0')

    _ = env.reset()

    while True:
        env.render()
        time.sleep(0.1)

        ac = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(ac)

        if terminated:
            break

if __name__ == "__main__":
    main()