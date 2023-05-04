import gymnasium as gym
import time
from gymnasium.envs.registration import register
import argparse

def main():
    register(
        id='empty-v0',
        entry_point='marl_grid.envs:VertWallEnv10x10x2',
    )
    env = gym.make('empty-v0')

    num_agents = env.num_agents

    _ = env.reset()

    while True:
        env.render()
        time.sleep(0.3)

        ac = [env.action_space.sample() for _ in range(num_agents)]

        obs, reward, terminated, truncated, info = env.step(ac)

        if terminated:
            break

if __name__ == "__main__":
    main()