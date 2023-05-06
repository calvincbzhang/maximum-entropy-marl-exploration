import gymnasium as gym
from gymnasium.envs.registration import register

import time
import argparse


def compute_occupancy(trajectories, agent_id):
    """
    Compute the occupancy measure for a single agent
    :param trajectories: list of trajectories
    :param agent_id: id of the agent
    :return: occupancy measure
    """




def main(args):

    env_name = args.env_name + "Env"
    num_agents = args.num_agents

    register(
        id='GridWorld-v0',
        entry_point='marl_grid.envs:' + env_name
    )

    env = gym.make("GridWorld-v0", size = args.size, num_agents = args.num_agents)

    for k in range(args.num_episodes):

        trajectories = []
        occupancies = []

        # Collect a batch of trajectories with the current policy
        for i in range(args.batch_size):

            obs, _ = env.reset()
            trajectory = [obs]

            while True:

                env.render()
                time.sleep(0.001)

                # TODO - replace this with the output of your policy
                actions = [env.action_space.sample() for _ in range(num_agents)]

                obs, _, terminated, _, _ = env.step(actions)

                trajectory.append(actions)
                trajectory.append(obs)

                if terminated:
                    break

            trajectories.append(trajectory)

        for i in range(num_agents):
            # Compute the local occupancy measure for each agent
            occupancy = compute_occupancy(trajectories, i)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="Empty", help="Name of the environment")
    parser.add_argument("--size", type=int, default=10, help="Size of the gridworld")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--batch_size" , type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes")

    args = parser.parse_args()

    main(args)