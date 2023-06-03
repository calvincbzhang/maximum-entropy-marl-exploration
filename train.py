import gymnasium as gym
from gymnasium.envs.registration import register

import time
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SoftMaxPolicy(nn.Module):
    """
    Simple policy with softmax parametrization
    """

    def __init__(self, input_size, output_size):
        super(SoftMaxPolicy, self).__init__()

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return F.softmax(x, dim=-1)
    

# Simple neural network policy with 3 hidden layers and relu activations
class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.softmax(self.linear3(x), dim=-1)


def main(args):

    env_name = args.env_name + "Env"
    num_agents = args.num_agents

    register(
        id='GridWorld-v0',
        entry_point='marl_grid.envs:' + env_name
    )

    env = gym.make("GridWorld-v0", size = args.size, num_agents = args.num_agents)

    agents = [Policy(env.observation_space.shape[1], env.action_space.n) for _ in range(num_agents)]

    optimizers = [optim.Adam(params=agent.parameters(), lr=args.lr) for agent in agents]

    for agent in agents:
        agent.train()  # Set the agent to training mode
        for param in agent.parameters():
            param.requires_grad = True  # Enable gradient computation

    for k in range(args.num_episodes):

        print(f"==================== Episode {k} ====================")

        env.set_render_mode("rgb_array")

        trajectories = []
        occupancies = np.zeros((args.size, args.size))

        # Collect a batch of trajectories with the current policy
        for _ in range(args.batch_size):

            obs, _ = env.reset()
            trajectory = []

            for _ in range(args.horizon):

                trajectory.append(obs)

                probs = [agents[i](torch.tensor(obs[i], dtype=torch.float32)) for i in range(num_agents)]
                samplers = [torch.distributions.Categorical(probs[i]) for i in range(num_agents)]
                actions = [samplers[i].sample().item() for i in range(num_agents)]

                obs, _, _, _, _ = env.step(actions)

                trajectory.append(actions)

            trajectories.append(trajectory)

        # Compute the occupancy measure
        for trajectory in trajectories:
            for state in trajectory[::2]:
                for agent in range(num_agents):
                    occupancies[state[agent][0], state[agent][1]] += 1

        # Normalize the occupancy measure
        occupancies = occupancies / (args.batch_size * args.horizon * args.num_agents)

        # Convert the occupancy measure to a tensor
        occupancies = torch.tensor(occupancies, dtype=torch.float32, requires_grad=True)

        # Compute the entropy
        entropy = -torch.nansum(occupancies * torch.log(occupancies))
        print(f"Entropy: {entropy}")

        # Reward function for every state (for every state -(log(occu(s))) + 1)
        reward_fn = - (torch.log(occupancies) + 1)
        # Set infinities to a large number
        reward_fn[torch.isinf(reward_fn)] = 10
        # Zero out the borders
        reward_fn[0, :] = 0
        reward_fn[-1, :] = 0
        reward_fn[:, 0] = 0
        reward_fn[:, -1] = 0
        # print(f"Occupancies: {occupancies}")
        # print(f"Reward function: {reward_fn}")

        # So far, we have estimated a reward function
        # Now, we need to use this reward function to implement a policy gradient algorithm

        print("==================== Running policy gradient ====================")

        env.set_render_mode("rgb_array")

        for _ in range(10):

            rewards = []
            actions = []
            states = []

            obs, _ = env.reset()

            gamma = 0.99

            for _ in range(args.horizon):

                probs = [agents[i](torch.tensor(obs[i], dtype=torch.float32)) for i in range(num_agents)]
                samplers = [torch.distributions.Categorical(probs[i]) for i in range(num_agents)]
                action = [samplers[i].sample().item() for i in range(num_agents)]

                new_obs, _, _, _, _ = env.step(action)

                states.append(obs)
                actions.append(action)
                rewards.append([reward_fn[obs[i][0], obs[i][1]] for i in range(num_agents)])

                obs = new_obs

            rewards = torch.tensor(rewards, dtype=torch.float32)
            # Compute rewards-to-go
            R = torch.zeros_like(rewards)  # Initialize R with zeros

            for agent in range(num_agents):
                for i in range(rewards.size(0)):
                    R[i, agent] = torch.sum(rewards[i:, agent] * (gamma ** torch.arange(i, rewards.size(0))))

            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions))

            probs = [agents[i](states[:, i, :]) for i in range(num_agents)]
            samplers = [torch.distributions.Categorical(probs[i]) for i in range(num_agents)]
            log_probs = [-samplers[i].log_prob(actions[:, i]) for i in range(num_agents)]
            losses = [torch.sum(log_probs[i] * R[:, i]) for i in range(num_agents)]

            for i in range(num_agents):
                optimizers[i].zero_grad()
                losses[i].backward()
                optimizers[i].step()

            print(f"Losses: {losses}")

        # Test the policy

        print("==================== Testing the policy ====================")

        env.set_render_mode("human")

        obs, _ = env.reset()

        for _ in range(args.horizon):

            probs = [agents[i](torch.tensor(obs[i], dtype=torch.float32)) for i in range(num_agents)]
            samplers = [torch.distributions.Categorical(probs[i]) for i in range(num_agents)]
            actions = [samplers[i].sample().item() for i in range(num_agents)]

            obs, _, _, _, _ = env.step(actions)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="Empty", help="Name of the environment")
    parser.add_argument("--size", type=int, default=10, help="Size of the gridworld")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--batch_size" , type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--horizon", type=int, default=10, help="Horizon")

    args = parser.parse_args()

    main(args)