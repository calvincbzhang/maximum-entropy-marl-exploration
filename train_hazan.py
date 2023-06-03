import gymnasium as gym
from gymnasium.envs.registration import register

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy


class Policy(nn.Module):
    def __init__(self, env, gamma, lr, obs_dim, action_dim):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(obs_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, action_dim)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()

        self.env = env
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        action_scores = self.linear3(x)
        return F.softmax(action_scores, dim=1)
        
    def get_probs(self, obs):
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        probs = self.forward(obs)
        return probs.squeeze(0)
    
    def select_action(self, obs):
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        probs = self.forward(obs)
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)
    

def select_action(probs):
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item()


def grad_ent(pt):
    grad_p = -np.log(pt)
    grad_p[grad_p > 100] = 1000
    return grad_p


def main(args):

    env_name = args.env_name + "Env"
    num_agents = args.num_agents

    eps = np.finfo(np.float32).eps.item() 

    register(
        id='GridWorld-v0',
        entry_point='marl_grid.envs:' + env_name
    )

    env = gym.make("GridWorld-v0", size = args.size, num_agents = args.num_agents)

    reward_fn = np.zeros(shape=(args.size, args.size))
    online_reward_fn = np.zeros(shape=(args.size, args.size))

    running_avg_p = np.zeros(shape=(args.size, args.size))
    running_avg_ent = 0
    running_avg_entropies = []
    running_avg_ps = []

    # running_avg_p_online = np.zeros(shape=(args.size, args.size))
    # running_avg_ent_online = 0
    # running_avg_entropies_online = []
    # running_avg_ps_online = []

    # running_avg_p_baseline = np.zeros(shape=(args.size, args.size))
    # running_avg_ent_baseline = 0
    # running_avg_entropies_baseline = []
    # running_avg_ps_baseline = []

    # online_average_ps = []
    
    policies = []

    # online_policies = []

    for e in range(args.num_episodes):

        env.set_render_mode("rgb_array")

        policy = [Policy(env, args.gamma, args.lr, env.observation_space.shape[0], env.action_space.n) for _ in range(num_agents)]
        optimizers = [torch.optim.Adam(policy[i].parameters(), lr=args.lr) for i in range(num_agents)]
        # online_policy = [Policy(env, args.gamma, args.lr, env.observation_space.shape[0], env.action_space.n) for _ in range(num_agents)]
        # online_optimizers = [torch.optim.Adam(online_policy[i].parameters(), lr=args.lr) for i in range(num_agents)]

        if e != 0:
            # learn policy
            running_reward = np.array([0. for _ in range(num_agents)])
            running_loss = np.array([0. for _ in range(num_agents)])
            for i_episode in range(100):
                obs, _ = env.reset()
                ep_reward = np.array([0. for _ in range(num_agents)])
                rewards = []
                saved_log_probs = []
                for t in range(args.horizon):
                    actions_and_log_probs = [policy[i].select_action(obs[i]) for i in range(num_agents)]
                    actions = [actions_and_log_probs[i][0] for i in range(num_agents)]
                    log_probs = [actions_and_log_probs[i][1] for i in range(num_agents)]
                    obs, _, _, _, _ = env.step(actions)
                    reward = [reward_fn[obs[i][0], obs[i][1]] for i in range(num_agents)]
                    ep_reward += reward
                    rewards.append(reward)
                    saved_log_probs.append(log_probs)

                running_reward = running_reward * 0.99 + ep_reward * 0.01
                if (i_episode == 0):
                    running_reward = ep_reward

                R = np.zeros(shape=(num_agents))
                policy_loss = []
                rtg = []

                for r in rewards[::-1]:
                    R = r + args.gamma * R
                    rtg.insert(0, R)
                
                rtg = torch.tensor(np.array(rtg))
                rtg = (rtg - rtg.mean()) / (rtg.std() + eps)

                for log_prob, reward in zip(saved_log_probs, rewards):
                    policy_loss.append(-torch.tensor(log_prob) * torch.tensor(np.array(reward)))

                policy_loss = torch.stack(policy_loss).sum(dim=0).requires_grad_(True)

                for i in range(num_agents):
                    optimizers[i].zero_grad()
                    policy_loss[i].backward()
                    optimizers[i].step()

                running_loss = running_loss * 0.99 + policy_loss.detach().numpy() * 0.01

            # learn online policy
        #     running_reward_online = np.array([0. for _ in range(num_agents)])
        #     running_loss_online = np.array([0. for _ in range(num_agents)])
        #     for i_episode in range(1000):
        #         obs, _ = env.reset()
        #         ep_reward = np.array([0. for _ in range(num_agents)])
        #         rewards = []
        #         saved_log_probs = []
        #         for t in range(args.horizon):
        #             actions_and_log_probs = [online_policy[i].select_action(obs[i]) for i in range(num_agents)]
        #             actions = [actions_and_log_probs[i][0] for i in range(num_agents)]
        #             log_probs = [actions_and_log_probs[i][1] for i in range(num_agents)]
        #             obs, _, _, _, _ = env.step(actions)
        #             reward = [online_reward_fn[obs[i][0], obs[i][1]] for i in range(num_agents)]
        #             ep_reward += reward
        #             rewards.append(reward)
        #             saved_log_probs.append(log_probs)

        #         running_reward_online = running_reward_online * 0.99 + ep_reward * 0.01
        #         if (i_episode == 0):
        #             running_reward_online = ep_reward

        #         R = np.zeros(shape=(num_agents))
        #         policy_loss = []
        #         rtg = []

        #         for r in rewards[::-1]:
        #             R = r + args.gamma * R
        #             rtg.insert(0, R)
                
        #         rtg = torch.tensor(np.array(rtg))
        #         rtg = (rtg - rtg.mean()) / (rtg.std() + eps)

        #         for log_prob, reward in zip(saved_log_probs, rewards):
        #             policy_loss.append(-torch.tensor(log_prob) * torch.tensor(np.array(reward)))

        #         policy_loss = torch.stack(policy_loss).sum(dim=0).requires_grad_(True)

        #         for i in range(num_agents):
        #             online_optimizers[i].zero_grad()
        #             policy_loss[i].backward()
        #             online_optimizers[i].step()

        #         running_loss_online = running_loss_online * 0.99 + policy_loss * 0.01

        policies.append(policy)
        # online_policies.append(online_policy)

        # Execute the average policy so far and estimate the entropy
        average_p = np.zeros(shape=(args.size, args.size))
        avg_entropy = 0

        env.set_render_mode("human")

        for run in range(2):

            obs, _ = env.reset()

            p = np.zeros(shape=(args.size, args.size))

            for t in range(args.horizon):
                probs = torch.stack([torch.tensor(np.zeros(shape=(env.action_space.n))) for _ in range(num_agents)])
                var = torch.stack([torch.tensor(np.zeros(shape=(env.action_space.n))) for _ in range(num_agents)])

                for policy in policies:
                    prob = torch.stack([policy[i].get_probs(obs[i]) for i in range(num_agents)])
                    probs += prob

                probs /= len(policies)
                actions = [select_action(probs[i]) for i in range(num_agents)]

                obs, _, _, _, _ = env.step(actions)
                for o in obs:
                    p[o[0], o[1]] += 1

            p /= args.horizon

            average_p += p
            avg_entropy += scipy.stats.entropy(average_p.flatten())

            env.set_render_mode("rgb_array")

        average_p /= 2
        avg_entropy /= 2
        
        # Execute the online policy so far and estimate the entropy
        # online_p = np.zeros(shape=(args.size, args.size))
        # online_avg_entropy = 0
        
        # for run in range(2):
                
        #         obs, _ = env.reset()
    
        #         p = np.zeros(shape=(args.size, args.size))
    
        #         for t in range(args.horizon):
        #             probs = torch.stack([torch.tensor(np.zeros(shape=(env.action_space.n))) for _ in range(num_agents)])
        #             var = torch.stack([torch.tensor(np.zeros(shape=(env.action_space.n))) for _ in range(num_agents)])
    
        #             for policy in online_policies:
        #                 prob = torch.stack([policy[i].get_probs(obs[i]) for i in range(num_agents)])
        #                 probs += prob
    
        #             probs /= len(online_policies)
        #             actions = [select_action(probs[i]) for i in range(num_agents)]
    
        #             obs, _, _, _, _ = env.step(actions)
        #             for o in obs:
        #                 p[o[0], o[1]] += 1
    
        #         p /= args.horizon
    
        #         online_p += p
        #         online_avg_entropy += scipy.stats.entropy(online_p.flatten())

        # online_p /= 2
        # online_avg_entropy /= 2

        # Get next distribtuion p
        p = np.zeros(shape=(args.size, args.size))

        obs, _ = env.reset()

        for t in range(args.horizon):
            actions_and_log_probs = [policy[i].select_action(obs[i]) for i in range(num_agents)]
            actions = [actions_and_log_probs[i][0] for i in range(num_agents)]

            obs, _, _, _, _ = env.step(actions)
            for o in obs:
                p[o[0], o[1]] += 1

        p /= args.horizon

        # p_online = np.zeros(shape=(args.size, args.size))

        # obs, _ = env.reset()

        # for t in range(args.horizon):
        #     actions_and_log_probs = [online_policy[i].select_action(obs[i]) for i in range(num_agents)]
        #     actions = [actions_and_log_probs[i][0] for i in range(num_agents)]

        #     obs, _, _, _, _ = env.step(actions)
        #     for o in obs:
        #         p_online[o[0], o[1]] += 1

        reward_fn = grad_ent(average_p)

        # Update experimental running averages.
        running_avg_ent = running_avg_ent * (e)/float(e+1) + avg_entropy/float(e+1)
        running_avg_p = running_avg_p * (e)/float(e+1) + average_p/float(e+1)
        running_avg_entropies.append(running_avg_ent)
        running_avg_ps.append(running_avg_p)

        print("--------------------------------")
        print("p=")
        print(p)

        print("average_p =") 
        print(average_p)

        print("---------------------")

        print("round_avg_ent[%d] = %f" % (e, avg_entropy))
        print("running_avg_ent = %s" % running_avg_ent)



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