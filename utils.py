import torch
import numpy as np
import scipy
import logging
import matplotlib.pyplot as plt
import os
import wandb

eps = np.finfo(np.float32).eps.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(probs):
    """
    Function to sample an action from a probability distribution.
    """
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item()

def grad_ent(pt):
    grad_p = -np.log(pt)
    grad_p[grad_p > 100] = 1000
    return grad_p

def heatmap(running_avg_p, avg_p, e, folder_name):
    """
    Function to plot the heatmap of the probability distribution.

    Args:
        running_avg_p (numpy.ndarray): Running average probability distribution.
        avg_p (numpy.ndarray): Average probability distribution.
        e (int): Episode number.
        env (object): The environment object.
        folder_name (str): Folder name to save the plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

    ax1.set_title("Running Average Probability Distribution")
    ax2.set_title("Average Probability Distribution")

    ax1.imshow(running_avg_p, interpolation="nearest")
    ax2.imshow(avg_p, interpolation="nearest")

    # Add legend
    cbar1 = ax1.figure.colorbar(ax1.imshow(running_avg_p, interpolation="nearest"), ax=ax1)
    cbar2 = ax2.figure.colorbar(ax2.imshow(avg_p, interpolation="nearest"), ax=ax2)

    # Log image to wandb
    wandb.log({"Probability Distributions": wandb.Image(fig)})

    fig.savefig(os.path.join(folder_name, f"heatmap_{e}.png"))
    plt.close(fig)

def learn_policy(env, train_steps, horizon, policy, reward_fn, optimizers, gamma):
    """
    Function to learn a policy using REINFORCE.

    Args:
        env (object): The environment object.
        train_steps (int): Number of training steps.
        horizon (int): Horizon for each episode.
        policy (list): List of policy networks for each agent.
        reward_fn (function): Reward function.
        optimizers (list): List of optimizers for each agent's policy network.
        gamma (float): Discount factor.

    Returns:
        list: List of updated policy networks.
    """
    num_agents = env.num_agents

    running_reward = np.array([0. for _ in range(num_agents)])
    running_loss = np.array([0. for _ in range(num_agents)])

    print(f"======== Learning Policy ========")
    logging.info(f"======== Learning Policy ========")

    for e in range(train_steps):

        rewards = []
        saved_log_probs = []

        obs, _ = env.reset()

        ep_reward = np.array([0. for _ in range(num_agents)])

        for t in range(horizon):

            obs = [torch.from_numpy(obs[i]).float().unsqueeze(0).to(device) for i in range(num_agents)]

            actions_and_log_probs = [policy[i].select_action(obs[i]) for i in range(num_agents)]

            actions = [actions_and_log_probs[i][0] for i in range(num_agents)]
            log_probs = [actions_and_log_probs[i][1] for i in range(num_agents)]

            obs, _, _, _, _ = env.step(actions)
            reward = [reward_fn[obs[i][0], obs[i][1]] for i in range(num_agents)]
            ep_reward += reward

            rewards.append(reward)
            saved_log_probs.append(log_probs)

        # Update running reward
        running_reward = running_reward * 0.99 + ep_reward * 0.01
        if (e == 0):
            running_reward = ep_reward

        # Compute reward-to-go
        R = np.zeros(shape=(num_agents))

        policy_loss = []
        rtg = []

        for r in rewards[::-1]:
            R = r + gamma * R
            rtg.insert(0, R)
        
        rtg = torch.tensor(np.array(rtg))
        rtg = (rtg - rtg.mean()) / (rtg.std() + eps)

        # Compute policy loss
        for log_prob, reward in zip(saved_log_probs, rewards):
            policy_loss.append(-torch.tensor(log_prob) * torch.tensor(np.array(reward)))

        policy_loss = torch.stack(policy_loss).sum(dim=0).requires_grad_(True)

        # Update policy
        for i in range(num_agents):
            optimizers[i].zero_grad()
            policy_loss[i].backward()
            optimizers[i].step()

        running_loss = running_loss * 0.99 + policy_loss.detach().numpy() * 0.01

        if e % 100 == 0:
            print(f"Step {e}/{train_steps} | Running Reward: {running_reward} | Running Loss: {running_loss}")
            logging.info(f"Step {e}/{train_steps} | Running Reward: {running_reward} | Running Loss: {running_loss}")

    return policy

def execute_average_policy(env, horizon, policies, avg_rounds=2):
    """
    Function to execute the average policy over multiple rounds and calculate the average probability distribution and entropy.

    Args:
        env (object): The environment object.
        horizon (int): Horizon for each round.
        policies (list): List of policy networks for each agent.
        avg_rounds (int): Number of rounds to average over.

    Returns:
        numpy.ndarray: Average probability distribution.
        float: Average entropy.
    """
    average_p = np.zeros(shape=(env.width, env.height))
    avg_entropy = 0

    env.set_render_mode("rgb_array")

    for run in range(avg_rounds):

        obs, _ = env.reset()

        p = np.zeros(shape=(env.width, env.height))

        for t in range(horizon):
            probs = torch.stack([torch.tensor(np.zeros(shape=(env.action_space.n))) for _ in range(env.num_agents)]).to(device)
            # var = torch.stack([torch.tensor(np.zeros(shape=(env.action_space.n))) for _ in range(env.num_agents)]).to(device)

            obs = [torch.from_numpy(obs[i]).float().unsqueeze(0).to(device) for i in range(env.num_agents)]

            for policy in policies:
                prob = torch.stack([policy[i].get_probs(obs[i]) for i in range(env.num_agents)])
                probs += prob

            probs /= len(policies)
            actions = [select_action(probs[i]) for i in range(env.num_agents)]

            obs, _, _, _, _ = env.step(actions)
            for o in obs:
                p[o[0], o[1]] += 1

        p /= horizon

        average_p += p
        avg_entropy += scipy.stats.entropy(average_p.flatten())

        env.set_render_mode("rgb_array") # Disable rendering for the rest of the rounds

    average_p /= avg_rounds
    avg_entropy /= avg_rounds

    return average_p, avg_entropy

def execute(env, horizon, policy):
    """
    Function to execute a policy in the environment and calculate the resulting probability distribution.

    Args:
        env (object): The environment object.
        horizon (int): Horizon for the execution.
        policy (list): List of policy networks for each agent.

    Returns:
        numpy.ndarray: Probability distribution.
    """
    p = np.zeros(shape=(env.width, env.height))

    obs, _ = env.reset()

    for t in range(horizon):
        obs = [torch.from_numpy(obs[i]).float().unsqueeze(0) for i in range(env.num_agents)]
        actions_and_log_probs = [policy[i].select_action(obs[i]) for i in range(env.num_agents)]
        actions = [actions_and_log_probs[i][0] for i in range(env.num_agents)]

        obs, _, _, _, _ = env.step(actions)
        for o in obs:
            p[o[0], o[1]] += 1

    p /= horizon

    return p

def execute_random(env, horizon):
    """
    Function to execute random actions in the environment and calculate the resulting probability distribution.

    Args:
        env (object): The environment object.
        horizon (int): Horizon for the execution.

    Returns:
        numpy.ndarray: Probability distribution.
    """
    p = np.zeros(shape=(env.width, env.height))

    obs, _ = env.reset()

    for t in range(horizon):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        obs, _, _, _, _ = env.step(actions)
        for o in obs:
            p[o[0], o[1]] += 1

    p /= horizon

    return p
