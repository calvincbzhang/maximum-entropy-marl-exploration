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

def compute_entropy_upper_bound(env):
    # count non-walls in grid
    num_states = 0
    for i in range(env.size):
        for j in range(env.size):
            if not (env.grid.get(i, j) and env.grid.get(i, j).type == "wall"):
                num_states += 1

    p = 1.0 / num_states

    # compute entropy upper bound
    entropy_upper = -num_states * p * np.log(p)

    return entropy_upper

def heatmap(running_avg_p, avg_p, running_avg_p_baseline, p_baseline, e, folder_name):
    """
    Function to plot the heatmap of the probability distribution.

    Args:
        running_avg_p (numpy.ndarray): Running average probability distribution.
        avg_p (numpy.ndarray): Average probability distribution.
        running_avg_p_baseline (numpy.ndarray): Running average probability distribution for baseline.
        p_baseline (numpy.ndarray): Average probability distribution for baseline.
        e (int): Episode number.
        folder_name (str): Folder name to save the heatmap.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 10))

    axs[0, 0].imshow(running_avg_p, interpolation="nearest")
    axs[0, 0].set_title(f"Running Average Probability Distribution")

    axs[0, 1].imshow(avg_p, interpolation="nearest")
    axs[0, 1].set_title(f"Average Probability Distribution")

    axs[1, 0].imshow(running_avg_p_baseline, interpolation="nearest")
    axs[1, 0].set_title(f"Running Average Probability Distribution (Baseline)")

    axs[1, 1].imshow(p_baseline, interpolation="nearest")
    axs[1, 1].set_title(f"Average Probability Distribution (Baseline)")

    # Add legend
    cbar1 = axs[0, 0].figure.colorbar(axs[0, 0].imshow(running_avg_p, interpolation="nearest"), ax=axs[0, 0])
    cbar2 = axs[0, 1].figure.colorbar(axs[0, 1].imshow(avg_p, interpolation="nearest"), ax=axs[0, 1])
    cbar3 = axs[1, 0].figure.colorbar(axs[1, 0].imshow(running_avg_p_baseline, interpolation="nearest"), ax=axs[1, 0])
    cbar4 = axs[1, 1].figure.colorbar(axs[1, 1].imshow(p_baseline, interpolation="nearest"), ax=axs[1, 1])

    # Log image to wandb
    wandb.log({"Probability Distributions": [wandb.Image(fig, caption=f"Episode {e}")]})

    for ax in axs.flat:
        ax.set(xlabel="x", ylabel="y")
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(f"{folder_name}/heatmap_{e}.png")
    plt.close()

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
        obs = [torch.from_numpy(obs[i]).float().unsqueeze(0).to(device) for i in range(env.num_agents)]
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
