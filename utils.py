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

def geometric_weights(distributions, gamma=0.90):
    N = len(distributions)
    weights = [gamma**(N-i) for i in range(N)]
    return weights

def get_weights(distributions):
    weights = np.ones(len(distributions))/float(len(distributions)) 
    weights = geometric_weights(distributions)
    weights = np.absolute(weights) / np.sum(weights)
    print(weights)
    print(weights.sum())
    
    if not np.isclose(weights.sum(), 1, rtol=1e-8):
        weights /= weights.sum()
        print('re-normalizing: %f' % weights.sum())
    
    return weights

def compute_entropy_upper_bound(env):
    # count non-walls in grid
    num_states = 0
    for i in range(env.width):
        for j in range(env.height):
            if not (env.grid.get(i, j) and env.grid.get(i, j).type == "wall"):
                num_states += 1

    p = 1.0 / num_states

    # compute entropy upper bound
    entropy_upper = -num_states * p * np.log(p)

    return entropy_upper

def heatmap(running_avg_p, avg_p, running_avg_p_baseline, p_baseline, e, height, width, folder_name):
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

    # Find the widest range
    all_data = [running_avg_p, avg_p, running_avg_p_baseline, p_baseline]
    vmin = min(np.min(data) for data in all_data)
    vmax = max(np.max(data) for data in all_data)

    print(f"================ Heatmap ================")
    print(f"=========== vmax: {vmax} ===========")

    logging.info(f"================ Heatmap ================")
    logging.info(f"=========== vmax: {vmax} ===========")

    if height/float(width) < 2:
        fig, axs = plt.subplots(2, 2, figsize=(1.5 * width, height))
        # Plot the heatmap with the range
        axs[0, 0].imshow(running_avg_p, interpolation="none", vmin=vmin, vmax=vmax)
        axs[0, 0].set_title(f"Running Average Probability Distribution")

        axs[0, 1].imshow(avg_p, interpolation="none", vmin=vmin, vmax=vmax)
        axs[0, 1].set_title(f"Average Probability Distribution")

        axs[1, 0].imshow(running_avg_p_baseline, interpolation="none", vmin=vmin, vmax=vmax)
        axs[1, 0].set_title(f"Running Average Probability Distribution (Baseline)")

        axs[1, 1].imshow(p_baseline, interpolation="none", vmin=vmin, vmax=vmax)
        axs[1, 1].set_title(f"Average Probability Distribution (Baseline)")

        # Add legend
        cbar = axs[0, 0].figure.colorbar(axs[0, 0].imshow(running_avg_p, interpolation="none", vmin=vmin, vmax=vmax), ax=axs)
    else:
        fig, axs = plt.subplots(4, 1, figsize=(1.01 * height, 5 * width))
        # Plot the heatmap with the range
        axs[0].imshow(running_avg_p, interpolation="none", vmin=vmin, vmax=vmax)
        axs[0].set_title(f"Running Average Probability Distribution")

        axs[1].imshow(avg_p, interpolation="none", vmin=vmin, vmax=vmax)
        axs[1].set_title(f"Average Probability Distribution")

        axs[2].imshow(running_avg_p_baseline, interpolation="none", vmin=vmin, vmax=vmax)
        axs[2].set_title(f"Running Average Probability Distribution (Baseline)")

        axs[3].imshow(p_baseline, interpolation="none", vmin=vmin, vmax=vmax)
        axs[3].set_title(f"Average Probability Distribution (Baseline)")

        # Add legend
        cbar = axs[0].figure.colorbar(axs[0].imshow(running_avg_p, interpolation="nearest", vmin=vmin, vmax=vmax), ax=axs)

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

        reward = [reward_fn[obs[i][0], obs[i][1]] for i in range(num_agents)]
        ep_reward += reward

        rewards.append(reward)

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

        rtg = []

        for r in rewards[::-1]:
            R = r + gamma * R
            rtg.insert(0, R)
        
        rtg = torch.tensor(np.array(rtg))
        rtg = (rtg - rtg.mean()) / (rtg.std() + eps)

        joint_loss = np.array([0. for _ in range(num_agents)])

        for i in range(num_agents):

            policy_loss = []

            # Compute policy loss
            for log_prob, reward in zip(saved_log_probs, rewards):
                policy_loss.append((-log_prob[i]) * reward[i])

            policy_loss = torch.stack(policy_loss).sum()

            # Update policy
            optimizers[i].zero_grad()
            policy_loss.backward()
            optimizers[i].step()

            joint_loss[i] = policy_loss.item()

        running_loss = running_loss * 0.99 + joint_loss * 0.01

        if e % 100 == 0:
            print(f"Step {e}/{train_steps} | Running Reward: {running_reward} | Running Loss: {running_loss}")
            logging.info(f"Step {e}/{train_steps} | Running Reward: {running_reward} | Running Loss: {running_loss}")

    return policy

def execute_average_policy(env, horizon, policies, weights, entropies, avg_rounds=10):
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

    # if len(policies) > 10:
    #     # get indices of most entropic policies
    #     indices = np.argsort(entropies)[-10:]
    #     policies = [policies[i] for i in indices]
    #     entropies = [entropies[i] for i in indices]
    #     weights = [weights[i] for i in indices]
    #     # normalize weights
    #     weights = [w / sum(weights) for w in weights]

    average_p = np.zeros(shape=(env.width, env.height))
    avg_entropy = 0

    env.set_render_mode("rgb_array")

    no_walls = np.ones(shape=(env.width, env.height))

    for run in range(avg_rounds):

        obs, _ = env.reset()

        p = np.zeros(shape=(env.width, env.height))

        for o in obs:
            p[o[0], o[1]] += 1

        for t in range(horizon):
            probs = torch.stack([torch.tensor(np.zeros(shape=(env.action_space.n))) for _ in range(env.num_agents)]).to(device)
            # var = torch.stack([torch.tensor(np.zeros(shape=(env.action_space.n))) for _ in range(env.num_agents)]).to(device)

            obs = [torch.from_numpy(obs[i]).float().unsqueeze(0).to(device) for i in range(env.num_agents)]

            for policy, weight, entropy in zip(policies, weights, entropies):
                prob = torch.stack([policy[i].get_probs(obs[i]) for i in range(env.num_agents)])
                probs += (prob * weight)

            probs /= (len(policies) * sum(weights))

            # epsilon greedy
            # if np.random.uniform() < 0.1:
            actions = [select_action(probs[i]) for i in range(env.num_agents)]
            # else:
                # actions = [torch.argmax(probs[i]).item() for i in range(env.num_agents)]

            obs, _, _, _, info = env.step(actions)

            wall_pos = info["wall_pos"]

            for w in wall_pos:
                no_walls[w[0], w[1]] = 0

            for o in obs:
                p[o[0], o[1]] += 1

        p /= ((horizon+1) * env.num_agents)

        average_p += p
        avg_entropy += scipy.stats.entropy(average_p.flatten())

        env.set_render_mode("rgb_array") # Disable rendering for the rest of the rounds

    average_p /= avg_rounds
    avg_entropy /= avg_rounds

    return average_p, avg_entropy, no_walls

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

    average_p = np.zeros(shape=(env.width, env.height))

    for round in range(10):
        p = np.zeros(shape=(env.width, env.height))

        obs, _ = env.reset()
        for o in obs:
            p[o[0], o[1]] += 1

        for t in range(horizon):
            obs = [torch.from_numpy(obs[i]).float().unsqueeze(0).to(device) for i in range(env.num_agents)]

            # epsilon greedy
            # if np.random.uniform() < 0.1:
            actions = [policy[i].select_action(obs[i])[0] for i in range(env.num_agents)]
            # else:
                # actions = [policy[i].select_action_greedy(obs[i]) for i in range(env.num_agents)]

            obs, _, _, _, _ = env.step(actions)
            for o in obs:
                p[o[0], o[1]] += 1

        p /= ((horizon+1) * env.num_agents)

        average_p += p
    
    average_p /= 10

    return average_p

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
    for o in obs:
        p[o[0], o[1]] += 1

    for t in range(horizon):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        obs, _, _, _, _ = env.step(actions)
        for o in obs:
            p[o[0], o[1]] += 1

    p /= ((horizon+1) * env.num_agents)

    return p
