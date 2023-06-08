import gymnasium as gym
from gymnasium.envs.registration import register

import numpy as np
import argparse

import torch
import torch.optim.lr_scheduler as scheduler
import os

import scipy

from policies import *
from utils import *

import logging
import datetime
import yaml

import wandb


def main(config, folder_name):

    env_name = config["env_name"] + "Env"
    size = config["size"]
    if size == "None":
        height = config["height"]
        width = config["width"]
    else:
        height = size
        width = size
    num_agents = config["num_agents"]
    init_pos = config["init_pos"]

    num_episodes = config["num_episodes"]
    train_steps = config["train_steps"]
    horizon = config["horizon"]
    gamma = config["gamma"]
    lr = config["lr"]

    register(
        id='GridWorld-v0',
        entry_point='marl_grid.envs:' + env_name
    )

    # Print and log environment information
    print(f"======== Running {env_name} with size {height} x {width} and {num_agents} agents ========")
    logging.info(f"======== Running {env_name} with size {height} x {width} and {num_agents} agents ========")
    # Print and log hyperparameters
    print(f"Parameters: num_episodes={num_episodes}, horizon={horizon}, gamma={gamma}, lr={lr}")
    logging.info(f"Parameters: num_episodes={num_episodes}, horizon={horizon}, gamma={gamma}, lr={lr}")

    env = gym.make("GridWorld-v0", height=height, width=width, num_agents=num_agents, initial_positions=init_pos)

    entropy_upper = compute_entropy_upper_bound(env)

    print(f"======== Entropy Upper Bound: {entropy_upper} ========")

    reward_fn = np.zeros(shape=(width, height))

    running_avg_p = np.zeros(shape=(width, height))
    running_avg_ent = 0
    running_avg_entropies = []
    running_avg_ps = []

    running_avg_p_baseline = np.zeros(shape=(width, height))
    running_avg_ent_baseline = 0
    running_avg_entropies_baseline = []
    running_avg_ps_baseline = []
    
    policies = []
    entropies = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"======== Running on {device} ========")
    logging.info(f"======== Running on {device} ========")

    # variable to remember walls
    non_walls = np.ones(shape=(width, height))

    for e in range(num_episodes):

        env.set_render_mode("rgb_array")

        print(f"======== Episode {e}/{num_episodes} ========")
        logging.info(f"======== Episode {e}/{num_episodes} ========")

        policy = [Policy(env.observation_space.shape[1], env.action_space.n).to(device) for _ in range(num_agents)]
        optimizers = [torch.optim.Adam(policy[i].parameters(), lr=lr) for i in range(num_agents)]
        # schedulers = [scheduler.LinearLR(optimizers[i], start_factor=0.5, total_iters=train_steps/4.) for i in range(num_agents)]

        if e != 0:
            policy = learn_policy(env, train_steps, horizon, policy, reward_fn, optimizers, gamma)

        policies.append(policy)

        # Execute the random policy and estimate the entropy
        a = 10 # average over this many rounds
        p_baseline = execute_random(env, horizon)
        avg_entropy_baseline = scipy.stats.entropy(p_baseline.flatten())
        for av in range(a-1):
            next_p_baseline = execute_random(env, horizon)
            p_baseline += next_p_baseline
            avg_entropy_baseline += scipy.stats.entropy(next_p_baseline.flatten())
        p_baseline /= float(a)
        avg_entropy_baseline /= float(a)

        # Get next distribtuion p
        p = execute(env, horizon, policy)

        entropies.append(scipy.stats.entropy(p.flatten()))

        # Execute the average policy so far and estimate the entropy
        average_p, avg_entropy, no_walls = execute_average_policy(env, horizon, policies, entropies)

        non_walls = np.logical_and(non_walls, no_walls)

        # Force first round to be equal
        if e == 0:
            average_p = p_baseline
            avg_entropy = avg_entropy_baseline

        # Get the reward function
        reward_fn = grad_ent(average_p) * non_walls

        # Update experimental running averages.
        running_avg_ent = running_avg_ent * (e)/float(e+1) + avg_entropy/float(e+1)
        running_avg_p = running_avg_p * (e)/float(e+1) + average_p/float(e+1)
        running_avg_entropies.append(running_avg_ent)
        running_avg_ps.append(running_avg_p)

        # Update baseline running averages.
        running_avg_ent_baseline = running_avg_ent_baseline * (e)/float(e+1) + avg_entropy_baseline/float(e+1)
        running_avg_p_baseline = running_avg_p_baseline * (e)/float(e+1) + p_baseline/float(e+1)
        running_avg_entropies_baseline.append(running_avg_ent_baseline)
        running_avg_ps_baseline.append(running_avg_p_baseline)

        # Log to wandb
        wandb.log({"Average Entropy": avg_entropy, "Running Average Entropy": running_avg_ent, "Average Entropy Baseline": avg_entropy_baseline, "Running Average Entropy Baseline": running_avg_ent_baseline})

        # Print and log results
        print(f"=========== p ===========")
        print(p)
        print("=========== average_p ===========")
        print(average_p)
        print("=========== reward_fn ===========")
        print(reward_fn)

        logging.info("=========== p ===========")
        logging.info(p)
        logging.info("=========== average_p ===========")
        logging.info(average_p)
        logging.info("=========== reward_fn ===========")
        logging.info(reward_fn)

        print("========================")
        logging.info("========================")

        # Print round average entropy, running average entropy, round entropy baseline, running average entropy baseline
        print(f"Round Average Entropy[{e}] = {avg_entropy} \t Running Average Entropy = {running_avg_ent} \t Entropy of p = {entropies[-1]}")
        print(f"Round Entropy Baseline[{e}] = {avg_entropy_baseline} \t Running Average Entropy Baseline = {running_avg_ent_baseline}")

        logging.info(f"Round Average Entropy[{e}] = {avg_entropy} \t Running Average Entropy = {running_avg_ent}")
        logging.info(f"Round Entropy Baseline[{e}] = {avg_entropy_baseline} \t Running Average Entropy Baseline = {running_avg_ent_baseline}")

        print("\n")
        logging.info("\n")

        heatmap(running_avg_p, average_p, running_avg_p_baseline, p_baseline, e, height, width, folder_name)
        # plot binary mask of walls
        fig = plt.figure()
        plt.imshow(non_walls)
        plt.title("Walls")
        wandb.log({"Walls": wandb.Image(fig)})
        plt.close()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='empty.yaml', help='config file')
    args = parser.parse_args()

    # load config file
    with open('configs/' + args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wandb.init(project="marl-maxent-exploration", config=config)

    # set up logging
    timestap = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if config['size'] == "None":
        folder_name = "logs/" + config['env_name'] + "_" + str(config['height']) + "_" + str(config['width']) + "_" + str(config['num_agents']) + "_" + timestap
    else:
        folder_name = "logs/" + config['env_name'] + "_" + str(config['size']) + "_" + str(config['num_agents']) + "_" + timestap
        
    os.mkdir(folder_name)
    logging.basicConfig(filename=folder_name+"/logs.txt", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    main(config, folder_name)

    wandb.finish()