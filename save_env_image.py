import numpy as np
import argparse
import yaml
import pandas as pd
import gymnasium as gym
from gymnasium.envs.registration import register

import matplotlib.pyplot as plt


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='empty_10_2_short.yaml', help='config file')
    args = parser.parse_args()

    # load config file
    with open('configs/' + args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    file_name = args.config[:-5]

    env_name = config["env_name"] + "Env"
    size = config["size"]
    if size == "None":
        height = config["height"]
        width = config["width"]
    else:
        height = size
        width = size
    num_agents = config["num_agents"]
    init_pos = config["init_pos"] if config["init_pos"] != "None" else None

    register(
        id='GridWorld-v0',
        entry_point='marl_grid.envs:' + env_name
    )

    # make env
    env = gym.make("GridWorld-v0", height=height, width=width, num_agents=num_agents, initial_positions=init_pos)

    # set render mode
    env.set_render_mode("rgb_array")

    env.reset()

    img = env.render()

    # save image
    plt.imsave(f"images/{file_name}_env.pdf", img)