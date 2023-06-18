import numpy as np
import argparse
import yaml
import pandas as pd

import matplotlib.pyplot as plt


if __name__ == "__main__":

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='empty_10_2_short', help='config file')
    args = parser.parse_args()

    file_name = args.config

    # load running average and running average baseline
    running_avg = pd.read_csv("data/" + file_name + "_running_average.csv", skiprows=0).iloc[:, 1]
    running_avg_baseline = pd.read_csv("data/" + file_name + "_running_average_baseline.csv", skiprows=0).iloc[:, 1]

    # load average and average baseline
    average = pd.read_csv("data/" + file_name + "_average.csv", skiprows=0).iloc[:, 1]
    average_baseline = pd.read_csv("data/" + file_name + "_average_baseline.csv", skiprows=0).iloc[:, 1]

    # plot running average and running average baseline
    plt.figure(figsize=(4, 3))
    plt.plot(running_avg, label="MEME Policy")
    plt.plot(running_avg_baseline, label="Random")
    # add noise from average and average baseline
    plt.fill_between(np.arange(len(running_avg)), running_avg - np.abs(average - running_avg) * 0.5, running_avg + np.abs(average - running_avg) * 0.5, alpha=0.2)
    plt.fill_between(np.arange(len(running_avg_baseline)), running_avg_baseline - np.abs(average_baseline - running_avg_baseline) * 0.5, running_avg_baseline + np.abs(average_baseline - running_avg_baseline) * 0.5, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Running Average Entropy")
    plt.legend(loc="upper left")
    plt.savefig("images/" + file_name + ".pdf", bbox_inches='tight')
    plt.close()