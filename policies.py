import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """
    Neural Network Policy
    """
    def __init__(self, obs_dim, action_dim):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(obs_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, action_dim)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        action_scores = self.linear3(x)
        return F.softmax(action_scores, dim=1)
    
    def select_action(self, obs):
        probs = self.forward(obs)
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)
        
    def get_probs(self, obs):
        probs = self.forward(obs)
        return probs.squeeze(0)
    

class SimplePolicy(nn.Module):
    """
    Simple Neural Network Policy
    """
    def __init__(self, obs_dim, action_dim):
        super(SimplePolicy, self).__init__()

        self.linear = nn.Linear(obs_dim, action_dim)

        torch.nn.init.xavier_uniform_(self.linear.weight)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)
    
    def select_action(self, obs):
        probs = self.forward(obs)
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)
        
    def get_probs(self, obs):
        probs = self.forward(obs)
        return probs.squeeze(0)