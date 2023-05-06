import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.distributions import Categorical


# Define the actor and critic neural network architectures
class ActorCritic(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value
    

# Define the Maximum Entropy A2C agent
class MaxEntA2C:
    def __init__(self, obs_size, act_size, hidden_size, lr, gamma, entropy_coef):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(obs_size, act_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        policy, _ = self.policy(state)
        dist = Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def update(self, state, action, log_prob, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        log_prob = log_prob.to(self.device)
        
        _, next_value = self.policy(next_state)
        _, value = self.policy(state)
        
        advantage = reward + self.gamma * (1 - done) * next_value.detach() - value.detach()
        actor_loss = -log_prob * advantage
        critic_loss = F.smooth_l1_loss(value, reward + self.gamma * (1 - done) * next_value.detach())
        entropy = -(self.policy * torch.log(self.policy)).sum(dim=-1)
        entropy_loss = -self.entropy_coef * entropy
        
        loss = actor_loss + critic_loss + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()