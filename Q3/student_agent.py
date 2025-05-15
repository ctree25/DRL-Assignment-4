import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import Actor


obs_dim = 67
act_dim = 21
act_limit = 1.0
device = "cpu"
actor_path = "./sac_actor_humanoid_32600.pth"


class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float64)
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))
        self.actor.eval()

    def act(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.select_action(obs, deterministic=True)
        return action.cpu().numpy().flatten()
