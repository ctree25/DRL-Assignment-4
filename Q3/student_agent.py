import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_f_2 import Actor, make_env


device = "cpu"
actor_path = "./sac_actor_humanoid_900.pth"



class Agent(object):
    def __init__(self):
        env = make_env()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float64)
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))
        self.actor.eval()

    def act(self, observation):
        a = self.actor.select_action(observation)
        return a
