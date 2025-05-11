import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import Actor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        # self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor.load_state_dict(torch.load("sac_actor.pth", map_location=device))
        self.actor.eval()

    def act(self, observation):
        # return self.action_space.sample()
        obs = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.select_action(obs, deterministic=True)  # deterministic for eval
        return action.cpu().numpy().flatten()