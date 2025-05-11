import gymnasium
import numpy as np
import torch
from train import Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

act_limit = 1.0
obs_dim = 5
act_dim = 1

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor.load_state_dict(torch.load("sac_actor_cartpole_100.pth", map_location=device))
        self.actor.eval()

    def act(self, observation):
        # return self.action_space.sample()
        obs = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.select_action(obs, deterministic=True)
        return action.cpu().numpy().flatten()



