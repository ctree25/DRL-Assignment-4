import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from dmc import make_dmc_env
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs_dim = 5
act_dim = 1
act_limit = 1.0

def make_env():
    env_name = "cartpole-balance"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t * act_limit
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def select_action(self, obs, deterministic=True):
        mu, std = self.forward(obs)
        if deterministic:
            x = mu
        else:
            normal = torch.distributions.Normal(mu, std)
            x = normal.sample()
        y = torch.tanh(x)
        return y * act_limit
    
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_out(x)

class ReplayBuffer:
    def __init__(self, size=100_000):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32, device=device),
            torch.tensor(a, dtype=torch.float32, device=device),
            torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(s_, dtype=torch.float32, device=device),
            torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
    

def update(buffer, batch_size=256, gamma=0.99, tau=0.005):
    s, a, r, s_, d = buffer.sample(batch_size)

    with torch.no_grad():
        a_, log_prob_ = actor.sample(s_)
        q1_target = target_critic1(s_, a_)
        q2_target = target_critic2(s_, a_)
        q_min = torch.min(q1_target, q2_target) - alpha() * log_prob_
        q_backup = r + (1 - d) * gamma * q_min

    q1 = critic1(s, a)
    q2 = critic2(s, a)
    critic1_loss = F.mse_loss(q1, q_backup)
    critic2_loss = F.mse_loss(q2, q_backup)

    critic1_opt.zero_grad()
    critic1_loss.backward()
    critic1_opt.step()

    critic2_opt.zero_grad()
    critic2_loss.backward()
    critic2_opt.step()

    a_sample, log_pi = actor.sample(s)
    q1_pi = critic1(s, a_sample)
    q2_pi = critic2(s, a_sample)
    q_pi = torch.min(q1_pi, q2_pi)
    actor_loss = (alpha() * log_pi - q_pi).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

actor = Actor(obs_dim, act_dim).to(device)
critic1 = Critic(obs_dim, act_dim).to(device)
critic2 = Critic(obs_dim, act_dim).to(device)
target_critic1 = Critic(obs_dim, act_dim).to(device)
target_critic2 = Critic(obs_dim, act_dim).to(device)
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
critic1_opt = optim.Adam(critic1.parameters(), lr=3e-4)
critic2_opt = optim.Adam(critic2.parameters(), lr=3e-4)

log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=device)
alpha_opt = optim.Adam([log_alpha], lr=3e-4)
alpha = lambda: log_alpha.exp()
target_entropy = -act_dim


def main():
    env = make_env()
    buffer = ReplayBuffer()
    max_episodes = 200
    batch_size = 256
    warmup_steps = 1000

    for ep in tqdm(range(max_episodes), desc="Training eps"):
        s, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            s_tensor = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                if len(buffer) > warmup_steps:
                    a, _ = actor.sample(s_tensor)
                else:
                    a = torch.tensor(env.action_space.sample()).unsqueeze(0).to(device)

            a_np = a.cpu().numpy().flatten()
            s_, r, terminated, truncated, info = env.step(a_np)
            done = terminated or truncated
            buffer.add(s, a_np, r, s_, done)
            s = s_
            ep_reward += r

            if len(buffer) > batch_size:
                update(buffer, batch_size)

     

        print(f"Episode {ep}  Reward: {ep_reward:.2f}  Alpha: {alpha().item():.4f}")
        if (ep+1)%50 == 0:
            torch.save(actor.state_dict(), f"sac_actor_cartpole_{ep+1}.pth")

    torch.save(actor.state_dict(), "sac_actor_cartpole.pth")

if __name__ == '__main__':
    main()