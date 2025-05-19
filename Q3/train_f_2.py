import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from dmc import make_dmc_env
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import imageio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("------------------device------------------", device)



def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), -20, 2)
        return mu, log_std

    def sample(self, obs):
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std.exp())
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def select_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _ = self.sample(obs)
        return action.cpu().numpy().flatten()
    
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
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
    def __init__(self, size=1000_000):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.FloatTensor(np.array(a)).to(device),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(s_)).to(device),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class SAC:
    def __init__(self, obs_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim).to(device)
        self.target_critic1 = Critic(obs_dim, act_dim).to(device)
        self.target_critic2 = Critic(obs_dim, act_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp()
        self.target_entropy = -act_dim

        
        self.buffer = ReplayBuffer(size=1000_000)


    def update(self, buffer, batch_size=256, gamma=0.99, tau=0.005):
        s, a, r, s_, d = buffer.sample(batch_size)

        with torch.no_grad():
            a_, log_prob_ = self.actor.sample(s_)
            q1_target = self.target_critic1(s_, a_)
            q2_target = self.target_critic2(s_, a_)
            q_min = torch.min(q1_target, q2_target) - self.alpha * log_prob_
            q_backup = r + (1 - d) * gamma * q_min

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic1_loss = F.mse_loss(q1, q_backup)
        critic2_loss = F.mse_loss(q2, q_backup)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        a_sample, log_pi = self.actor.sample(s)
        q1_pi = self.critic1(s, a_sample)
        q2_pi = self.critic2(s, a_sample)
        q_pi = torch.min(q1_pi, q2_pi)
        
        
        actor_loss = (self.alpha.detach() * log_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



def main():
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    sac = SAC(obs_dim, act_dim)
    
    max_episodes = 10000
    batch_size = 256
    warmup_steps = 10000

    reward_history = [] # Store the total rewards for each episode
    avg_reward_history = []
    
    train_record_num = 8
    save_dir = f"./train_record/train_{train_record_num}/"
    os.makedirs(save_dir, exist_ok=True)
    train_log = open(f"{save_dir}log.txt", "w", buffering=1)
    avg_train_log = open(f"{save_dir}avg_log.txt", "w", buffering=1)
    train_step_log = open(f"{save_dir}step_log.txt", "w", buffering=1)


    for ep in tqdm(range(max_episodes), desc="Training eps"):
        s, _ = env.reset()
        ep_reward = 0
        done = False
        cur_step = 0

        while not done:

            with torch.no_grad():
                if len(sac.buffer) > warmup_steps:
                    a = sac.actor.select_action(s)
                else:
                    a = torch.FloatTensor(env.action_space.sample()).unsqueeze(0).to(device)
                    a = a.cpu().numpy().flatten()

            s_, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            sac.buffer.add(s, a, r, s_, done)
            s = s_
            ep_reward += r
            cur_step += 1
            print(f"Episode {ep} current step: {cur_step} Reward: {r:.2f} {abs(r)} ", file=train_step_log, flush=True)

            if len(sac.buffer) > batch_size and len(sac.buffer) > warmup_steps:
                sac.update(sac.buffer, batch_size)

        reward_history.append(ep_reward)
        
        print(f"Episode {ep}  Reward: {ep_reward:.2f}  Alpha: {sac.alpha:.4f}", file=train_log, flush=True)
        if (ep+1)%10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {ep} Avg Reward (last 10 episode): {avg_reward}", file=avg_train_log, flush=True)
            avg_reward_history.append(avg_reward)

        if (ep+1)%100 == 0:
            torch.save(sac.actor.state_dict(), f"{save_dir}sac_actor_humanoid_{ep+1}.pth")


    torch.save(sac.actor.state_dict(), f"{save_dir}sac_actor_humanoid.pth")
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training History")
    plt.savefig(f"{save_dir}training_reward_plot.png")  
    plt.close()

    plt.plot(avg_reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Avg Training History")
    plt.savefig(f"{save_dir}avg_training_reward_plot.png")  
    plt.close()

if __name__ == '__main__':
    main()
