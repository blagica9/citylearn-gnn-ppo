import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch_geometric.nn import GCNConv, GATv2Conv
from citylearn.citylearn import CityLearnEnv
from sklearn.preprocessing import StandardScaler

class GNNPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super().__init__()
        self.conv1 = GATv2Conv(obs_dim, hidden_dim, edge_dim=1, dropout=0.1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, dropout=0.1)

        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x, edge_index, edge_attr):
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr))

        mu = self.mu_head(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        value = self.value_head(x).squeeze(-1)
        return dist, value

class GNNRLC:
    def __init__(self, env: CityLearnEnv, edge_index, edge_attr, obs_dim, act_dim,
                 hidden_dim=256, lr=3e-4, discount=0.99, reward_scaling=1.0):
        self.env = env
        self.edge_index = edge_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.discount = discount
        self.reward_scaling = reward_scaling
        self.edge_attr = edge_attr

        self.policy = GNNPolicy(obs_dim, hidden_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Креираме StandardScaler за секој building
        self.scalers = [StandardScaler() for _ in range(len(env.buildings))]
        self.fit_scalers()

    def fit_scalers(self):
        observations = []
        obs = self.env.reset()
        done = False

        while not done:
            observations.append(obs)
            # Одиме со акции 0 за да земеме набљудувања
            actions = [[0.0] * len(a.high) for a in self.env.action_space]
            _, _, done, _ = self.env.step(actions)

        observations = np.array(observations)  # (steps, buildings, features)

        for i, scaler in enumerate(self.scalers):
            data = observations[:, i, :]  # сите набљудувања за building i
            scaler.fit(data)

    def encode_obs(self, obs):
        encoded = []

        for i, scaler in enumerate(self.scalers):
            data = np.array(obs[i]).reshape(1, -1)  # (1, features)
            normalized = scaler.transform(data).flatten()
            encoded.append(normalized)

        encoded = np.array(encoded)  # (buildings, features)
        return torch.tensor(encoded, dtype=torch.float32).to(self.device)

    def update(self, episodes=10, clip_eps=0.2):
        for ep in range(episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            log_probs, values, rewards, entropies = [], [], [], []

            while not done:
                obs_tensor = self.encode_obs(obs)
                dist, value = self.policy(obs_tensor, self.edge_index, self.edge_attr)
                actions_tensor = dist.sample()
                log_prob = dist.log_prob(actions_tensor).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                actions = actions_tensor.cpu().detach().numpy()
                next_obs, reward, done, _ = self.env.step(actions)

                reward = [r * self.reward_scaling for r in reward]

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor(reward, dtype=torch.float32).to(self.device))
                entropies.append(entropy)

                obs = next_obs
                total_reward += sum(reward)

            rewards = torch.stack(rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            values = torch.stack(values)
            log_probs = torch.stack(log_probs)
            entropies = torch.stack(entropies)

            returns = torch.zeros_like(rewards)
            running_return = torch.zeros(rewards.shape[1]).to(self.device)
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + self.discount * running_return
                returns[t] = running_return

            advantages = returns - values.detach()
            ratio = torch.exp(log_probs - log_probs.detach())
            clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = (returns - values).pow(2).mean()
            entropy_bonus = entropies.mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Episode {ep+1} | Total reward: {total_reward:.2f} | Policy loss: {policy_loss.item():.4f} | Value loss: {value_loss.item():.4f} | Entropy: {entropy_bonus.item():.4f}")
