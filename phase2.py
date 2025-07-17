import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from citylearn.citylearn import CityLearnEnv
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

# --------------------- GRAPH CONSTRUCTION ---------------------
def build_graph(schema_path, env_steps=24, k=2):
    with open(schema_path) as f:
        schema = json.load(f)
    buildings = schema["buildings"]
    static_features = []
    for b_data in buildings.values():
        battery = b_data.get("electrical_storage", {}).get("attributes", {})
        pv = b_data.get("pv", {}).get("attributes", {})
        static_feat = [
            battery.get("capacity", 0.0),
            battery.get("nominal_power", 0.0),
            battery.get("efficiency", 0.0),
            pv.get("nominal_power", 0.0)
        ]
        static_features.append(static_feat)
    env = CityLearnEnv(schema=schema_path)
    obs = env.reset()
    n_buildings = len(env.buildings)
    energy_demands = np.zeros(n_buildings)
    solar_production = np.zeros(n_buildings)
    feature_names = env.observation_names[0]
    energy_idx = feature_names.index('net_electricity_consumption')
    solar_idx = feature_names.index('solar_generation')
    for _ in range(env_steps):
        actions = [[0.0] * len(a.high) for a in env.action_space]
        obs, _, done, _ = env.step(actions)
        for i, ob in enumerate(obs):
            energy_demands[i] += ob[energy_idx]
            solar_production[i] += ob[solar_idx]
        if done:
            break
    env.close()
    energy_demands /= env_steps
    solar_production /= env_steps
    static_features = np.array(static_features)
    combined_features = np.hstack([
        static_features,
        energy_demands.reshape(-1, 1),
        solar_production.reshape(-1, 1)
    ])
    combined_features = StandardScaler().fit_transform(combined_features)
    dists = euclidean_distances(combined_features)
    edge_index = []
    edge_attr = []
    for i in range(len(combined_features)):
        nearest = np.argsort(dists[i])[1:k+1]
        for j in nearest:
            weight = 1.0 / (dists[i][j] + 1e-6)
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([[weight], [weight]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    return Data(edge_index=edge_index, edge_attr=edge_attr)

# --------------------- POLICY ---------------------
class GNNLSTMPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1)
        self.actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, act_dim))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x_seq, edge_index, edge_attr):
        lstm_out, _ = self.lstm(x_seq)
        final = lstm_out[:, -1, :]
        x = torch.relu(self.conv1(final, edge_index, edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr)) + x
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        value = self.critic(x).squeeze(-1)
        return dist, value

# --------------------- AGENT ---------------------
class GNNLSTMAgent:
    def __init__(self, env, edge_index, edge_attr, hidden_dim=128, seq_len=6, lr=3e-4, gamma=0.99):
        self.env = env
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.seq_len = seq_len
        self.gamma = gamma
        self.device = torch.device("cpu")
        obs = env.reset()
        encoded = self.encode_time_features(obs)
        self.obs_dim = encoded.shape[-1]
        self.act_dim = len(env.action_space[0].high)
        self.policy = GNNLSTMPolicy(self.obs_dim, hidden_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scalers = [StandardScaler() for _ in range(len(env.buildings))]
        self.fit_scalers()
        checkpoint_path = "checkpoint_ep80.pt"
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path)
            self.policy.load_state_dict(state_dict, strict=False)
            print("\u2705 Loaded weights from Phase 1: checkpoint_ep80.pt")

    def encode_time_features(self, raw_obs):
        month_idx = self.env.observation_names[0].index("month")
        hour_idx = self.env.observation_names[0].index("hour")
        day_type_idx = self.env.observation_names[0].index("day_type")
        encoded_obs = []
        for ob in raw_obs:
            ob = np.array(ob)
            month = ob[month_idx]
            hour = ob[hour_idx]
            day_type = ob[day_type_idx]
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            non_time = np.delete(ob, [month_idx, hour_idx, day_type_idx])
            full = np.concatenate([non_time, [month_sin, month_cos, hour_sin, hour_cos, day_type]])
            encoded_obs.append(full)
        return np.array(encoded_obs)

    def fit_scalers(self):
        observations = []
        obs = self.env.reset()
        done = False
        while not done:
            encoded = self.encode_time_features(obs)
            observations.append(encoded)
            actions = [[0.0] * self.act_dim for _ in self.env.action_space]
            obs, _, done, _ = self.env.step(actions)
        observations = np.array(observations)
        for i, scaler in enumerate(self.scalers):
            scaler.fit(observations[:, i, :])

    def encode_obs(self, obs):
        encoded = self.encode_time_features(obs)
        return np.array([self.scalers[i].transform(encoded[i].reshape(1, -1)).flatten() for i in range(len(encoded))])

    def update(self, episodes=100):
        total_rewards = []
        with open("training_log_phase2.csv", mode="a", newline="") as log_file:
            writer = csv.writer(log_file)
            if os.stat("training_log_phase2.csv").st_size == 0:
                writer.writerow(["Episode", "TotalReward", "PolicyLoss", "ValueLoss"])
            for ep in range(episodes):
                obs = self.env.reset()
                obs_buffer, log_probs, values, rewards = [], [], [], []
                done, total_reward = False, 0
                while not done:
                    obs_encoded = self.encode_obs(obs)
                    obs_buffer.append(obs_encoded)
                    if len(obs_buffer) < self.seq_len:
                        actions = [[0.0] * self.act_dim for _ in range(len(obs))]
                        obs, _, done, _ = self.env.step(actions)
                        continue
                    obs_seq = np.array(obs_buffer[-self.seq_len:])
                    obs_seq_tensor = torch.tensor(obs_seq, dtype=torch.float32).permute(1, 0, 2).to(self.device)
                    dist, value = self.policy(obs_seq_tensor, self.edge_index, self.edge_attr)
                    actions_tensor = dist.sample()
                    log_prob = dist.log_prob(actions_tensor).sum(dim=-1)
                    obs_np = actions_tensor.detach().cpu().numpy()
                    clipped = [np.clip(obs_np[i], self.env.action_space[i].low, self.env.action_space[i].high) for i in range(len(obs_np))]
                    obs, reward, done, _ = self.env.step(clipped)
                    reward = torch.tensor(reward, dtype=torch.float32)
                    reward = torch.clamp(reward, -100.0, 100.0)
                    total_reward += reward.sum().item()
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(reward)
                rewards = torch.stack(rewards)
                values = torch.stack(values)
                log_probs = torch.stack(log_probs)
                returns = torch.zeros_like(rewards)
                running_return = torch.zeros(rewards.shape[1])
                for t in reversed(range(len(rewards))):
                    running_return = rewards[t] + self.gamma * running_return
                    returns[t] = running_return
                returns = (returns - returns.mean()) / (returns.std() + 1e-6)
                advantages = returns - values.detach()
                policy_loss = -torch.mean(log_probs * advantages.detach())
                value_loss = torch.mean((returns - values) ** 2)
                loss = policy_loss + 0.5 * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (ep + 1) % 10 == 0:
                    torch.save(self.policy.state_dict(), f"checkpoint_phase2_ep{ep+1}.pt")
                writer.writerow([ep + 1, total_reward, policy_loss.item(), value_loss.item()])
                print(f"Episode {ep + 1} | Reward: {total_reward:.2f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")
                total_rewards.append(total_reward)
        plt.plot(total_rewards)
        plt.title("Phase 2 Training Curve")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid()
        plt.savefig("training_curve_phase2.png")
        print("\u2705 Training complete. Curve saved as 'training_curve_phase2.png'.")

# --------------------- MAIN ---------------------
if __name__ == "__main__":
    schema_path = "data/others/citylearn_challenge_2022_phase_2/schema.json"
    env = CityLearnEnv(schema=schema_path)
    graph = build_graph(schema_path)
    agent = GNNLSTMAgent(env=env, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
    agent.update(episodes=100)
