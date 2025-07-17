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

    def update(self, episodes=100, start_episode=1):
        total_rewards = []
        for ep in range(start_episode, start_episode + episodes):
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
            torch.save(self.policy.state_dict(), f"checkpoint_ep{ep}.pt")
            with open(f"training_log_{self.env.schema_name}.csv", mode="a", newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow([ep, total_reward, policy_loss.item(), value_loss.item()])
            print(f"Episode {ep} | Reward: {total_reward:.2f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")
            total_rewards.append(total_reward)
        plt.plot(total_rewards)
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid()
        plt.savefig("training_curve.png")
        print("\u2705 Training complete. Curve saved as 'training_curve.png'.")

# --------------------- LOOP MULTI-SCENARIOS ---------------------
def train_on_scenarios(policy_path, scenario_folder, episodes=110):
    scenarios = [d for d in os.listdir(scenario_folder) if os.path.isdir(os.path.join(scenario_folder, d))]
    if not scenarios:
        print("No scenarios found.")
        return

    first_schema = os.path.join(scenario_folder, scenarios[0], "schema.json")
    graph = build_graph(first_schema)
    env = CityLearnEnv(schema=first_schema)
    agent = GNNLSTMAgent(env=env, edge_index=graph.edge_index, edge_attr=graph.edge_attr)

    if os.path.exists(policy_path):
        agent.policy.load_state_dict(torch.load(policy_path))
        print(f"\n✅ Loaded checkpoint from {policy_path}")

    for scenario in scenarios:
        schema_path = os.path.join(scenario_folder, scenario, "schema.json")
        env = CityLearnEnv(schema=schema_path)
        graph = build_graph(schema_path)
        agent.env = env
        agent.edge_index = graph.edge_index
        agent.edge_attr = graph.edge_attr
        agent.fit_scalers()
        print(f"\n🔁 Training on scenario {scenario}")
        agent.update(episodes=episodes, start_episode=1)

# --------------------- MAIN ---------------------
if __name__ == "__main__":
    train_on_scenarios(
        policy_path="checkpoint_ep80.pt",
        scenario_folder="data/others",
        episodes=110
    )
