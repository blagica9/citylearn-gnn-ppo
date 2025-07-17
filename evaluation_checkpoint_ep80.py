import os
import torch
import numpy as np
from citylearn.citylearn import CityLearnEnv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.data import Data
from gnn_lstm_ppo import GNNLSTMAgent  # Осигурај се дека фајлот е во иста папка или во PYTHONPATH

# --------------------- GRAPH CONSTRUCTION ---------------------
def build_graph(schema_path, env_steps=24, k=2):
    import json
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

# --------------------- EVALUATION FUNCTION ---------------------
def evaluate_agent(agent, episodes=3):
    total_rewards = []
    for ep in range(episodes):
        obs = agent.env.reset()
        obs_buffer = []
        done, total_reward = False, 0
        while not done:
            obs_encoded = agent.encode_obs(obs)
            obs_buffer.append(obs_encoded)
            if len(obs_buffer) < agent.seq_len:
                actions = [[0.0] * agent.act_dim for _ in range(len(obs))]
                obs, _, done, _ = agent.env.step(actions)
                continue

            obs_seq = np.array(obs_buffer[-agent.seq_len:])
            obs_seq_tensor = torch.tensor(obs_seq, dtype=torch.float32).permute(1, 0, 2).to(agent.device)
            with torch.no_grad():
                dist, _ = agent.policy(obs_seq_tensor, agent.edge_index, agent.edge_attr)
                actions_tensor = dist.mean

            obs_np = actions_tensor.cpu().numpy()
            clipped = [np.clip(obs_np[i], agent.env.action_space[i].low, agent.env.action_space[i].high) for i in range(len(obs_np))]
            obs, reward, done, _ = agent.env.step(clipped)
            total_reward += sum(reward)
        print(f"Eval Episode {ep+1} | Total Reward: {total_reward:.2f}")
        total_rewards.append(total_reward)
    avg_reward = np.mean(total_rewards)
    print(f"\n\U00002705 Average Evaluation Reward over {episodes} episodes: {avg_reward:.2f}")

# --------------------- MAIN ---------------------
if __name__ == "__main__":
    schema_path = "data/others/citylearn_challenge_2022_phase_2/schema.json"  # ново сценарио
    graph = build_graph(schema_path)
    env = CityLearnEnv(schema=schema_path)
    agent = GNNLSTMAgent(env=env, edge_index=graph.edge_index, edge_attr=graph.edge_attr)

    # Вчитување checkpoint
    checkpoint_path = "checkpoint_ep80.pt"
    if os.path.exists(checkpoint_path):
        agent.policy.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
        agent.policy.eval()
        print(f"✅ Loaded agent from {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")

    evaluate_agent(agent, episodes=3)
