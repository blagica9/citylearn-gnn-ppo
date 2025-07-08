import os
import json
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from citylearn.citylearn import CityLearnEnv
import torch_geometric
from gnn_rlc import GNNRLC, GNNPolicy


def build_graph(schema_path, env_steps=24, k=2):
    """Гради граф врз основа на статични и просечни динамички карактеристики на зградите."""
    with open(schema_path) as f:
        schema = json.load(f)

    buildings = schema["buildings"]
    building_names = list(buildings.keys())
    static_features = []

    # Екстракција на статични карактеристики за секоја зграда
    for b in building_names:
        b_data = buildings[b]
        battery = b_data.get("electrical_storage", {}).get("attributes", {})
        pv = b_data.get("pv", {}).get("attributes", {})
        cs = b_data.get("cooling_storage", {})
        dhw = b_data.get("dhw_storage", {})
        hp = b_data.get("cooling_device", {}).get("attributes", {})

        static_feat = [
            battery.get("capacity", 0.0),
            battery.get("nominal_power", 0.0),
            battery.get("efficiency", 0.0),
            pv.get("nominal_power", 0.0),
            cs.get("autosize_attributes", {}).get("safety_factor", 0.0),
            cs.get("attributes", {}).get("loss_coefficient", 0.0),
            dhw.get("autosize_attributes", {}).get("safety_factor", 0.0),
            dhw.get("attributes", {}).get("loss_coefficient", 0.0),
            hp.get("efficiency", 0.0),
            hp.get("target_cooling_temperature", 0.0)
        ]
        static_features.append(static_feat)

    # Екстракција на динамички просечни вредности преку environment симулација
    env = CityLearnEnv(schema=schema_path)
    obs = env.reset()
    n_buildings = len(env.buildings)
    energy_demands = np.zeros(n_buildings)
    solar_production = np.zeros(n_buildings)

    try:
        feature_names = env.observation_names[0]
        energy_idx = feature_names.index('net_electricity_consumption')
        solar_idx = feature_names.index('solar_generation')
    except ValueError:
        print("⚠️ Observation does not include required features!")
        return None

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

    # Комбинација на статички + динамички фичери и изградба на граф
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
            edge_index.extend([[i, j], [j, i]])  # bidirectional
            edge_attr.extend([[weight], [weight]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return torch_geometric.data.Data(edge_index=edge_index, edge_attr=edge_attr)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scenario_folders = [
        os.path.join("data", d)
        for d in os.listdir("data")
        if os.path.isdir(os.path.join("data", d))
    ]
    schema_paths = [os.path.join(folder, "schema.json") for folder in scenario_folders]

    # Иницијално сценарио
    init_env = CityLearnEnv(schema=schema_paths[0])
    obs_dim = len(init_env.observation_space[0].low)
    act_dim = len(init_env.action_space[0].high)

    # Иницијален граф
    pyg_graph = build_graph(schema_paths[0])
    edge_index = pyg_graph.edge_index.to(device)
    edge_attr = pyg_graph.edge_attr.to(device)

    # Иницијализација на централен GNN PPO агент
    agent = GNNRLC(
        env=init_env,
        edge_index=edge_index,
        edge_attr=edge_attr,
        obs_dim=obs_dim,
        act_dim=act_dim
    )

    for schema_path in schema_paths:
        print(f"\n--- Training on scenario: {schema_path} ---")

        env = CityLearnEnv(schema=schema_path)
        pyg_graph = build_graph(schema_path)

        agent.env = env
        agent.edge_index = pyg_graph.edge_index.to(device)
        agent.edge_attr = pyg_graph.edge_attr.to(device)

        # Скалери за нови згради
        agent.scalers = [StandardScaler() for _ in range(len(env.buildings))]
        agent.fit_scalers()

        # Адаптација ако се сменети input/output димензии
        new_obs_dim = len(env.observation_space[0].low)
        new_act_dim = len(env.action_space[0].high)

        if new_obs_dim != agent.obs_dim or new_act_dim != agent.act_dim:
            print("🔄 Адаптација на политика поради различни димензии.")
            agent.obs_dim = new_obs_dim
            agent.act_dim = new_act_dim
            agent.policy = GNNPolicy(new_obs_dim, agent.hidden_dim, new_act_dim).to(agent.device)
            agent.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=agent.lr)

        agent.update(episodes=50)

    torch.save(agent.policy.state_dict(), "gnn_rlc_agent.pt")
    print("✅ Завршено. Агентот е зачуван во gnn_rlc_agent.pt")
