import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from sb3_contrib.common.wrappers import ActionMasker

from ODT import LEOEnv as LEOEnvODT
from ODT import OnlineDecisionTransformer, predict_valid_action_dt
from HandoverEnvironment import mask_fn
from LEOEnvironmentRL import load_route_from_csv


SCENARIOS = [
    "load_cycle_1",
    "load_cycle_2",
    "load_cycle_5",
    "medium_aircraft",
    "large_aircraft",
    "snr_congested",
]


def _discounted_return(rewards, gamma=0.99):
    ret = 0.0
    for r in rewards[::-1]:
        ret = float(r) + gamma * ret
    return ret


def _calibrated_percentile_targets(dataset_path, percentiles=(50, 75, 90), gamma=0.99):
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)
    if not trajectories:
        raise RuntimeError("Offline dataset has no trajectories.")

    all_rewards = np.concatenate([traj["rewards"] for traj in trajectories])
    reward_mean = float(np.mean(all_rewards))
    reward_std = float(np.std(all_rewards)) if np.std(all_rewards) > 0 else 1.0

    norm_returns = []
    for traj in trajectories:
        rewards = (np.asarray(traj["rewards"], dtype=np.float32) - reward_mean) / reward_std
        rewards = np.clip(rewards, -5.0, 5.0)
        norm_returns.append(_discounted_return(rewards, gamma=gamma))

    targets = []
    for p in percentiles:
        t = float(np.percentile(norm_returns, p))
        targets.append(float(np.clip(t, -10.0, 10.0)))
    return targets


def _build_agent_from_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dim = checkpoint["model_state_dict"]["state_embedding.weight"].shape[1]
    action_dim = checkpoint["model_state_dict"]["action_embedding.weight"].shape[0]
    embed_dim = checkpoint["model_state_dict"]["state_embedding.weight"].shape[0]
    pos_len = checkpoint["model_state_dict"]["pos_embedding"].shape[1]
    max_length = pos_len // 3

    block_indices = []
    for key in checkpoint["model_state_dict"].keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
    num_layers = max(block_indices) + 1 if block_indices else 1

    agent = OnlineDecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_length=max_length,
        embed_dim=embed_dim,
        num_layers=num_layers,
        target_return=1.0,
        buffer_size=150,
        device="cpu",
    )
    agent.load(model_path)
    agent.set_eval_mode(True)
    return agent, state_dim, action_dim


def _evaluate_target(constellation_name, route, model_path, target_return, scenarios):
    total_rewards = []
    mean_alloc_ratio = []
    total_service_drop = []
    total_handovers = []

    for scenario in scenarios:
        env = LEOEnvODT(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        env.env.earth.Training = False

        agent, state_dim, action_dim = _build_agent_from_checkpoint(model_path)
        agent.target_return = float(target_return)
        agent.recent_episode_returns.clear()
        agent.recent_step_rewards.clear()

        obs, info = env.reset()
        done = False
        truncated = False

        ep_reward = 0.0
        ratios = []
        drops = []
        handovers = []

        while not (done or truncated):
            mask = env.env._get_action_mask()[:action_dim]
            action = predict_valid_action_dt(agent, obs[:state_dim], mask)
            next_obs, reward, done, truncated, info = env.step(action)

            # Keep trajectory memory for adaptive RTG behavior.
            agent.step(obs[:state_dim], action, reward, next_obs[:state_dim], done or truncated)

            ep_reward += float(reward)
            ratios.append(float(next_obs[7]))
            drops.append(float(next_obs[15]))
            handovers.append(float(next_obs[5]))

            obs = next_obs

        total_rewards.append(ep_reward)
        mean_alloc_ratio.append(float(np.mean(ratios)) if ratios else 0.0)
        total_service_drop.append(float(np.sum(drops)) if drops else 0.0)
        total_handovers.append(float(max(handovers)) if handovers else 0.0)

    return {
        "target_return": float(target_return),
        "avg_episode_reward": float(np.mean(total_rewards)),
        "avg_allocation_ratio": float(np.mean(mean_alloc_ratio)),
        "avg_total_service_drop_s": float(np.mean(total_service_drop)),
        "avg_total_handovers": float(np.mean(total_handovers)),
    }


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    base_dir = os.path.dirname(__file__)
    input_params = pd.read_csv(os.path.join(base_dir, "input.csv"))
    constellation_name = input_params["Constellation"][0]
    route, _ = load_route_from_csv(os.path.join(base_dir, "route_5s_interpolated.csv"), skip_rows=0)

    dataset_path = os.path.join(base_dir, "odt_offline_dataset.pkl")
    model_path = os.path.join(base_dir, "decision_transformer_offline.pth")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    targets = _calibrated_percentile_targets(dataset_path, percentiles=(50, 75, 90))
    print(f"RTG candidates (p50/p75/p90): {targets}")

    rows = []
    for t in targets:
        print(f"Evaluating target_return={t:.4f} ...")
        rows.append(_evaluate_target(constellation_name, route, model_path, t, SCENARIOS))

    results = pd.DataFrame(rows).sort_values("avg_episode_reward", ascending=False)
    print("\nRTG sweep results (sorted by avg_episode_reward):")
    print(results.to_string(index=False))

    best = results.iloc[0]
    print(
        f"\nBest target_return = {best['target_return']:.4f} "
        f"(avg_episode_reward={best['avg_episode_reward']:.4f})"
    )


if __name__ == "__main__":
    main()
