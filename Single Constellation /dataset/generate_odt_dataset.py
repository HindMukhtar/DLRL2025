import os
import sys
import contextlib
import io
import pickle
import random
import json

import numpy as np
import pandas as pd
import torch
import math
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from Training.HandoverEnvironment import LEOEnv as PPOEnv, predict_valid_action
from Environment.LEOEnvironmentRL import load_route_from_csv

DATASET_DIRNAME = "dataset"
MODELS_DIRNAME = "models"


def mask_fn(env):
    return env.action_mask


def _resolve_model_path(base_dir, name):
    zip_path = os.path.join(base_dir, MODELS_DIRNAME, f"{name}.zip")
    if os.path.exists(zip_path):
        return zip_path
    zip_fallback = os.path.join(base_dir, f"{name}.zip")
    if os.path.exists(zip_fallback):
        return zip_fallback
    model_path = os.path.join(base_dir, MODELS_DIRNAME, name)
    if os.path.exists(model_path):
        return model_path
    return os.path.join(base_dir, name)

def _collect_trajectories(
    env,
    predict_fn,
    model,
    episodes,
    quiet=True,
    scenario=None,
    source=None,
    seed_base=42,
    episode_seeds=None,
):
    trajectories = []
    for ep in range(episodes):
        print(f"Episode {ep+1}/{episodes}")
        sink = io.StringIO() if quiet else None
        cm = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
        with cm:
            if episode_seeds is not None and ep < len(episode_seeds):
                episode_seed = int(episode_seeds[ep])
            else:
                episode_seed = int(seed_base) + int(ep)
            obs, info = env.reset(seed=episode_seed)
            done = False
            truncated = False

            states = []
            actions = []
            rewards = []

            while not (done or truncated):
                mask = env.env._get_action_mask()
                if not np.any(mask):
                    action = 0
                elif model is None:
                    action = predict_fn(env)
                else:
                    action = predict_fn(model, obs, mask)
                next_obs, reward, done, truncated, info = env.step(action)
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                obs = next_obs

            if states:
                trajectories.append(
                    {
                        "states": np.array(states, dtype=np.float32),
                        "actions": np.array(actions, dtype=np.int64),
                        "rewards": np.array(rewards, dtype=np.float32),
                        "scenario": scenario,
                        "source": source,
                        "seed": int(seed_base),
                        "episode_seed": int(episode_seed),
                    }
                )
    return trajectories

def _candidate_metrics(base_env, candidate):
    ac = base_env.aircraft
    qoe = base_env.last_qoe if getattr(base_env, "last_qoe", None) is not None else base_env._refresh_qoe_cache()

    demand_mb = qoe.get("demand_MB", getattr(ac, "demand", 0.0))
    latency_req_s = qoe.get("latency_req_s", 0.1)
    deltaT = getattr(ac, "deltaT", 1.0) or 1.0

    beam = candidate["beam"]
    snr = candidate["snr"]

    # Throughput model aligned with env: min(shannon, demand, max_ds_speed).
    shannon_capacity_mbps = beam.effective_bw * math.log2(1 + 10 ** (snr / 10)) / 1e6
    demand_mbps = (demand_mb * 8.0) / max(deltaT, 1e-9)
    served_mbps = min(shannon_capacity_mbps, beam.max_ds_speed, demand_mbps)
    allocated_mb = (served_mbps * deltaT) / 8.0

    if demand_mb > 0:
        throughput_satisfaction = allocated_mb / demand_mb
    else:
        throughput_satisfaction = 1.0

    # Queue delay estimate aligned with env implementation.
    if served_mbps <= 0.0:
        queue_delay_s = 1000.0
    else:
        demand_Mb = demand_mb * 8.0
        service_Mb = served_mbps * deltaT
        if demand_Mb <= service_Mb:
            queue_delay_s = 0.0
        else:
            queue_delay_s = (demand_Mb - service_Mb) / served_mbps

    # Propagation + fixed processing (deterministic part; skip random jitter for action ranking).
    sat = candidate["sat"]
    aircraft_to_sat_m = ac._calculate_3d_distance(sat)
    sat_to_gateway_m = ac._calculate_3d_distance_to_gateway(sat)
    c = 299792458.0
    propagation_latency_s = (aircraft_to_sat_m + sat_to_gateway_m) * 2.0 / c + ac.fixed_processing_latency_s

    total_latency_s = queue_delay_s + propagation_latency_s
    if latency_req_s > 0:
        over = max(0.0, total_latency_s - latency_req_s)
        latency_satisfaction = math.exp(-over / latency_req_s)
    else:
        latency_satisfaction = 1.0

    service_drop_s = deltaT if (demand_mb > 0 and allocated_mb <= 0) else 0.0
    drop_risk = 1.0 if service_drop_s > 0 else 0.0
    return {
        "throughput_satisfaction": throughput_satisfaction,
        "latency_satisfaction": latency_satisfaction,
        "total_latency_s": total_latency_s,
        "service_drop_s": service_drop_s,
        "drop_risk": drop_risk,
        "allocated_mb": allocated_mb,
        "demand_mb": demand_mb,
    }


def _oracle_action(env):
    base_env = env.env
    mask = base_env._get_action_mask()
    candidates = base_env.current_beam_candidates

    # Match environment reward weights.
    w_thr = 0.7
    w_lat = 0.3
    w_drop = 0.5

    best_action = -1
    best_reward = -float("inf")

    for i, candidate in enumerate(candidates):
        if not mask[i] or candidate is None:
            continue
        metrics = _candidate_metrics(base_env, candidate)

        reward = (
            w_thr * metrics["throughput_satisfaction"]
            + w_lat * metrics["latency_satisfaction"]
            - w_drop * metrics["service_drop_s"]
        )

        if reward > best_reward:
            best_reward = reward
            best_action = i

    return best_action


def _oracle_latency_action(env):
    base_env = env.env
    mask = base_env._get_action_mask()
    candidates = base_env.current_beam_candidates

    best_action = -1
    best_latency = float("inf")
    best_drop = float("inf")
    for i, candidate in enumerate(candidates):
        if not mask[i] or candidate is None:
            continue
        metrics = _candidate_metrics(base_env, candidate)
        latency = metrics["total_latency_s"]
        drop = metrics["service_drop_s"]
        if (latency < best_latency) or (latency == best_latency and drop < best_drop):
            best_latency = latency
            best_drop = drop
            best_action = i
    return best_action


def _oracle_drop_action(env):
    base_env = env.env
    mask = base_env._get_action_mask()
    candidates = base_env.current_beam_candidates

    best_action = -1
    best_key = None
    for i, candidate in enumerate(candidates):
        if not mask[i] or candidate is None:
            continue
        metrics = _candidate_metrics(base_env, candidate)
        # Priority:
        # 1) avoid drop risk
        # 2) maximize throughput satisfaction
        # 3) lower latency
        key = (
            metrics["drop_risk"],
            -metrics["throughput_satisfaction"],
            metrics["total_latency_s"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_action = i
    return best_action


def main():
    base_dir = PROJECT_DIR
    dataset_dir = SCRIPT_DIR
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, MODELS_DIRNAME), exist_ok=True)
    aug_dataset_path = os.path.join(dataset_dir, "odt_offline_dataset_aug_multisim_v2.pkl")
    input_path = os.path.join(base_dir, "Inputs", "input.csv")
    if not os.path.exists(input_path):
        input_path = os.path.join(base_dir, "input.csv")
    input_params = pd.read_csv(input_path)
    constellation_name = input_params["Constellation"][0]
    route_path = os.path.join(base_dir, "routes", "route_5s_interpolated.csv")
    if not os.path.exists(route_path):
        route_path = os.path.join(base_dir, "route_5s_interpolated.csv")
    route, _ = load_route_from_csv(route_path, skip_rows=0)
    scenarios = [
        "load_cycle_1",
        "load_cycle_2",
        "load_cycle_5",
        "large_aircraft",
        "medium_aircraft",
        "snr_congested",
    ]
    dataset_seeds = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

    # Per-scenario episode budgets by source (equal samples per active agent).
    # 4 active sources x 10 episodes = 40 episodes per scenario per seed.
    scenario_episode_budget = {
        "load_cycle_1": {
            "ppo": 10,
            "oracle_latency": 10,
            "oracle_drop": 10,
            "oracle": 10,
        },
        "snr_congested": {
            "ppo": 10,
            "oracle": 10,
            "oracle_drop": 10,
            "oracle_latency": 10,
        },
        "load_cycle_2": {
            "ppo": 10,
            "oracle": 10,
            "oracle_latency": 10,
            "oracle_drop": 10,
        },
        "load_cycle_5": {
            "ppo": 10,
            "oracle": 10,
            "oracle_latency": 10,
            "oracle_drop": 10,
        },
        "medium_aircraft": {
            "ppo": 10,
            "oracle": 10,
            "oracle_latency": 10,
            "oracle_drop": 10,
        },
        "large_aircraft": {
            "ppo": 10,
            "oracle": 10,
            "oracle_latency": 10,
            "oracle_drop": 10,
        },
    }
    trajectories = []

    ppo_model_path = _resolve_model_path(base_dir, "handover_ppo_agent")
    ppo_model = MaskablePPO.load(ppo_model_path, device="cpu")

    for seed in dataset_seeds:
        print(f"\n========== DATASET SEED: {seed} ==========")
        os.environ["EVAL_SEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        for scenario in scenarios:
            print(f"Collecting trajectories for scenario: {scenario}")
            episode_budget = scenario_episode_budget.get(scenario, {})
            print(f"Episode budget: {episode_budget}")

            if episode_budget.get("ppo", 0) > 0:
                print("PPO Agent Trajectories...")
                ppo_env = PPOEnv(constellation_name, route, scenario=scenario)
                ppo_env = ActionMasker(ppo_env, mask_fn)
                ppo_env.env.earth.Training = True
                trajectories.extend(
                    _collect_trajectories(
                        ppo_env,
                        predict_valid_action,
                        ppo_model,
                        episode_budget.get("ppo", 0),
                        quiet=True,
                        scenario=scenario,
                        source="ppo",
                        seed_base=seed * 10000,
                    )
                )

            if episode_budget.get("oracle", 0) > 0:
                print("Oracle Agent Trajectories...")
                oracle_env = PPOEnv(constellation_name, route, scenario=scenario)
                oracle_env = ActionMasker(oracle_env, mask_fn)
                oracle_env.env.earth.Training = True
                trajectories.extend(
                    _collect_trajectories(
                        oracle_env,
                        _oracle_action,
                        None,
                        episode_budget.get("oracle", 0),
                        quiet=True,
                        scenario=scenario,
                        source="oracle",
                        seed_base=seed * 10000,
                    )
                )

            if episode_budget.get("oracle_latency", 0) > 0:
                print("Oracle Latency Trajectories...")
                oracle_latency_env = PPOEnv(constellation_name, route, scenario=scenario)
                oracle_latency_env = ActionMasker(oracle_latency_env, mask_fn)
                oracle_latency_env.env.earth.Training = True
                trajectories.extend(
                    _collect_trajectories(
                        oracle_latency_env,
                        _oracle_latency_action,
                        None,
                        episode_budget.get("oracle_latency", 0),
                        quiet=True,
                        scenario=scenario,
                        source="oracle_latency",
                        seed_base=seed * 10000,
                    )
                )

            if episode_budget.get("oracle_drop", 0) > 0:
                print("Oracle Drop Trajectories...")
                oracle_drop_env = PPOEnv(constellation_name, route, scenario=scenario)
                oracle_drop_env = ActionMasker(oracle_drop_env, mask_fn)
                oracle_drop_env.env.earth.Training = True
                trajectories.extend(
                    _collect_trajectories(
                        oracle_drop_env,
                        _oracle_drop_action,
                        None,
                        episode_budget.get("oracle_drop", 0),
                        quiet=True,
                        scenario=scenario,
                        source="oracle_drop",
                        seed_base=seed * 10000,
                    )
                )

    with open(aug_dataset_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved augmentation dataset with {len(trajectories)} trajectories to {aug_dataset_path}")
    split_meta_path = os.path.join(dataset_dir, "odt_dataset_seed_split_multisim_v2.json")
    with open(split_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_seeds": dataset_seeds,
                "scenarios": scenarios,
            },
            f,
            indent=2,
        )
    print(f"Saved dataset seed metadata to {split_meta_path}")


if __name__ == "__main__":
    main()
