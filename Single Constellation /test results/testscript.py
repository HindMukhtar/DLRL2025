import sys
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR
if not os.path.isdir(os.path.join(PROJECT_DIR, "Environment")):
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from Environment.LEOEnvironmentRL import initialize, load_route_from_csv  # Use RL version
import pandas as pd
from stable_baselines3 import DQN
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import torch
import random
import sb3_contrib
from Training.HandoverEnvironment import LEOEnv as LEOEnvPPO
from Training.HandoverEnvironment import mask_fn, predict_valid_action
from Training.HandoverEnvironment_DQN import LEOEnv as LEOEnvDQN
from Training.HandoverEnvironment_DQN import predict_valid_action as predict_valid_action_dqn
#from HandoverEnvironment_ODT import LEOEnv as LEOEnvODT
#from HandoverEnvironment_ODT import predict_valid_action_dt
from Training.ODT import LEOEnv as LEOEnvODT
from Training.ODT import predict_valid_action_dt
from Training.ODT import OnlineDecisionTransformer
import Environment.LEOEnvironment as LEOEnvModule
from Environment.LEOEnvironment import LEOEnv as LEOEnvBase
import pickle
import gc  # Add garbage collection
import math

DATASET_DIRNAME = "dataset"
MODELS_DIRNAME = "models"
TEST_RESULTS_DIRNAME = "test results"


def _ensure_dirs(base_dir_path):
    dataset_dir = os.path.join(base_dir_path, DATASET_DIRNAME)
    models_dir = os.path.join(base_dir_path, MODELS_DIRNAME)
    results_dir = os.path.join(base_dir_path, TEST_RESULTS_DIRNAME)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return dataset_dir, models_dir, results_dir


def _resolve_file(base_dir_path, filename, folder_name=None):
    candidates = []
    if folder_name is not None:
        candidates.append(os.path.join(base_dir_path, folder_name, filename))
    candidates.append(os.path.join(base_dir_path, filename))
    if not filename.endswith(".zip"):
        if folder_name is not None:
            candidates.append(os.path.join(base_dir_path, folder_name, f"{filename}.zip"))
        candidates.append(os.path.join(base_dir_path, f"{filename}.zip"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]

# ==============================================
# MODEL SELECTION - Choose which model to test
# ==============================================
# Options: 'ODT', 'DQN', 'PPO', 'BASELINE', 'ODT_FINETUNED', 'ORACLE', 'ORACLE_LATENCY', 'ORACLE_DROP', 'BASELINE_LEGACY'
SELECTED_MODEL = os.getenv("SELECTED_MODEL", "ODT")  # Default: ODT
EVAL_SEED = int(os.getenv("EVAL_SEED", "42"))
RESULTS_TAG = os.getenv("RESULTS_TAG", "").strip()

def append_observation_to_file(obs, step, model_name, filename):
    """Append single observation to file"""
    # Create header if file doesn't exist
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("step, lat, lon, alt, snr, load, handovers, allocated_bw, allocation_ratio, demand_MB, throughput_req, queing_delay_s, propagation_latency_s, transmission_rate_mbps, latency_req_s, beam_capacity, service_drop_s, dwell_remaining_s, ttt_remaining_s\n")
    
    # Append observation data
    with open(filename, 'a') as f:
        
        f.write(f"{step},{obs[0]},{obs[1]},{obs[2]},{obs[3]},{obs[4]},{obs[5]},{obs[6]},{obs[7]},{obs[8]},{obs[9]},{obs[10]},{obs[11]},{obs[12]},{obs[13]},{obs[14]},{obs[15]},{obs[16]},{obs[17]}\n")
    
    # Periodic garbage collection every 100 steps to manage memory
    if step % 100 == 0:
        gc.collect()

print(f"Selected Model for Testing: {SELECTED_MODEL}")
print("=" * 50)

# Freeze evaluation randomness for reproducible comparisons.
np.random.seed(EVAL_SEED)
random.seed(EVAL_SEED)
torch.manual_seed(EVAL_SEED)

# Initialize common parameters
base_dir = PROJECT_DIR
dataset_dir, models_dir, test_results_dir = _ensure_dirs(base_dir)
input_path = os.path.join(base_dir, "Inputs", "input.csv")
if not os.path.exists(input_path):
    input_path = os.path.join(base_dir, "input.csv")
inputParams = pd.read_csv(input_path)
constellation_name = inputParams['Constellation'][0]
route_path = os.path.join(base_dir, "routes", "route_5s_interpolated.csv")
if not os.path.exists(route_path):
    route_path = os.path.join(base_dir, "route_5s_interpolated.csv")
route, route_duration = load_route_from_csv(route_path, skip_rows=0)

SCENARIOS = [
    "load_cycle_1"
    #"load_cycle_2",
    #"load_cycle_5",
    #"medium_aircraft",
    #"large_aircraft",
    #"snr_congested",
]
scenarios_env = os.getenv("SCENARIOS")
if scenarios_env:
    parsed = [s.strip() for s in scenarios_env.split(",") if s.strip()]
    if parsed:
        SCENARIOS = parsed
ODT_MODELS = [
    #("ODT", "decision_transformer_offline_best.pth"),
    ("ODT_FINETUNED", "decision_transformer_online_finetune_best_20.pth"),
]


def _calibrate_rtg_from_dataset(base_dir_path, fallback_value):
    dataset_path = _resolve_file(base_dir_path, "odt_offline_dataset.pkl", DATASET_DIRNAME)
    if not os.path.exists(dataset_path):
        return float(fallback_value)
    try:
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)
        if not trajectories:
            return float(fallback_value)
        all_rewards = np.concatenate([traj["rewards"] for traj in trajectories])
        reward_mean = float(np.mean(all_rewards))
        reward_std = float(np.std(all_rewards)) if np.std(all_rewards) > 0 else 1.0
        returns = []
        for traj in trajectories:
            rewards = (np.asarray(traj["rewards"], dtype=np.float32) - reward_mean) / reward_std
            rewards = np.clip(rewards, -5.0, 5.0)
            ret = 0.0
            for r in rewards[::-1]:
                ret = float(r) + 0.99 * ret
            returns.append(ret)
        if not returns:
            return float(fallback_value)
        return float(np.clip(np.percentile(returns, 75), -10.0, 10.0))
    except Exception:
        return float(fallback_value)


def predict_valid_action_oracle(base_env, obs, mask):
    """
    Oracle-like one-step action selector using immediate reward proxy.
    """
    if not np.any(mask):
        return -1

    candidates = getattr(base_env.aircraft, "current_steering_candidates", [])
    ac = base_env.aircraft
    qoe = base_env.last_qoe if getattr(base_env, "last_qoe", None) is not None else base_env._refresh_qoe_cache()

    demand_mb = qoe.get("demand_MB", getattr(ac, "demand", 0.0))
    latency_req_s = qoe.get("latency_req_s", 0.1)
    deltaT = getattr(ac, "deltaT", 1.0) or 1.0

    w_thr = 0.7
    w_lat = 0.3
    w_drop = 0.5

    best_action = -1
    best_reward = -float("inf")

    for i, cand in enumerate(candidates):
        if i >= len(mask) or i >= len(candidates) or not mask[i] or cand is None:
            continue

        beam = cand["beam"]
        snr = cand["snr"]

        shannon_capacity_mbps = beam.effective_bw * math.log2(1 + 10 ** (snr / 10)) / 1e6
        demand_mbps = (demand_mb * 8.0) / max(deltaT, 1e-9)
        served_mbps = min(shannon_capacity_mbps, beam.max_ds_speed, demand_mbps)
        allocated_mb = (served_mbps * deltaT) / 8.0

        throughput_satisfaction = (allocated_mb / demand_mb) if demand_mb > 0 else 1.0

        if served_mbps <= 0.0:
            queue_delay_s = 1000.0
        else:
            demand_Mb = demand_mb * 8.0
            service_Mb = served_mbps * deltaT
            queue_delay_s = 0.0 if demand_Mb <= service_Mb else (demand_Mb - service_Mb) / served_mbps

        sat = cand["sat"]
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

        reward = (
            w_thr * throughput_satisfaction
            + w_lat * latency_satisfaction
            - w_drop * service_drop_s
        )

        if reward > best_reward:
            best_reward = reward
            best_action = i

    return best_action


def _oracle_candidate_metrics(base_env, cand):
    ac = base_env.aircraft
    qoe = base_env.last_qoe if getattr(base_env, "last_qoe", None) is not None else base_env._refresh_qoe_cache()

    demand_mb = qoe.get("demand_MB", getattr(ac, "demand", 0.0))
    latency_req_s = qoe.get("latency_req_s", 0.1)
    deltaT = getattr(ac, "deltaT", 1.0) or 1.0

    beam = cand["beam"]
    snr = cand["snr"]

    shannon_capacity_mbps = beam.effective_bw * math.log2(1 + 10 ** (snr / 10)) / 1e6
    demand_mbps = (demand_mb * 8.0) / max(deltaT, 1e-9)
    served_mbps = min(shannon_capacity_mbps, beam.max_ds_speed, demand_mbps)
    allocated_mb = (served_mbps * deltaT) / 8.0

    if served_mbps <= 0.0:
        queue_delay_s = 1000.0
    else:
        demand_Mb = demand_mb * 8.0
        service_Mb = served_mbps * deltaT
        queue_delay_s = 0.0 if demand_Mb <= service_Mb else (demand_Mb - service_Mb) / served_mbps

    sat = cand["sat"]
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

    throughput_satisfaction = (allocated_mb / demand_mb) if demand_mb > 0 else 1.0
    service_drop_s = deltaT if (demand_mb > 0 and allocated_mb <= 0) else 0.0

    return {
        "total_latency_s": total_latency_s,
        "latency_satisfaction": latency_satisfaction,
        "throughput_satisfaction": throughput_satisfaction,
        "service_drop_s": service_drop_s,
        "drop_risk": 1.0 if service_drop_s > 0 else 0.0,
    }


def predict_valid_action_oracle_latency(base_env, obs, mask):
    if not np.any(mask):
        return -1
    candidates = getattr(base_env.aircraft, "current_steering_candidates", [])
    best_action = -1
    best_latency = float("inf")
    best_drop = float("inf")
    for i, cand in enumerate(candidates):
        if i >= len(mask) or i >= len(candidates) or not mask[i] or cand is None:
            continue
        m = _oracle_candidate_metrics(base_env, cand)
        if (m["total_latency_s"] < best_latency) or (
            m["total_latency_s"] == best_latency and m["service_drop_s"] < best_drop
        ):
            best_latency = m["total_latency_s"]
            best_drop = m["service_drop_s"]
            best_action = i
    return best_action


def predict_valid_action_oracle_drop(base_env, obs, mask):
    if not np.any(mask):
        return -1
    candidates = getattr(base_env.aircraft, "current_steering_candidates", [])
    best_action = -1
    best_key = None
    for i, cand in enumerate(candidates):
        if i >= len(mask) or i >= len(candidates) or not mask[i] or cand is None:
            continue
        m = _oracle_candidate_metrics(base_env, cand)
        key = (
            m["drop_risk"],
            -m["throughput_satisfaction"],
            m["total_latency_s"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_action = i
    return best_action


def predict_valid_action_baseline_rl(agent, obs, mask, base_env=None):
    """
    Baseline policy on RL wrapper: choose highest-SNR valid candidate.
    """
    if not np.any(mask):
        return -1
    if base_env is None:
        # Fallback: first valid
        return int(np.where(mask)[0][0])

    candidates = getattr(base_env.aircraft, "current_steering_candidates", [])
    best_idx = -1
    best_snr = -float("inf")
    for i, cand in enumerate(candidates):
        if i >= len(mask) or i >= len(candidates) or not mask[i] or cand is None:
            continue
        snr = float(cand.get("snr", -1e9))
        if snr > best_snr:
            best_snr = snr
            best_idx = i
    return best_idx if best_idx >= 0 else int(np.where(mask)[0][0])

for scenario in SCENARIOS:
    scenario_suffix = scenario
    tag_suffix = f"_{RESULTS_TAG}" if RESULTS_TAG else ""
    if SELECTED_MODEL == 'PPO':
        print("Loading PPO Agent...")
        env = LEOEnvPPO(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        agent = MaskablePPO.load(_resolve_file(base_dir, "handover_ppo_agent", MODELS_DIRNAME), env=env)
        env.env.earth.Training = False
        predict_fn = predict_valid_action
        
    elif SELECTED_MODEL == 'DQN':
        print("Loading DQN Agent...")
        env = LEOEnvDQN(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        agent = DQN.load(_resolve_file(base_dir, "handover_dqn_agent", MODELS_DIRNAME), env=env)
        env.env.earth.Training = False
        predict_fn = predict_valid_action_dqn

    elif SELECTED_MODEL == 'ORACLE':
        print("Loading Oracle policy...")
        env = LEOEnvPPO(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        agent = None
        env.env.earth.Training = False
        predict_fn = predict_valid_action_oracle

    elif SELECTED_MODEL == 'ORACLE_LATENCY':
        print("Loading Oracle Latency policy...")
        env = LEOEnvPPO(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        agent = None
        env.env.earth.Training = False
        predict_fn = predict_valid_action_oracle_latency

    elif SELECTED_MODEL == 'ORACLE_DROP':
        print("Loading Oracle Drop policy...")
        env = LEOEnvPPO(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        agent = None
        env.env.earth.Training = False
        predict_fn = predict_valid_action_oracle_drop
        
    elif SELECTED_MODEL in ('ODT', 'ODT_FINETUNED'):
        pass
        
    elif SELECTED_MODEL == 'BASELINE':
        print("Loading Baseline policy on RL wrapper...")
        env = LEOEnvPPO(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        env.env.earth.Training = False
        agent = None
        predict_fn = predict_valid_action_baseline_rl

    elif SELECTED_MODEL == 'BASELINE_LEGACY':
        print("Loading Baseline Legacy Environment...")
        LEOEnvModule.VERBOSE = False
        env = LEOEnvBase(constellation_name, route, scenario=scenario)
        env.earth.Training = False
        agent = None
        predict_fn = None
        
    else:
        raise ValueError(f"Invalid model selection: {SELECTED_MODEL}. Choose from 'ODT', 'ODT_FINETUNED', 'DQN', 'PPO', 'BASELINE', 'ORACLE', 'ORACLE_LATENCY', 'ORACLE_DROP', 'BASELINE_LEGACY'")

    if SELECTED_MODEL != 'ODT':
        print(f"Model {SELECTED_MODEL} loaded successfully!")
        print(f"Route duration: {route_duration}")
        print("=" * 50)

    def run_evaluation(model_label, env, agent, predict_fn, odt_state_dim=None, odt_action_dim=None):
        done = False
        step_count = 0
        results_filename = os.path.join(
            test_results_dir,
            f"{model_label}_observations_{scenario_suffix}{tag_suffix}.csv",
        )
        if os.path.exists(results_filename):
            os.remove(results_filename)

        obs, info = env.reset(seed=EVAL_SEED)

        print(f"Starting evaluation for {model_label} model (scenario: {scenario_suffix})...")
        print(f"Full route duration: {route_duration} steps")
        print(f"Results will be saved to: {results_filename}")

        while not done:
            if step_count % 25 == 0:
                print(f"Step {step_count} - Model: {model_label} - Scenario: {scenario_suffix}")

            if model_label == 'BASELINE_LEGACY':
                obs, reward, done, truncated, info = env.step()
            elif model_label in ('ORACLE', 'ORACLE_LATENCY', 'ORACLE_DROP'):
                mask = env.env._get_action_mask()
                action = predict_fn(env.env, obs, mask)
                obs, reward, done, truncated, info = env.env.step(action)
            elif model_label.startswith('ODT'):
                mask = env.env._get_action_mask()[:odt_action_dim]
                prev_obs = obs
                action = predict_fn(agent, prev_obs[:odt_state_dim], mask)
                next_obs, reward, done, truncated, info = env.step(action)
                # Keep context trajectory updated, but strict eval mode avoids buffer training writes.
                agent.step(prev_obs[:odt_state_dim], action, reward, next_obs[:odt_state_dim], done or truncated)
                obs = next_obs
            else:
                mask = env.env._get_action_mask()
                if model_label == 'BASELINE':
                    action = predict_fn(agent, obs, mask, env.env)
                else:
                    action = predict_fn(agent, obs, mask)
                if model_label == 'DQN':
                    obs, reward, done, truncated, info = env.step(action)
                else:
                    obs, reward, done, truncated, info = env.env.step(action)

            observation_data = [obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], obs[16], obs[17]]
            append_observation_to_file(observation_data, step_count, model_label, results_filename)
            step_count += 1

        print(f"\nCompleted {step_count} steps for {model_label} (scenario: {scenario_suffix})")

        final_results = {
            'model_tested': model_label,
            'scenario': scenario_suffix,
            'steps_completed': step_count,
            'route_duration': route_duration,
            'completion_status': done,
            'observations_file': results_filename
        }

        with open(
            os.path.join(
                test_results_dir,
                f"{model_label}_summary_{scenario_suffix}{tag_suffix}.pkl",
            ),
            'wb',
        ) as f:
            pickle.dump(final_results, f)

        print("All results saved successfully!")
        print(f"Model tested: {model_label}")
        print(f"Final step count: {step_count}")
        print(f"Route duration: {route_duration}")
        print(f"Completion status: {done}")
        print(f"Observations saved to: {results_filename}")
        print("Memory optimizations applied:")
        print("- Single model execution")
        print("- Real-time file appending (no memory accumulation)")
        print("- Periodic garbage collection")
        print("- No step limit - full route completed")

    if SELECTED_MODEL in ('ODT', 'ODT_FINETUNED'):
        selected_models = ODT_MODELS
        if SELECTED_MODEL == 'ODT_FINETUNED':
            selected_models = [m for m in ODT_MODELS if m[0] == 'ODT_FINETUNED']
        for model_label, model_file in selected_models:
            print(f"Loading {model_label} Agent...")
            env = LEOEnvODT(constellation_name, route, scenario=scenario)
            env = ActionMasker(env, mask_fn)
            model_path = _resolve_file(base_dir, model_file, MODELS_DIRNAME)
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
                target_return=float(checkpoint.get("target_return", 1.0)),
                buffer_size=150
            )
            agent.load(model_path)
            # RTG calibration at inference: prefer checkpoint value, fallback to dataset p75.
            agent.target_return = _calibrate_rtg_from_dataset(base_dir, agent.target_return)
            # Strict evaluation path.
            agent.set_eval_mode(True)
            agent.recent_episode_returns.clear()
            agent.recent_step_rewards.clear()
            env.env.earth.Training = False
            run_evaluation(model_label, env, agent, predict_valid_action_dt, state_dim, action_dim)
    else:
        run_evaluation(SELECTED_MODEL, env, agent, predict_fn)
