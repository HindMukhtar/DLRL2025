import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from sb3_contrib.common.wrappers import ActionMasker

from HandoverEnvironment import LEOEnv as PPOEnv, predict_valid_action
from HandoverEnvironment_DQN import LEOEnv as DQNEnv, predict_valid_action as predict_valid_action_dqn
from LEOEnvironment import LEOEnv as BaselineEnv
from LEOEnvironmentRL import load_route_from_csv


def mask_fn(env):
    return env.action_mask


def _resolve_model_path(base_dir, name):
    zip_path = os.path.join(base_dir, f"{name}.zip")
    if os.path.exists(zip_path):
        return zip_path
    return os.path.join(base_dir, name)

def _collect_trajectories(env, predict_fn, model, episodes):
    trajectories = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        states = []
        actions = []
        rewards = []

        while not (done or truncated):
            mask = env.env._get_action_mask()
            if not np.any(mask):
                break
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
                }
            )
    return trajectories

def _collect_baseline_trajectories(env, episodes):
    trajectories = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        states = []
        actions = []
        rewards = []

        while not (done or truncated):
            next_obs, reward, done, truncated, info = env.step()
            beam = env.aircraft.connected_beam
            if beam and beam.id in env.all_beam_ids:
                action = env.all_beam_ids.index(beam.id)
            else:
                action = 0
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
                }
            )
    return trajectories


def main():
    base_dir = os.path.dirname(__file__)
    input_params = pd.read_csv(os.path.join(base_dir, "input.csv"))
    constellation_name = input_params["Constellation"][0]
    route, _ = load_route_from_csv(os.path.join(base_dir, "route_5s_interpolated.csv"), skip_rows=0)

    episodes = 50
    trajectories = []

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    ppo_env = PPOEnv(constellation_name, route)
    ppo_env = ActionMasker(ppo_env, mask_fn)
    ppo_model_path = _resolve_model_path(base_dir, "handover_ppo_agent")
    ppo_model = MaskablePPO.load(ppo_model_path, device="cpu")
    trajectories.extend(_collect_trajectories(ppo_env, predict_valid_action, ppo_model, episodes))

    dqn_env = DQNEnv(constellation_name, route)
    dqn_env = ActionMasker(dqn_env, mask_fn)
    dqn_model_path = _resolve_model_path(base_dir, "handover_dqn_agent")
    dqn_model = DQN.load(dqn_model_path, device="cpu")
    trajectories.extend(_collect_trajectories(dqn_env, predict_valid_action_dqn, dqn_model, episodes))

    baseline_env = BaselineEnv(constellation_name, route)
    trajectories.extend(_collect_baseline_trajectories(baseline_env, episodes))

    output_path = os.path.join(base_dir, "odt_offline_dataset.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Saved {len(trajectories)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
