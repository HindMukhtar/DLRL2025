import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
from LEOEnvironmentRL import initialize, load_route_from_csv  # Use RL version
import pandas as pd
import os
from stable_baselines3 import DQN
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import torch
import random
import sb3_contrib
from HandoverEnvironment import LEOEnv as LEOEnvPPO 
from HandoverEnvironment import mask_fn, predict_valid_action
from HandoverEnvironment_DQN import LEOEnv as LEOEnvDQN
from HandoverEnvironment_DQN import predict_valid_action as predict_valid_action_dqn
from HandoverEnvironment_ODT import LEOEnv as LEOEnvODT
from HandoverEnvironment_ODT import predict_valid_action_dt
from ODT import OnlineDecisionTransformer
from LEOEnvironment import LEOEnv as LEOEnvBase
import pickle

# Create PPO environment
inputParams = pd.read_csv("input.csv")
constellation_name = inputParams['Constellation'][0]
route, route_duration = load_route_from_csv('route.csv', skip_rows=3)
ppo_env = LEOEnvPPO(constellation_name, route)
ppo_env = ActionMasker(ppo_env, mask_fn)
# Evaluation with debugging
obs, info = ppo_env.reset()
print(f"Initial mask sum: {np.sum(ppo_env.action_mask) if hasattr(ppo_env, 'action_mask') else 'No mask attr'}")

# Load PPO Agent 
ppo_agent = MaskablePPO("MlpPolicy", ppo_env, verbose=1)
ppo_agent.load("handover_ppo_agent")

# Create DQN Environment
inputParams = pd.read_csv("input.csv")
constellation_name = inputParams['Constellation'][0]
route, route_duration = load_route_from_csv('route.csv', skip_rows=3)
dqn_env = LEOEnvDQN(constellation_name, route)
dqn_env = ActionMasker(dqn_env, mask_fn)
# Evaluation with debugging
obs, info = dqn_env.reset()
print(f"Initial mask sum: {np.sum(dqn_env.action_mask) if hasattr(dqn_env, 'action_mask') else 'No mask attr'}")

# Load DQN Agent 
dqn_agent = DQN("MlpPolicy", dqn_env, verbose=1, buffer_size=100) 
dqn_agent.load("handover_dqn_agent")

# Create ODT Environment 
inputParams = pd.read_csv("input.csv")
constellation_name = inputParams['Constellation'][0]
route, route_duration = load_route_from_csv('route.csv', skip_rows=3)
odt_env = LEOEnvODT(constellation_name, route)
odt_env = ActionMasker(odt_env, mask_fn)
# Evaluation with debugging
obs, info = odt_env.reset()
print(f"Initial mask sum: {np.sum(odt_env.action_mask) if hasattr(odt_env, 'action_mask') else 'No mask attr'}")

# Load ODT Agent 
odt_agent = OnlineDecisionTransformer(
        state_dim=odt_env.observation_space.shape[0],
        action_dim=odt_env.action_space.n,
        max_length=20,
        embed_dim=64,
        num_layers=2,
        target_return=1.0
    )

model_path = 'decision_transformer_final.pth'
odt_agent.load(model_path)

# Initalize baseline environment
inputParams = pd.read_csv("input.csv")
constellation_name = inputParams['Constellation'][0]
route, route_duration = load_route_from_csv('route.csv', skip_rows=3)
base_env = LEOEnvBase(constellation_name, route)

# Evaluation with debugging
obs, info = base_env.reset()

# set training to false to enable saving plots 
base_env.earth.Training = False
dqn_env.env.earth.Training = False
ppo_env.env.earth.Training = False
odt_env.env.earth.Training = False

done_ppo = False
done_dqn = False
done_base = False
done_odt = False
step_count = 0

obs_ppo, info_ppo = ppo_env.reset()
obs_dqn, info_dqn = dqn_env.reset()
obs_base, info_base = base_env.reset()
obs_odt, info_odt = odt_env.reset()

obs_ppo_list = []
obs_dqn_list = []
obs_base_list = []
obs_odt_list = []

while not (done_ppo or done_dqn or done_base or done_odt):
    print(f"\n--- Step {step_count} ---")
    
    # PPO Agent Step
    mask_ppo = ppo_env.env._get_action_mask()
    print(f"PPO Valid actions: {np.sum(mask_ppo)}")
    action_ppo = predict_valid_action(ppo_agent, obs_ppo, mask_ppo)
    print(f"PPO Action: {action_ppo}, Valid: {mask_ppo[action_ppo]}")
    obs_ppo, reward_ppo, done_ppo, truncated_ppo, info_ppo = ppo_env.env.step(action_ppo)
    obs_ppo_list.append(obs_ppo)

    # DQN Agent Step
    mask_dqn = dqn_env.env._get_action_mask()
    print(f"DQN Valid actions: {np.sum(mask_dqn)}")
    action_dqn = predict_valid_action_dqn(dqn_agent, obs_dqn, mask_dqn)
    print(f"DQN Action: {action_dqn}, Valid: {mask_dqn[action_dqn]}")
    obs_dqn, reward_dqn, done_dqn, truncated_dqn, info_dqn = dqn_env.step(action_dqn)
    obs_dqn_list.append(obs_dqn)


    # ODT Agent Step
    mask_odt = odt_env.env._get_action_mask()
    print(f"ODT Valid actions: {np.sum(mask_odt)}")
    action_odt = predict_valid_action_dt(odt_agent, obs_odt, mask_odt)
    print(f"ODT Action: {action_odt}, Valid: {mask_odt[action_odt]}")
    obs_odt, reward_odt, done_odt, truncated_odt, info_odt = odt_env.step(action_odt)
    odt_agent.step(obs_odt, action_odt, reward_odt, obs_odt, done_odt or truncated_odt)
    obs_odt_list.append(obs_odt)

    # Baseline Environment Step
    obs_base, reward_base, done_base, truncated_base, info_base = base_env.step()
    obs_base_list.append(obs_base)
    
    step_count += 1

# Save results 

# Save observation lists to files for later extraction

# Save as pickle files (recommended for preserving exact numpy arrays)
with open('obs_base_list.pkl', 'wb') as f:
    pickle.dump(obs_base_list, f)

with open('obs_ppo_list.pkl', 'wb') as f:
    pickle.dump(obs_ppo_list, f)

with open('obs_dqn_list.pkl', 'wb') as f:
    pickle.dump(obs_dqn_list, f)

with open('obs_odt_list.pkl', 'wb') as f:
    pickle.dump(obs_odt_list, f)

# Alternative: Save as numpy arrays
np.save('obs_base_list.npy', np.array(obs_base_list))
np.save('obs_ppo_list.npy', np.array(obs_ppo_list))
np.save('obs_dqn_list.npy', np.array(obs_dqn_list))
np.save('obs_odt_list.npy', np.array(obs_odt_list))

