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
import gc  # Add garbage collection

# ==============================================
# MODEL SELECTION - Choose which model to test
# ==============================================
# Options: 'ODT', 'DQN', 'PPO', 'BASELINE'
SELECTED_MODEL = 'BASELINE'  # Default: ODT

def append_observation_to_file(obs, step, model_name, filename):
    """Append single observation to file"""
    # Create header if file doesn't exist
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("step, snr, load, capacity, handovers, total_allocated_bw, allocation\n")
    
    # Append observation data
    with open(filename, 'a') as f:
        f.write(f"{step},{obs[0]},{obs[1]},{obs[2]},{obs[3]},{obs[4]},{obs[5]}\n")
    
    # Periodic garbage collection every 100 steps to manage memory
    if step % 100 == 0:
        gc.collect()

print(f"Selected Model for Testing: {SELECTED_MODEL}")
print("=" * 50)

# Initialize common parameters
inputParams = pd.read_csv("input.csv")
constellation_name = inputParams['Constellation'][0]
route, route_duration = load_route_from_csv('route_10s_interpolated.csv', skip_rows=3)

# Initialize only the selected model and environment
if SELECTED_MODEL == 'PPO':
    print("Loading PPO Agent...")
    env = LEOEnvPPO(constellation_name, route)
    env = ActionMasker(env, mask_fn)
    agent = MaskablePPO("MlpPolicy", env, verbose=0)
    agent.load("handover_ppo_agent")
    env.env.earth.Training = False
    predict_fn = predict_valid_action
    
elif SELECTED_MODEL == 'DQN':
    print("Loading DQN Agent...")
    env = LEOEnvDQN(constellation_name, route)
    env = ActionMasker(env, mask_fn)
    agent = DQN("MlpPolicy", env, verbose=0, buffer_size=50)
    agent.load("handover_dqn_agent")
    env.env.earth.Training = False
    predict_fn = predict_valid_action_dqn
    
elif SELECTED_MODEL == 'ODT':
    print("Loading ODT Agent...")
    env = LEOEnvODT(constellation_name, route)
    env = ActionMasker(env, mask_fn)
    agent = OnlineDecisionTransformer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        max_length=20,
        embed_dim=64,  
        num_layers=2,
        target_return=1.0
    )
    model_path = 'decision_transformer_final.pth'
    agent.load(model_path)
    env.env.earth.Training = False
    predict_fn = predict_valid_action_dt
    
elif SELECTED_MODEL == 'BASELINE':
    print("Loading Baseline Environment...")
    env = LEOEnvBase(constellation_name, route)
    env.earth.Training = False
    agent = None  # Baseline doesn't use an agent
    predict_fn = None
    
else:
    raise ValueError(f"Invalid model selection: {SELECTED_MODEL}. Choose from 'ODT', 'DQN', 'PPO', 'BASELINE'")

print(f"Model {SELECTED_MODEL} loaded successfully!")
print(f"Route duration: {route_duration}")
print("=" * 50)

# Initialize environment and tracking variables
done = False
step_count = 0
results_filename = f'{SELECTED_MODEL}_observations.csv'

# Reset environment
if SELECTED_MODEL == 'BASELINE':
    obs, info = env.reset()
else:
    obs, info = env.reset()

print(f"Starting evaluation for {SELECTED_MODEL} model...")
print(f"Full route duration: {route_duration} steps")
print(f"Results will be saved to: {results_filename}")

while not done:
    # Reduce print frequency to save memory
    if step_count % 25 == 0:
        print(f"Step {step_count} - Model: {SELECTED_MODEL}")
    
    # Take action based on selected model
    if SELECTED_MODEL == 'BASELINE':
        obs, reward, done, truncated, info = env.step()
        observation_data = [obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]  # SNR, load, capacity, handovers, total allocated bw,  allocation
    elif SELECTED_MODEL == 'ODT':
        mask = env.env._get_action_mask()
        action = predict_fn(agent, obs, mask)
        obs, reward, done, truncated, info = env.step(action)
        agent.step(obs, action, reward, obs, done or truncated)
        observation_data = [obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]
    else:  # PPO or DQN
        mask = env.env._get_action_mask()
        action = predict_fn(agent, obs, mask)
        if SELECTED_MODEL == 'DQN':
            obs, reward, done, truncated, info = env.step(action)
        else:  # PPO
            obs, reward, done, truncated, info = env.env.step(action)
        observation_data = [obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]
    
    # Append observation to file immediately
    append_observation_to_file(observation_data, step_count, SELECTED_MODEL, results_filename)
    
    step_count += 1

print(f"\nCompleted {step_count} steps for {SELECTED_MODEL}")

# Save final results summary
print("Saving final summary...")

final_results = {
    'model_tested': SELECTED_MODEL,
    'steps_completed': step_count,
    'route_duration': route_duration,
    'completion_status': done,
    'observations_file': results_filename
}

with open(f'{SELECTED_MODEL}_summary.pkl', 'wb') as f:
    pickle.dump(final_results, f)

print("All results saved successfully!")
print(f"Model tested: {SELECTED_MODEL}")
print(f"Final step count: {step_count}")
print(f"Route duration: {route_duration}")
print(f"Completion status: {done}")
print(f"Observations saved to: {results_filename}")
print("Memory optimizations applied:")
print("- Single model execution")
print("- Real-time file appending (no memory accumulation)")
print("- Periodic garbage collection")
print("- No step limit - full route completed")

