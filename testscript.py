import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
from LEOEnvironmentRL import initialize, load_route_from_csv
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
from HandoverEnvironment_ODT_Multiband import LEOEnv as LEOEnvODTMultiband


from ODT import OnlineDecisionTransformer
from LEOEnvironment import LEOEnv as LEOEnvBase
import pickle
import gc

# ==============================================
# MODEL SELECTION - Choose which model to test
# ==============================================
# Options: 'ODT', 'ODT_MULTIBAND', 'DQN', 'PPO', 'BASELINE'
SELECTED_MODEL = 'BASELINE'

def append_observation_to_file(obs, step, model_name, filename):
    """Append single observation to file"""
    # Create header if file doesn't exist
    if not os.path.exists(filename):
        # Determine header based on observation size
        header_base = "step,lat,lon,alt,snr,load,handovers,allocated_bw,allocation_ratio,demand_MB,throughput_req,queuing_delay_s,propagation_latency_s,transmission_rate_mbps,latency_req_s,beam_capacity"
        
        # Adjust header for multiband (larger state space)
        if len(obs) > 15:
            # Add headers for second band or extra features
            extra_cols = [f"feat_{i}" for i in range(16, len(obs)+1)]
            header_str = header_base + "," + ",".join(extra_cols) + "\n"
        else:
            header_str = header_base + "\n"
            
        with open(filename, 'w') as f:
            f.write(header_str)
    
    with open(filename, 'a') as f:
        obs_str = f"{step}," + ",".join([f"{x}" for x in obs]) + "\n"
        f.write(obs_str)
    
    if step % 100 == 0:
        gc.collect()

def print_observation_info(obs, step):
    """Print observation information"""
    print(f"\n--- Step {step} Status ---")
    print(f"Position: ({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.0f}m)")
    print(f"SNR: {obs[3]:.2f}dB, Load: {obs[4]:.2f}, Handovers: {int(obs[5])}")
    print(f"Allocation: {obs[7]:.2%}, Demand: {obs[8]:.2f}MB, Capacity: {obs[14]:.2f}MB")
    print(f"Throughput Req: {obs[9]:.2f}Mbps, Latency: {(obs[10]+obs[11])*1000:.2f}ms")

print(f"Selected Model for Testing: {SELECTED_MODEL}")
print("=" * 60)
print("CONFIGURATION")
print("OneWeb: fc = 12.5 GHz (Ku-band)")
print("Starlink: fc = 12.0 GHz (Ku-band)")
print("=" * 60)

# Initialize common parameters
inputParams = pd.read_csv("input.csv")
constellation_name = inputParams['Constellation'][0]
route, route_duration = load_route_from_csv('route_1s_interpolated_short.csv', skip_rows=3)
# route, route_duration = load_route_from_csv('route_30s_interpolated.csv', skip_rows=3)

print(f"\nConstellation: {constellation_name}")
print(f"Route duration: {route_duration} seconds")

# Initialize only the selected model and environment
if SELECTED_MODEL == 'PPO':
    print("\nLoading PPO Agent...")
    env = LEOEnvPPO(constellation_name, route)
    env = ActionMasker(env, mask_fn)
    agent = MaskablePPO("MlpPolicy", env, verbose=0)
    agent.load("handover_ppo_agent")
    env.env.earth.Training = False
    predict_fn = predict_valid_action
    
elif SELECTED_MODEL == 'DQN':
    print("\nLoading DQN Agent...")
    env = LEOEnvDQN(constellation_name, route)
    env = ActionMasker(env, mask_fn)
    agent = DQN("MlpPolicy", env, verbose=0, buffer_size=50)
    agent.load("handover_dqn_agent")
    env.env.earth.Training = False
    predict_fn = predict_valid_action_dqn
    
elif SELECTED_MODEL == 'ODT':
    print("\nLoading ODT Agent...")
    env = LEOEnvODT(constellation_name, route)
    env = ActionMasker(env, mask_fn)
    
    # Initialize ODT with 15-dimensional state space (single frequency)
    agent = OnlineDecisionTransformer(
        state_dim=15,  
        action_dim=env.action_space.n,
        max_length=20,
        embed_dim=128,
        num_layers=3,
        num_heads=4,
        target_return=1.0
    )
    
    model_path = 'decision_transformer_final.pth'
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded ODT model from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}")
        print("Using randomly initialized ODT model")
    
    env.env.earth.Training = False
    predict_fn = predict_valid_action_dt

elif SELECTED_MODEL == 'ODT_MULTIBAND':
    print("\nLoading ODT Multiband Agent...")
    
    if LEOEnvODTMultiband is None:
        raise ImportError("Cannot load ODT_MULTIBAND: Environment module missing.")

    env = LEOEnvODTMultiband(constellation_name, route)
    env = ActionMasker(env, mask_fn)
    
    # Initialize ODT with 29-dimensional state space (Multiband)
    # Assumes Multiband adds ~14 dims for the second band (15+14=29)
    agent = OnlineDecisionTransformer(
        state_dim=29,  
        action_dim=env.action_space.n,
        max_length=20,
        embed_dim=128,
        num_layers=3,
        num_heads=4,
        target_return=1.0
    )
    
    model_path = 'decision_transformer_multiband.pth'
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded ODT Multiband model from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}")
        print("Using randomly initialized ODT model")
    
    env.env.earth.Training = False
    predict_fn = predict_valid_action_dt
    
elif SELECTED_MODEL == 'BASELINE':
    print("\nLoading Baseline Environment...")
    env = LEOEnvBase(constellation_name, route)
    env.earth.Training = False
    agent = None
    predict_fn = None
    
else:
    raise ValueError(f"Invalid model selection: {SELECTED_MODEL}")

print(f"Model {SELECTED_MODEL} loaded successfully!")
print("=" * 60)

# Initialize tracking variables
done = False
step_count = 0
results_filename = f'{SELECTED_MODEL}_observations.csv'
summary_filename = f'{SELECTED_MODEL}_summary.pkl'

# Additional tracking
snr_history = []
load_history = []
allocation_history = []

# Reset environment
if SELECTED_MODEL == 'BASELINE':
    obs, info = env.reset()
else:
    obs, info = env.reset()

print(f"\nStarting evaluation for {SELECTED_MODEL} model...")
print(f"Full route duration: {route_duration} steps")
print(f"Results will be saved to: {results_filename}")
print("=" * 60)

# Print initial status
print_observation_info(obs, step_count)

while not done:
    # Reduce print frequency
    if step_count % 25 == 0 and step_count > 0:
        print(f"\n--- Progress: Step {step_count}/{int(route_duration)} ---")
    
    # Take action based on selected model
    if SELECTED_MODEL == 'BASELINE':
        obs, reward, done, truncated, info = env.step()
        
    elif SELECTED_MODEL in ['ODT', 'ODT_MULTIBAND']:
        mask = env.env._get_action_mask()
        action = predict_fn(agent, obs, mask)
        obs, reward, done, truncated, info = env.step(action)
        
        # Online learning step
        agent.step(obs, action, reward, obs, done or truncated)
        
    else:  # PPO or DQN
        mask = env.env._get_action_mask()
        action = predict_fn(agent, obs, mask)
        
        if SELECTED_MODEL == 'DQN':
            obs, reward, done, truncated, info = env.step(action)
        else:  # PPO
            obs, reward, done, truncated, info = env.env.step(action)
    
    # Track metrics
    snr_history.append(obs[3])
    load_history.append(obs[4])
    allocation_history.append(obs[7])
    
    # Append observation to file
    append_observation_to_file(obs, step_count, SELECTED_MODEL, results_filename)
    
    # Print detailed info every 100 steps
    if step_count % 100 == 0 and step_count > 0:
        print_observation_info(obs, step_count)
    
    step_count += 1

print(f"\n{'=' * 60}")
print(f"Completed {step_count} steps for {SELECTED_MODEL}")
print("=" * 60)

# Calculate statistics
if snr_history:
    avg_snr = np.mean([s for s in snr_history if s > -100])
    min_snr = min([s for s in snr_history if s > -100])
    max_snr = max([s for s in snr_history if s > -100])
else:
    avg_snr, min_snr, max_snr = 0, 0, 0

avg_load = np.mean(load_history) if load_history else 0
avg_allocation = np.mean(allocation_history) if allocation_history else 0

print("\n--- PERFORMANCE SUMMARY ---")
print(f"Average SNR: {avg_snr:.2f} dB")
print(f"SNR Range: {min_snr:.2f} to {max_snr:.2f} dB")
print(f"Average Load: {avg_load:.2%}")
print(f"Average Allocation Ratio: {avg_allocation:.2%}")

if SELECTED_MODEL in ['ODT', 'ODT_MULTIBAND']:
    stats = agent.get_performance_stats()
    print(f"\nODT Performance Stats:")
    print(f"Current adaptive target: {stats['current_target']:.3f}")
    print(f"Average recent return: {stats['avg_recent_return']:.3f}")
    if 'return_std' in stats:
        print(f"Return std dev: {stats['return_std']:.3f}")

# Save comprehensive results
print("\nSaving final summary...")

final_results = {
    'model_tested': SELECTED_MODEL,
    'config_type': 'MULTIBAND' if SELECTED_MODEL == 'ODT_MULTIBAND' else 'SINGLE_FREQ',
    'state_dim': len(obs),
    'steps_completed': step_count,
    'route_duration': route_duration,
    'completion_status': done,
    'observations_file': results_filename,
    'performance_stats': {
        'avg_snr': avg_snr,
        'min_snr': min_snr,
        'max_snr': max_snr,
        'avg_load': avg_load,
        'avg_allocation': avg_allocation,
    }
}

if SELECTED_MODEL in ['ODT', 'ODT_MULTIBAND']:
    final_results['odt_stats'] = stats

with open(summary_filename, 'wb') as f:
    pickle.dump(final_results, f)

print(f"\nAll results saved successfully!")
print(f"Summary saved to: {summary_filename}")
print(f"Observations saved to: {results_filename}")
print("=" * 60)
print("\nEVALUATION COMPLETE")
print("Configuration:")
print(f"- Constellation: {constellation_name}")
print(f"- Model: {SELECTED_MODEL}")
print(f"- State Dimension: {len(obs)}")
print("=" * 60)