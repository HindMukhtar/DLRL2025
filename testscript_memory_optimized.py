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
import gc  # Garbage collection
import psutil  # Memory monitoring

def print_memory_usage(step):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Step {step} - Memory Usage: {memory_mb:.1f} MB")

def save_and_clear_observations(obs_dict, step, save_interval=50):
    """Save observations periodically and clear memory"""
    if step % save_interval == 0 and step > 0:
        print(f"Saving observations at step {step}...")
        
        # Save current batch
        for agent_name, obs_list in obs_dict.items():
            filename = f'{agent_name}_obs_batch_{step//save_interval}.npy'
            np.save(filename, np.array(obs_list))
            obs_list.clear()  # Clear memory
        
        # Force garbage collection
        gc.collect()
        print_memory_usage(step)

def create_lightweight_environment(env_type, constellation_name, route):
    """Create environment with memory optimizations"""
    if env_type == "ppo":
        env = LEOEnvPPO(constellation_name, route)
        env = ActionMasker(env, mask_fn)
    elif env_type == "dqn":
        env = LEOEnvDQN(constellation_name, route)
        env = ActionMasker(env, mask_fn)
    elif env_type == "odt":
        env = LEOEnvODT(constellation_name, route)
        env = ActionMasker(env, mask_fn)
    elif env_type == "base":
        env = LEOEnvBase(constellation_name, route)
    
    return env

def main():
    # Load configuration once
    inputParams = pd.read_csv("input.csv")
    constellation_name = inputParams['Constellation'][0]
    route, route_duration = load_route_from_csv('route.csv', skip_rows=3)
    
    print(f"Route duration: {route_duration}")
    print_memory_usage(0)
    
    # === MEMORY OPTIMIZATION 1: Sequential Testing ===
    # Instead of running all agents simultaneously, run them one at a time
    agents_to_test = ["ppo", "dqn", "odt", "base"]
    results = {}
    
    for agent_type in agents_to_test:
        print(f"\n{'='*50}")
        print(f"TESTING {agent_type.upper()} AGENT")
        print(f"{'='*50}")
        
        # Create environment for this agent only
        if agent_type == "ppo":
            env = create_lightweight_environment("ppo", constellation_name, route)
            agent = MaskablePPO("MlpPolicy", env, verbose=0)  # verbose=0 to reduce output
            agent.load("handover_ppo_agent")
            predict_func = predict_valid_action
        elif agent_type == "dqn":
            env = create_lightweight_environment("dqn", constellation_name, route)
            agent = DQN("MlpPolicy", env, verbose=0, buffer_size=100)
            agent.load("handover_dqn_agent")
            predict_func = predict_valid_action_dqn
        elif agent_type == "odt":
            env = create_lightweight_environment("odt", constellation_name, route)
            agent = OnlineDecisionTransformer(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                max_length=20,
                embed_dim=32,  # Reduced from 64
                num_layers=2,
                target_return=1.0
            )
            agent.load('decision_transformer_final.pth')
            predict_func = predict_valid_action_dt
        elif agent_type == "base":
            env = create_lightweight_environment("base", constellation_name, route)
            agent = None
            predict_func = None
        
        # Set training to false
        if hasattr(env, 'env'):
            env.env.earth.Training = False
        else:
            env.earth.Training = False
        
        # Run evaluation for this agent
        obs_list = []
        metrics = {'rewards': [], 'actions': [], 'handovers': 0}
        
        obs, info = env.reset()
        done = False
        step_count = 0
        
        # === MEMORY OPTIMIZATION 2: Limit steps and save periodically ===
        max_steps = min(route_duration, 200)  # Limit maximum steps
        save_interval = 25  # Save every 25 steps
        
        while not done and step_count < max_steps:
            if step_count % 20 == 0:  # Reduce print frequency
                print(f"Step {step_count}/{max_steps}")
                print_memory_usage(step_count)
            
            if agent_type != "base":
                # Get action from agent
                if agent_type == "odt":
                    mask = env.env._get_action_mask()
                    action = predict_func(agent, obs, mask)
                    obs, reward, done, truncated, info = env.step(action)
                    agent.step(obs, action, reward, obs, done or truncated)
                else:
                    mask = env.env._get_action_mask()
                    action = predict_func(agent, obs, mask)
                    obs, reward, done, truncated, info = env.step(action)
                
                metrics['actions'].append(action)
                metrics['rewards'].append(reward)
            else:
                # Baseline environment
                obs, reward, done, truncated, info = env.step()
                metrics['rewards'].append(reward)
            
            # === MEMORY OPTIMIZATION 3: Store only essential data ===
            # Instead of storing full observations, store only key metrics
            if agent_type != "base":
                essential_obs = {
                    'snr': obs[3],
                    'handovers': obs[6],
                    'allocation': obs[8]
                }
            else:
                essential_obs = {
                    'snr': obs[3] if len(obs) > 3 else 0,
                    'handovers': obs[6] if len(obs) > 6 else 0,
                    'allocation': obs[8] if len(obs) > 8 else 0
                }
            
            obs_list.append(essential_obs)
            
            # === MEMORY OPTIMIZATION 4: Periodic cleanup ===
            if step_count % save_interval == 0 and step_count > 0:
                # Save current progress
                batch_filename = f'{agent_type}_batch_{step_count//save_interval}.npy'
                np.save(batch_filename, obs_list[-save_interval:])
                
                # Keep only recent observations in memory
                if len(obs_list) > save_interval * 2:
                    obs_list = obs_list[-save_interval:]
                
                # Force garbage collection
                gc.collect()
            
            step_count += 1
            
            if done or truncated:
                break
        
        # Store final results
        results[agent_type] = {
            'steps': step_count,
            'total_reward': sum(metrics['rewards']),
            'avg_reward': np.mean(metrics['rewards']) if metrics['rewards'] else 0,
            'final_obs': obs_list[-1] if obs_list else None
        }
        
        # Save final observations
        final_filename = f'{agent_type}_final_observations.npy'
        np.save(final_filename, obs_list)
        
        print(f"{agent_type.upper()} Results:")
        print(f"  Steps: {step_count}")
        print(f"  Total Reward: {sum(metrics['rewards']):.3f}")
        print(f"  Avg Reward: {np.mean(metrics['rewards']):.3f}")
        
        # === MEMORY OPTIMIZATION 5: Cleanup after each agent ===
        del env, agent, obs_list, metrics
        gc.collect()
        print_memory_usage(step_count)
    
    # Save comparison results
    print(f"\n{'='*50}")
    print("FINAL COMPARISON")
    print(f"{'='*50}")
    
    for agent_type, result in results.items():
        print(f"{agent_type.upper()}: {result['steps']} steps, "
              f"Total Reward: {result['total_reward']:.3f}, "
              f"Avg Reward: {result['avg_reward']:.3f}")
    
    # Save comparison
    with open('agent_comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nAll results saved. Memory usage optimized!")

if __name__ == "__main__":
    main()