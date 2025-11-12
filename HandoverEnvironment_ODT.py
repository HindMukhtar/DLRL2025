import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
from LEOEnvironmentRL import initialize, load_route_from_csv  # Use RL version
import pandas as pd
import os
from ODT import ActionMasker 
from ODT import OnlineDecisionTransformer, LEOEnv, mask_fn, predict_valid_action_dt
import torch
import random

def main(): 
    # Create the environment
    inputParams = pd.read_csv("input.csv")
    constellation_name = inputParams['Constellation'][0]
    route, route_duration = load_route_from_csv('route.csv', skip_rows=3)

    env = LEOEnv(constellation_name, route)
    env = ActionMasker(env, mask_fn)


    model = OnlineDecisionTransformer(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            max_length=20,
            embed_dim=64,
            num_layers=2,
            target_return=1.0
        )
    
    # Load pretrained ODT model 
    model_path = 'decision_transformer_final.pth'
    model.load(model_path)

    # Evaluation with debugging
    obs, info = env.reset()
    print(f"Initial mask sum: {np.sum(env.action_mask) if hasattr(env, 'action_mask') else 'No mask attr'}")

    # set training to false to enable saving plots 
    env.env.earth.Training = False

    done = False
    step_count = 0
    while not done and step_count < route_duration:
        print(f"\n--- Step {step_count} ---")
        
        # Get current mask
        mask = env.env._get_action_mask()
        print(f"Valid actions: {np.sum(mask)}")
        
        # Predict valid action manually
        action = predict_valid_action_dt(model, obs, mask)
        print(f"Manually predicted valid action: {action}")
        print(f"Action is valid: {mask[action]}")
        
        next_obs, reward, done, truncated, info = env.step(action)
        # Add information to the agent 
        model.step(obs, action, reward, next_obs, done or truncated)
        obs = next_obs
        step_count += 1

    # Print evaluation summary using aircraft object
    aircraft = env.env.aircraft  # Access aircraft from your wrapped environment

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total evaluation steps: {step_count}")
    print(f"Aircraft '{aircraft.id}' total handovers: {aircraft.handover_count}")

    if aircraft.connected_beam:
        print(f"Aircraft '{aircraft.id}' final connected beam: {aircraft.connected_beam.id}")
        print(f"Aircraft '{aircraft.id}' final SNR: {aircraft.current_snr:.2f} dB")
        print(f"Aircraft '{aircraft.id}' total allocated BW: {aircraft.total_allocated_bandwidth:.2f} MB")
        if aircraft.allocation_ratios:
            print(f"Aircraft '{aircraft.id}' Average Allocation to demand: {sum(aircraft.allocation_ratios)/len(aircraft.allocation_ratios):.3f}")
        else:
            print(f"Aircraft '{aircraft.id}' Average Allocation to demand: N/A")
    else:
        print(f"Aircraft '{aircraft.id}' ended the evaluation with no connection.")

    print("="*50)

if __name__ == "__main__":
    main()
