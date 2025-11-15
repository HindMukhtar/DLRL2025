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
import gc  # Add garbage collection

def main(): 
    # Create the environment
    inputParams = pd.read_csv("input.csv")
    constellation_name = inputParams['Constellation'][0]
    route, route_duration = load_route_from_csv('route.csv', skip_rows=3)

    env = LEOEnv(constellation_name, route)
    env = ActionMasker(env, mask_fn)


    # Create memory-efficient ODT model for evaluation only
    model = OnlineDecisionTransformer(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            max_length=20,
            embed_dim=64,
            num_layers=2,
            target_return=1.0,
            buffer_size=1  # Minimal buffer since we don't train during evaluation
        )
    
    # Load pretrained ODT model 
    model_path = 'decision_transformer_final.pth'
    model.load(model_path)
    
    # Enable evaluation mode to prevent memory accumulation
    model.set_eval_mode(True)
    
    # Disable gradient computation for memory efficiency
    torch.set_grad_enabled(False)

    # Evaluation with minimal debugging
    obs, info = env.reset()

    # set training to false to enable saving plots 
    env.env.earth.Training = False

    done = False
    step_count = 0
    
    print(f"Starting evaluation for {route_duration} steps...")
    
    while not done and step_count < route_duration:
        # Reduce print frequency to every 50 steps
        if step_count % 50 == 0:
            print(f"Step {step_count}/{route_duration} ({100*step_count/route_duration:.1f}%)")
        
        # Get current mask
        mask = env.env._get_action_mask()
        
        # Predict action (no debugging prints)
        action = predict_valid_action_dt(model, obs, mask)
        
        next_obs, reward, done, truncated, info = env.step(action)
        
        # This will be skipped automatically in evaluation mode
        model.step(obs, action, reward, next_obs, done or truncated)
        
        obs = next_obs
        step_count += 1
        
        # Periodic memory cleanup
        gc.collect()

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
