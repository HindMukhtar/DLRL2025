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

class LEOEnv(gym.Env):
    """
    Gymnasium environment wrapper for the LEO satellite handover simulation.
    """

    def __init__(self, constellation_name, route):
        super(LEOEnv, self).__init__()

        # We'll set a placeholder action space, but update it dynamically
        self.action_space = spaces.Discrete(1)  # Will be updated in reset/step

        # Observation space: [aircraft_lat, aircraft_lon, snr, beam_load, beam_capacity]
        low = np.array([-90, -180, 0, -100, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([90, 180, 60000, 100, 1, 0.48, 1000, 1000, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.constellation = constellation_name 
        self.route = route 
        self.deltaT = 1

        self.env = None
        self.earth = None
        self.aircraft = None
        self.current_step = 0

        self.available_beams = []  # List of available beams for current step
        self.action_mask = None

        np.random.seed(42)
        random.seed(42)

        self._setup_simulation()

    def _setup_simulation(self):

        self.env = simpy.Environment()
        self.earth = initialize(self.env, self.constellation, self.route)
        self.aircraft = self.earth.aircraft[0]  # Assume single aircraft for now
        self.current_step = 0

        # Build global beam id list and mapping
        self.all_beam_ids = []
        self.beam_id_to_obj = {}
        for plane in self.earth.LEO:
            for sat in plane.sats:
                for beam in sat.beams:
                    self.all_beam_ids.append(beam.id)
                    self.beam_id_to_obj[beam.id] = beam

        self.action_space = spaces.Discrete(len(self.all_beam_ids))

        # Find available beams for the initial state
        self.available_beams = self._get_available_beams()

    def _get_available_beams(self):
        # Returns a list of candidate beams (dicts) from scan_nearby_fast
        return self.aircraft.scan_nearby_fast(self.earth.LEO)
    
    def _get_action_mask(self):
        mask = np.zeros(len(self.all_beam_ids), dtype=bool)
        available_ids = [b['beam'].id for b in self.available_beams]
        for i, beam_id in enumerate(self.all_beam_ids):
            if beam_id in available_ids:
                mask[i] = True
        return mask    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(42)
        random.seed(42)
        self._setup_simulation()
        self.action_mask = self._get_action_mask()  # Store mask
        obs = self._get_obs()
        info = {
            "available_beams": self.available_beams,
            "action_mask": self.action_mask  # Return mask in info
        }
        return obs, info

    def step(self, action):
        # Map action index to beam id
        print(f"Action received: {action}")
        print(len(self.all_beam_ids))
        reward_penalty = 0

        # Handle penalty action
        if action == -1:
            print("No valid actions available! Returning penalty and skipping step.")
            obs = self._get_obs()
            base_reward = self._get_reward()
            final_reward = base_reward - 1.0  # Penalty
            terminated = False
            truncated = False
            if self.current_step >= len(self.route) - 1:
                terminated = True
            info = {
                "available_beams": self.available_beams,
                "action_mask": self.action_mask
            }
            self.current_step += 1
            return obs, final_reward, terminated, truncated, info
        
        if 0 <= action < len(self.all_beam_ids):
            beam_id = self.all_beam_ids[action]
            available_ids = [b['beam'].id for b in self.available_beams]
            
            if beam_id in available_ids:
                chosen = next(b for b in self.available_beams if b['beam'].id == beam_id)
                
                if self.aircraft.connected_beam != chosen['beam']:
                    print(f"Aircraft {self.aircraft.id} HANDOVER from {self.aircraft.connected_beam.id if self.aircraft.connected_beam else 'None'} to beam {chosen['beam'].id}")
                    self.aircraft.connected_beam = chosen['beam']
                    self.aircraft.connected_satellite = chosen['sat']
                    self.aircraft.current_snr = chosen['snr']
                    self.aircraft.handover_count += 1
                else:
                    print(f"Aircraft {self.aircraft.id} STAYING CONNECTED to beam {chosen['beam'].id}")
                    # Update SNR in case it changed
                    self.aircraft.current_snr = chosen['snr']
            else:
                print(f"Invalid action: beam {beam_id} not available")
                self.aircraft.connected_beam = None 
                self.aircraft.connected_satellite = None 
                self.aircraft.current_snr = -100
                reward_penalty = -1.0
        else:
            print(f"Action {action} out of bounds")
            self.aircraft.connected_beam = None 
            self.aircraft.connected_satellite = None 
            self.aircraft.current_snr = -100 
            reward_penalty = -1.0

        # Advance simulation
        self.earth.step_aircraft()
        self.earth.advance_constellation(self.earth.deltaT, self.env.now)
        
        self.env.run(until=self.env.now + self.earth.deltaT)
        self.current_step += 1

        # Update available beams for next step
        self.available_beams = self._get_available_beams()

        # Update action mask for next step 
        self.action_mask = self._get_action_mask()

        obs = self._get_obs()
        base_reward = self._get_reward()
        final_reward = base_reward + reward_penalty  # Add penalty
        print(f"Current simulation step: {self.current_step}")
        terminated = False 
        truncated = False 
        if self.current_step >= len(self.route) - 1:
            terminated = True
        
        info = {
            "available_beams": self.available_beams,
            "action_mask": self.action_mask  # Return mask in info
        }

        print(f"final reward: {final_reward}, base reward: {base_reward}, penalty: {reward_penalty}")

        return obs, final_reward, terminated, truncated, info

    def _get_obs(self):
        ac = self.aircraft
        lat = ac.latitude
        lon = ac.longitude
        alt = ac.height
        snr = ac.current_snr if ac.current_snr is not None else -100
        load = ac.connected_beam.load if ac.connected_beam else 0
        cap = ac.connected_beam.capacity if ac.connected_beam else 0
        handovers = ac.handover_count
        total_allocated_bw = ac.total_allocated_bandwidth 
        if ac.allocation_ratios: 
            allocation_to_demand = ac.allocation_ratios[-1]
        else: 
            allocation_to_demand = 0 
        return np.array([lat, lon, alt, snr, load, cap, handovers, total_allocated_bw, allocation_to_demand], dtype=np.float32)

    def _get_reward(self):
        if self.aircraft.allocation_ratios:
            return self.aircraft.allocation_ratios[-1]
        return 0.0

    def render(self):
        if self.earth:
            self.earth.plotMap(plotSat=True, plotBeams=True, plotAircrafts=True, aircrafts=[self.aircraft])

    def close(self): 
        pass

# 2. Define mask function
def mask_fn(env):
    mask = env.action_mask
    if mask is not None:
        print(f"Mask function called: {np.sum(mask)} valid actions")
        print(f"Sample valid actions: {np.where(mask)[0][:5]}")
    else:
        print("Mask function called: mask is None!")
    return mask


def predict_valid_action(model, obs, mask):
    """Manually ensure only valid actions are predicted"""
    if not np.any(mask):
        print("No valid actions available! Returning penalty action.")
        return -1  # Use -1 to indicate no valid action
    # Convert numpy array to torch tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
    
    # Get all action probabilities
    actions, values, log_probs = model.policy.forward(obs_tensor, deterministic=True)
    
    # Get the action logits
    distribution = model.policy.get_distribution(obs_tensor)
    logits = distribution.distribution.logits.detach().numpy().flatten()
    
    # Mask invalid actions by setting their logits to very low values
    masked_logits = logits.copy()
    masked_logits[~mask] = -1e10
    
    # Select action with highest masked logit
    action = np.argmax(masked_logits)
    return action

def main():
    # Create the environment
    inputParams = pd.read_csv("input.csv")
    constellation_name = inputParams['Constellation'][0]
    route, route_duration = load_route_from_csv('route.csv', skip_rows=3)
    env = LEOEnv(constellation_name, route)
    env = ActionMasker(env, mask_fn)

    # Create the DQN agent
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("handover_ppo_agent")


    # Evaluation with debugging
    obs, info = env.reset()
    print(f"Initial mask sum: {np.sum(env.action_mask) if hasattr(env, 'action_mask') else 'No mask attr'}")

    # set training to false to enable saving plots 
    env.env.earth.Training = False

    done = False
    step_count = 0
    while not done:
        print(f"\n--- Step {step_count} ---")
        
        # Get current mask
        mask = env.env._get_action_mask()
        print(f"Valid actions: {np.sum(mask)}")
        
        # Predict valid action manually
        action = predict_valid_action(model, obs, mask)
        print(f"Manually predicted valid action: {action}")
        print(f"Action is valid: {mask[action]}")
        
        obs, reward, done, truncated, info = env.env.step(action)
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
