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

    def __init__(self, constellation_name, route, max_beams_per_step=64):
        super(LEOEnv, self).__init__()

        # Limit action space to reduce numerical instability
        self.max_beams_per_step = max_beams_per_step
        self.action_space = spaces.Discrete(self.max_beams_per_step)

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

        self.available_beams = []  # List of available beams for current step (limited)
        self.action_mask = None
        self.current_beam_candidates = []  # Current step's beam candidates

        np.random.seed(42)
        random.seed(42)

        self._setup_simulation()

    def _setup_simulation(self):

        self.env = simpy.Environment()
        self.earth = initialize(self.env, self.constellation, self.route)
        self.aircraft = self.earth.aircraft[0]  # Assume single aircraft for now
        self.current_step = 0

        # Find available beams for the initial state
        self.available_beams = self._get_available_beams()
        self._update_action_candidates()

    def _get_available_beams(self):
        # Returns a list of candidate beams (dicts) from scan_nearby_fast
        return self.aircraft.scan_nearby_fast(self.earth.LEO)
    
    def _update_action_candidates(self):
        """Update the current beam candidates (limited to max_beams_per_step)"""
        all_candidates = self.available_beams
        
        # Sort by SNR (descending) and limit to max_beams_per_step
        if all_candidates:
            sorted_candidates = sorted(all_candidates, key=lambda x: x['snr'], reverse=True)
            self.current_beam_candidates = sorted_candidates[:self.max_beams_per_step]
        else:
            self.current_beam_candidates = []
        
        # Pad with None if we have fewer candidates than max_beams_per_step
        while len(self.current_beam_candidates) < self.max_beams_per_step:
            self.current_beam_candidates.append(None)
    
    def _get_action_mask(self):
        """Create action mask for current beam candidates"""
        mask = np.zeros(self.max_beams_per_step, dtype=bool)
        for i, candidate in enumerate(self.current_beam_candidates):
            if candidate is not None:
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
            "available_beams": len([c for c in self.current_beam_candidates if c is not None]),
            "action_mask": self.action_mask  # Return mask in info
        }
        return obs, info

    def step(self, action):
        # Map action index to beam candidate
        print(f"Action received: {action}")
        print(f"Available beam candidates: {len([c for c in self.current_beam_candidates if c is not None])}")
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
                "available_beams": len([c for c in self.current_beam_candidates if c is not None]),
                "action_mask": self.action_mask
            }
            self.current_step += 1
            return obs, final_reward, terminated, truncated, info
        
        # Check if action is valid and within bounds
        if 0 <= action < self.max_beams_per_step and self.current_beam_candidates[action] is not None:
            chosen = self.current_beam_candidates[action]
            
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
            print(f"Invalid action: {action}")
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
        self._update_action_candidates()

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
            "available_beams": len([c for c in self.current_beam_candidates if c is not None]),
            "action_mask": self.action_mask  # Return mask in info
        }

        print(f"final reward: {final_reward}, base reward: {base_reward}, penalty: {reward_penalty}")

        return obs, final_reward, terminated, truncated, info

    def _get_obs(self):
        qoe = self.aircraft.get_qoe_metrics(self.aircraft.deltaT)
        ac = self.aircraft
        lat = ac.latitude
        lon = ac.longitude
        alt = ac.height
        handovers = ac.handover_count
        load = ac.connected_beam.load if ac.connected_beam else 0
        snr = qoe['SNR_dB'] if qoe and 'SNR_dB' in qoe else -100
        allocated_bw = qoe['allocated_bandwidth_MB'] if qoe and 'allocated_bandwidth_MB' in qoe else 0
        allocation_ratio = qoe['allocation_ratio'] if qoe and 'allocation_ratio' in qoe else 0
        demand_MB = qoe['demand_MB'] if qoe and 'demand_MB' in qoe else 0
        throughput_req = qoe['throughput_req_mbps'] if qoe and 'throughput_req_mbps' in qoe else 0
        queing_delay_s = qoe['queuing_delay_s'] if qoe and 'queuing_delay_s' in qoe else 0
        propagation_latency_s = qoe['propagation_latency_s'] if qoe and 'propagation_latency_s' in qoe else 0
        transmission_rate_mbps = qoe['transmission_rate_mbps'] if qoe and 'transmission_rate_mbps' in qoe else 0
        latency_req_s = qoe['latency_req_s'] if qoe and 'latency_req_s' in qoe else 0
        beam_capacity = qoe['beam_capacity_MB'] if qoe and 'beam_capacity_MB' in qoe else 0
        
        return np.array([lat, lon, alt, snr, load, handovers, allocated_bw, allocation_ratio, demand_MB, throughput_req, queing_delay_s, propagation_latency_s, transmission_rate_mbps, latency_req_s, beam_capacity], dtype=np.float32)

    def _get_reward(self):
        qoe = self.aircraft.get_qoe_metrics(self.aircraft.deltaT)
        if not qoe or "throughput_req_mbps" not in qoe or "latency_req_s" not in qoe:
            return 0.0

        print(f"QoE metrics: {qoe}")

        deltaT = self.aircraft.deltaT

        # --- Throughput satisfaction ---
        throughput_req = qoe["throughput_req_mbps"]          # sum of min app throughputs (Mbps)
        allocated_MB   = qoe["allocated_bandwidth_MB"]       # MB over this timestep
        allocated_mbps = (allocated_MB * 8.0) / deltaT       # Mb / s

        if throughput_req > 0:
            throughput_satisfaction = min(allocated_mbps / throughput_req, 1.0)
        else:
            throughput_satisfaction = 1.0

        # --- Latency satisfaction ---
        total_latency_s = qoe["queuing_delay_s"] + qoe["propagation_latency_s"]
        latency_req_s   = qoe["latency_req_s"]

        if latency_req_s > 0:
            if total_latency_s <= latency_req_s:
                latency_satisfaction = 1.0
            else:
                # degrade linearly from 1 at threshold to 0 at 2×threshold
                ratio = total_latency_s / latency_req_s
                latency_satisfaction = max(0.0, 1.0 - (ratio - 1.0))
        else:
            latency_satisfaction = 1.0

        # --- Combine throughput + latency ---
        w_thr = 0.7   # throughput weight
        w_lat = 0.3   # latency weight

        reward = (
            w_thr * throughput_satisfaction +
            w_lat * latency_satisfaction
        )

        # Optional: handover penalty if you track it
        # if self.handover_happened:
        #     reward -= 0.05

        return float(reward)

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
    
    # Get action using the model's predict method with action mask
    # This is more stable than manually manipulating logits
    try:
        action, _states = model.predict(obs_tensor.numpy(), action_masks=mask.reshape(1, -1), deterministic=True)
        return int(action[0]) if hasattr(action, '__len__') else int(action)
    except Exception as e:
        print(f"Prediction error: {e}. Falling back to random valid action.")
        # Fallback: select a random valid action
        valid_actions = np.where(mask)[0]
        return np.random.choice(valid_actions) if len(valid_actions) > 0 else -1

def main():
    # Create the environment
    inputParams = pd.read_csv("input.csv")
    constellation_name = inputParams['Constellation'][0]
    route, route_duration = load_route_from_csv('route_10s_interpolated.csv', skip_rows=0)
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
