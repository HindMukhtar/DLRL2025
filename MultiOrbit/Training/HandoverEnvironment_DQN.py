import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from Environment.LEOEnvironmentRL import initialize, load_route_from_csv  # Use RL version
import pandas as pd
import math
from stable_baselines3 import DQN
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import torch
import random

import sb3_contrib

def _get_default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
print(dir(sb3_contrib))

class LEOEnv(gym.Env):
    """
    Gymnasium environment wrapper for the LEO satellite handover simulation.
    """

    def __init__(self, constellation_name, route, scenario=None, max_beams_per_step=12, seed=None):
        super(LEOEnv, self).__init__()
        self.base_seed = int(seed) if seed is not None else int(os.getenv("EVAL_SEED", "42"))
        self.current_seed = self.base_seed
        self.max_beams_per_step = max_beams_per_step
        self.action_space = spaces.Discrete(max_beams_per_step)

        #[lat, lon, alt, snr, load, handovers, allocated_bw, allocation_ratio, demand_MB, throughput_req, queing_delay_s, propagation_latency_s, transmission_rate_mbps, latency_req_s, beam_capacity, service_drop_s, dwell_remaining_s, ttt_remaining_s]
        # Observation space: 
        low = np.array([-90, -180, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([90, 180, 60000, 100, 1, 1000, 1000, 1, 1500, 60, 10, 10, 100, 10, 1000, 10, 120, 60], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.constellation = constellation_name 
        self.route = route 
        self.deltaT = 1
        self.scenario = scenario

        self.env = None
        self.earth = None
        self.aircraft = None
        self.current_step = 0
        self.handover_occurred = False
        self.action_mask = None
        self.action_mask_matrix = None
        self.last_qoe = None
        self.steering_obs = None
        self.max_action_candidates = 1
        self.max_steer_apps = 1
        self.prev_handover_total = 0

        np.random.seed(self.current_seed)
        random.seed(self.current_seed)

        self._setup_simulation(self.current_seed)

    def _setup_simulation(self, seed):
        self.env = simpy.Environment()
        self.earth = initialize(
            self.env,
            self.constellation,
            self.route,
            scenario=self.scenario,
            demand_seed=int(seed),
            handover_seed=int(seed) + 1000,
        )
        self.aircraft = self.earth.aircraft[0]
        self.current_step = 0

        self.max_action_candidates = max(1, min(self.max_beams_per_step, self.aircraft.max_steer_candidates))
        self.max_steer_apps = max(1, int(getattr(self.aircraft, "max_steer_apps", 1)))
        self.action_space = spaces.Discrete(self.max_action_candidates)
        self.prev_handover_total = 0

        qoe_list = self.earth.step_aircraft(actions=None)
        self.last_qoe = qoe_list[0] if qoe_list else {}
        self.steering_obs = self.aircraft.get_steering_observation()
        self.action_mask = self._get_action_mask()
    
    def _get_action_mask(self):
        matrix = np.zeros((self.max_steer_apps, self.max_action_candidates), dtype=bool)
        if not isinstance(self.steering_obs, dict):
            self.action_mask_matrix = matrix
            return np.any(matrix, axis=0)
        raw = np.asarray(self.steering_obs.get("action_mask"))
        if raw.size == 0:
            self.action_mask_matrix = matrix
            return np.any(matrix, axis=0)
        raw = raw > 0.5
        if raw.ndim == 1:
            raw = np.tile(raw.reshape(1, -1), (self.max_steer_apps, 1))
        app_upto = min(raw.shape[0], self.max_steer_apps)
        cand_upto = min(raw.shape[1], self.max_action_candidates)
        matrix[:app_upto, :cand_upto] = raw[:app_upto, :cand_upto]
        self.action_mask_matrix = matrix
        return np.any(matrix, axis=0)

    def _build_action_indices(self, action):
        if action is None or action < 0 or action >= self.max_action_candidates:
            return None
        if self.action_mask is not None and (action >= len(self.action_mask) or not self.action_mask[action]):
            return None

        candidates = getattr(self.aircraft, "current_steering_candidates", [])
        mask_mat = self.action_mask_matrix
        if mask_mat is None or mask_mat.size == 0:
            return [int(action)] * self.max_steer_apps

        # Global best candidate per constellation by SNR.
        best_by_const = {"OneWeb": (-1, -1e9), "Intelsat": (-1, -1e9)}
        for i, cand in enumerate(candidates):
            if i >= len(self.action_mask) or not self.action_mask[i] or cand is None:
                continue
            const = self.aircraft._constellation_key_from_sat(cand.get("sat"))
            if const not in best_by_const:
                continue
            snr = float(cand.get("snr", -1e9))
            if snr > best_by_const[const][1]:
                best_by_const[const] = (i, snr)

        app_ctx = []
        if isinstance(self.steering_obs, dict):
            app_ctx = self.steering_obs.get("app_context", [])
        if not isinstance(app_ctx, (list, tuple)):
            app_ctx = []

        actions = np.full(self.max_steer_apps, -1, dtype=np.int64)
        for app_idx in range(self.max_steer_apps):
            row = mask_mat[app_idx]
            valid = np.where(row)[0]
            if len(valid) == 0:
                continue

            # Keep scalar policy intent for first slot.
            if app_idx == 0 and int(action) in valid:
                actions[app_idx] = int(action)
                continue

            latency_req = None
            if app_idx < len(app_ctx) and isinstance(app_ctx[app_idx], dict):
                try:
                    latency_req = float(app_ctx[app_idx].get("latency_req_s", np.nan))
                except Exception:
                    latency_req = None

            if latency_req is not None and np.isfinite(latency_req):
                preferred = "OneWeb" if latency_req <= 0.12 else "Intelsat"
            else:
                preferred = "OneWeb" if (app_idx % 2 == 0) else "Intelsat"
            fallback = "Intelsat" if preferred == "OneWeb" else "OneWeb"

            pick = -1
            pref_idx = best_by_const[preferred][0]
            if pref_idx >= 0 and pref_idx in valid:
                pick = pref_idx
            else:
                fb_idx = best_by_const[fallback][0]
                if fb_idx >= 0 and fb_idx in valid:
                    pick = fb_idx
                else:
                    best_local = -1
                    best_local_snr = -1e9
                    for i in valid:
                        cand = candidates[i] if i < len(candidates) else None
                        if cand is None:
                            continue
                        snr = float(cand.get("snr", -1e9))
                        if snr > best_local_snr:
                            best_local_snr = snr
                            best_local = int(i)
                    pick = best_local if best_local >= 0 else int(valid[0])
            actions[app_idx] = int(pick)

        return actions.tolist() if np.any(actions >= 0) else None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_seed = self.base_seed if seed is None else int(seed)
        np.random.seed(self.current_seed)
        random.seed(self.current_seed)
        self._setup_simulation(self.current_seed)
        obs = self._get_obs()
        info = {
            "available_beams": int(np.sum(self.action_mask)),
            "action_mask": self.action_mask
        }
        return obs, info

    def step(self, action):
        reward_penalty = 0.0
        self.handover_occurred = False
        action_indices = self._build_action_indices(int(action) if action is not None else -1)
        if action_indices is None:
            reward_penalty = -0.2

        # Advance simulation
        qoe_list = self.earth.step_aircraft(actions=action_indices, folder="DQN")
        self.last_qoe = qoe_list[0] if qoe_list else {}
        self.earth.advance_constellation(self.earth.deltaT, self.env.now)
        
        self.env.run(until=self.env.now + self.earth.deltaT)
        self.current_step += 1

        self.steering_obs = self.aircraft.get_steering_observation()
        self.action_mask = self._get_action_mask()

        obs = self._get_obs()
        final_reward = self._get_reward() + reward_penalty
        terminated = False 
        truncated = False 
        if self.current_step >= len(self.route) - 1:
            terminated = True
        
        info = {
            "available_beams": int(np.sum(self.action_mask)),
            "action_mask": self.action_mask
        }

        return obs, final_reward, terminated, truncated, info

    def _refresh_qoe_cache(self):
        self.last_qoe = self.aircraft.get_qoe_metrics(self.aircraft.deltaT)
        return self.last_qoe

    def _get_obs(self):
        qoe = self.last_qoe if self.last_qoe is not None else self._refresh_qoe_cache()
        ac = self.aircraft
        lat = ac.latitude
        lon = ac.longitude
        alt = ac.height
        handovers = sum(st["handover_count"] for st in ac.links.values())
        primary = ac._get_primary_link_state()
        load = primary["connected_beam"].load if primary and primary["connected_beam"] else 0
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
        service_drop_s = qoe['service_drop_s'] if qoe and 'service_drop_s' in qoe else 0
        dwell_remaining_s = qoe['dwell_remaining_s'] if qoe and 'dwell_remaining_s' in qoe else 0
        ttt_remaining_s = qoe['ttt_remaining_s'] if qoe and 'ttt_remaining_s' in qoe else 0
        
        return np.array([lat, lon, alt, snr, load, handovers, allocated_bw, allocation_ratio, demand_MB, throughput_req, queing_delay_s, propagation_latency_s, transmission_rate_mbps, latency_req_s, beam_capacity, service_drop_s, dwell_remaining_s, ttt_remaining_s], dtype=np.float32)

    def _get_reward(self):
        qoe = self.last_qoe if self.last_qoe is not None else self._refresh_qoe_cache()
        if not qoe or "throughput_req_mbps" not in qoe or "latency_req_s" not in qoe:
            return 0.0

        alloc = float(qoe.get("allocation_ratio", 0.0))
        drop_norm = float(qoe.get("service_drop_s", 0.0)) / max(1e-9, float(self.earth.deltaT))
        lat_violation = float(qoe.get("latency_violation_rate", 0.0))
        handovers = sum(st["handover_count"] for st in self.aircraft.links.values())
        handover_delta = max(0, handovers - self.prev_handover_total)
        self.prev_handover_total = handovers
        self.handover_occurred = handover_delta > 0

        reward = alloc - 0.8 * drop_norm - 0.6 * lat_violation - 0.02 * handover_delta
        return float(np.clip(reward, -1.0, 1.0))

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
    """
    Select the valid action with the highest Q-value for DQN.
    """
    if not np.any(mask):
        print("No valid actions available! Returning penalty action.")
        return -1  # Use -1 to indicate no valid action
    obs_tensor = torch.tensor(obs, dtype=torch.float32).reshape(1, -1).to(model.device)
    q_values = model.q_net(obs_tensor).detach().cpu().numpy().flatten()
    q_values[~mask] = -1e10  # Mask invalid actions
    action = np.argmax(q_values)
    return action

def main():
    # Create the environment
    base_dir = PROJECT_DIR
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    input_path = os.path.join(base_dir, "Inputs", "input.csv")
    if not os.path.exists(input_path):
        input_path = os.path.join(base_dir, "input.csv")
    inputParams = pd.read_csv(input_path)
    constellation_name = inputParams['Constellation'][0]
    route_candidates = [
        os.path.join(base_dir, "routes", "route_5s_interpolated.csv"),
        os.path.join(base_dir, "route_5s_interpolated.csv"),
        os.path.join(os.path.dirname(base_dir), "Single Constellation ", "routes", "route_5s_interpolated.csv"),
        os.path.join(os.path.dirname(base_dir), "Single Constellation", "routes", "route_5s_interpolated.csv"),
        os.path.join(os.path.dirname(base_dir), "Single Constellation ", "routes", "route.csv"),
    ]
    route_path = next((p for p in route_candidates if os.path.exists(p)), None)
    if route_path is None:
        raise FileNotFoundError("Could not find route file. Checked:\n" + "\n".join(route_candidates))
    route, route_duration = load_route_from_csv(route_path, skip_rows=0)
    scenarios = [
        "load_cycle_1",
        "load_cycle_2",
        "load_cycle_5",
        "medium_aircraft",
        "large_aircraft",
        "snr_congested",
    ]
    scenario = scenarios[0]
    env = LEOEnv(constellation_name, route, scenario=scenario)
    env = ActionMasker(env, mask_fn)

    # Create the DQN agent
    #model = DQN("MlpPolicy", env, verbose=1, buffer_size=100, learning_starts=10, batch_size=32)
    device = _get_default_device()
    print(f"DQN device: {device}")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=5000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        device=device,
    )
    # Call once just to initialize everything 
    model.learn(total_timesteps=1)
    # Train the agent
    #model.learn(total_timesteps=10000)
    # Custom training using action masking 
    total_timesteps = 100000
    timesteps_per_scenario = max(1, total_timesteps // len(scenarios))
    for scenario in scenarios:
        print(f"Training DQN on scenario: {scenario}")
        env = LEOEnv(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        obs, info = env.reset()
        for step in range(timesteps_per_scenario):
            # Get current mask
            mask = env.env._get_action_mask()  # Unwrap if using ActionMasker

            # Get Q-values from the model
            obs_tensor = torch.tensor(obs, dtype=torch.float32).reshape(1, -1).to(model.device)
            q_values = model.q_net(obs_tensor).detach().cpu().numpy().flatten()

            # Mask invalid actions
            q_values[~mask] = -1e10
            action = np.argmax(q_values)

            # Step in the environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Store transition in replay buffer
            model.replay_buffer.add(obs, next_obs, action, reward, done, [info])

            # Train the model
            if step > model.learning_starts:
                model.train(batch_size=model.batch_size, gradient_steps=1)

            obs = next_obs
            if done or truncated:
                obs, info = env.reset()

    # Save the trained model
    model.save(os.path.join(models_dir, "handover_dqn_agent"))


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
        action = predict_valid_action(model, obs, mask)
        print(f"Manually predicted valid action: {action}")
        print(f"Action is valid: {mask[action]}")
        
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

    # Print evaluation summary using aircraft object
    aircraft = env.env.aircraft  # Access aircraft from your wrapped environment

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total evaluation steps: {step_count}")
    total_handovers = sum(st["handover_count"] for st in aircraft.links.values())
    print(f"Aircraft '{aircraft.id}' total handovers: {total_handovers}")

    connected_states = list(aircraft._iter_connected_link_states())
    if connected_states:
        for ckey, st in connected_states:
            print(f"Aircraft '{aircraft.id}' final connected beam ({ckey}): {st['connected_beam'].id}")
            print(f"Aircraft '{aircraft.id}' final SNR ({ckey}): {st['current_snr']:.2f} dB")
            print(f"Aircraft '{aircraft.id}' final Latency ({ckey}): {st['current_latency']*1e3:.2f} ms")
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
