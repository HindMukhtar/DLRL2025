import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
from LEOEnvironmentRL import initialize
import random
import pandas as pd

# -----------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------

def predict_valid_action_dt(agent, obs, mask):
    """Predict valid action using Decision Transformer (Multiband compatible)"""
    if not np.any(mask):
        return -1
    
    # The agent's predict_action method handles the mask internally if designed correctly,
    # or we filter the output here. Assuming standard ODT implementation:
    action = agent.predict_action(obs, mask)
    return action

# -----------------------------------------------------------------------
# Multiband Environment Class
# -----------------------------------------------------------------------

class LEOEnv(gym.Env):
    """
    Gymnasium environment wrapper for LEO satellite handover with MULTIBAND support.
    Extends the state space to 29 dimensions to include secondary band metrics.
    """

    def __init__(self, constellation_name, route):
        super(LEOEnv, self).__init__()

        # Action space remains the same (selecting a beam)
        # Note: In a full multiband setup, actions might imply selecting a band, 
        # but here we keep the action space consistent with the base env.
        self.action_space = spaces.Discrete(1) 

        # ------------------------------------------------------------------
        # Observation Space Configuration (29 Dimensions)
        # ------------------------------------------------------------------
        # Dimensions 0-14: Primary Band (Ku) - Same as standard ODT
        # Dimensions 15-28: Secondary Band (Ka) - Multiband specific features
        
        # 0: lat, 1: lon, 2: alt
        # 3: snr, 4: load, 5: handovers
        # 6: allocated_bw, 7: allocation_ratio, 8: demand_MB
        # 9: throughput_req, 10: queuing_delay, 11: propagation_latency
        # 12: transmission_rate, 13: latency_req, 14: beam_capacity
        # --- BAND 2 (Ka-like) ---
        # 15: snr_2, 16: load_2, 17: capacity_2
        # 18: allocated_bw_2, 19: allocation_ratio_2, 20: queuing_delay_2
        # 21: transmission_rate_2, 22: rain_attenuation_prob, 23: available_bw_2
        # 24: interference_2, 25: spectral_efficiency_2, 26: link_margin_2
        # 27: jitter_2, 28: packet_loss_2

        low = np.full(29, -np.inf, dtype=np.float32)
        high = np.full(29, np.inf, dtype=np.float32)
        
        # Refine bounds for specific known ranges where possible
        low[0], high[0] = -90, 90      # Lat
        low[1], high[1] = -180, 180    # Lon
        low[3], high[3] = -100, 100    # SNR
        low[4], high[4] = 0, 1         # Load
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.constellation = constellation_name 
        self.route = route 
        self.deltaT = 1

        self.env = None
        self.earth = None
        self.aircraft = None
        self.current_step = 0
        self.handover_occurred = False

        self.available_beams = [] 
        self.action_mask = None

        np.random.seed(42)
        random.seed(42)

        self._setup_simulation()

    def _setup_simulation(self):
        self.env = simpy.Environment()
        self.earth = initialize(self.env, self.constellation, self.route)
        self.aircraft = self.earth.aircraft[0]
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
        self.available_beams = self._get_available_beams()

    def _get_available_beams(self):
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
        self.action_mask = self._get_action_mask()
        obs = self._get_obs()
        info = {
            "available_beams": self.available_beams,
            "action_mask": self.action_mask
        }
        return obs, info

    def step(self, action):
        reward_penalty = 0
        self.handover_occurred = False

        # Handle penalty action
        if action == -1:
            obs = self._get_obs()
            base_reward = self._get_reward()
            final_reward = base_reward - 1.0
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
                    self.aircraft.connected_beam = chosen['beam']
                    self.aircraft.connected_satellite = chosen['sat']
                    self.aircraft.current_snr = chosen['snr']
                    self.aircraft.handover_count += 1
                    self.handover_occurred = True
                else:
                    self.aircraft.current_snr = chosen['snr']
            else:
                self.aircraft.connected_beam = None 
                self.aircraft.connected_satellite = None 
                self.aircraft.current_snr = -100
                reward_penalty = -1.0
        else:
            self.aircraft.connected_beam = None 
            self.aircraft.connected_satellite = None 
            self.aircraft.current_snr = -100
            reward_penalty = -1.0

        # Advance simulation
        # Using "ODT_MULTIBAND" folder for saving plots if configured in LEOEnvironment
        self.earth.step_aircraft(folder="ODT_MULTIBAND") 
        self.earth.advance_constellation(self.earth.deltaT, self.env.now)
        
        self.env.run(until=self.env.now + self.earth.deltaT)
        self.current_step += 1

        self.available_beams = self._get_available_beams()
        self.action_mask = self._get_action_mask()

        obs = self._get_obs()
        base_reward = self._get_reward()
        final_reward = base_reward + reward_penalty
        
        terminated = False 
        truncated = False 
        if self.current_step >= len(self.route) - 1:
            terminated = True
        
        info = {
            "available_beams": self.available_beams,
            "action_mask": self.action_mask,
            "handover_occurred": self.handover_occurred
        }

        return obs, final_reward, terminated, truncated, info

    def _get_obs(self):
        """
        Constructs the 29-dimensional observation vector.
        Combines standard metrics (Band 1/Ku) with simulated/calculated metrics for Band 2 (Ka).
        """
        qoe = self.aircraft.get_qoe_metrics(self.aircraft.deltaT)
        ac = self.aircraft
        
        # --- Base Metrics (Band 1 - e.g., Ku) ---
        lat = ac.latitude
        lon = ac.longitude
        alt = ac.height
        snr = qoe['SNR_dB'] if qoe and 'SNR_dB' in qoe else -100
        load = ac.connected_beam.load if ac.connected_beam else 0
        handovers = ac.handover_count
        
        allocated_bw = qoe['allocated_bandwidth_MB'] if qoe and 'allocated_bandwidth_MB' in qoe else 0
        allocation_ratio = qoe['allocation_ratio'] if qoe and 'allocation_ratio' in qoe else 0
        demand_MB = qoe['demand_MB'] if qoe and 'demand_MB' in qoe else 0
        throughput_req = qoe['throughput_req_mbps'] if qoe and 'throughput_req_mbps' in qoe else 0
        
        queuing_delay_s = qoe['queuing_delay_s'] if qoe and 'queuing_delay_s' in qoe else 0
        propagation_latency_s = qoe['propagation_latency_s'] if qoe and 'propagation_latency_s' in qoe else 0
        transmission_rate_mbps = qoe['transmission_rate_mbps'] if qoe and 'transmission_rate_mbps' in qoe else 0
        latency_req_s = qoe['latency_req_s'] if qoe and 'latency_req_s' in qoe else 0
        beam_capacity = qoe['beam_capacity_MB'] if qoe and 'beam_capacity_MB' in qoe else 0

        # --- Multiband Metrics (Band 2 - e.g., Ka) ---
        # Since the underlying physics simulation might only be single-band, 
        # we calculate Band 2 metrics relative to Band 1 to represent realistic trade-offs:
        # Ka-band typically: Higher Capacity (+50%), Higher Rain Attenuation (Lower SNR), Lower Load.
        
        # 1. SNR for Band 2 (Simulate rain fade effect: -3 to -8 dB penalty relative to Ku)
        rain_fade_penalty = 5.0 
        snr_2 = snr - rain_fade_penalty if snr > -90 else -100
        
        # 2. Load for Band 2 (Usually less congested: 70% of Band 1 load)
        load_2 = load * 0.7
        
        # 3. Capacity for Band 2 (Higher bandwidth: +50%)
        capacity_2 = beam_capacity * 1.5
        
        # 4. Allocated BW (Higher capacity allows more allocation, capped by demand)
        allocated_bw_2 = min(demand_MB, capacity_2 * (1 - load_2))
        
        # 5. Allocation Ratio Band 2
        allocation_ratio_2 = allocated_bw_2 / demand_MB if demand_MB > 0 else 1.0
        
        # 6. Queuing Delay (Lower due to higher service rate)
        tx_rate_2 = transmission_rate_mbps * 1.5
        queuing_delay_2 = queuing_delay_s * 0.6 if tx_rate_2 > 0 else 1000
        
        # 7. Transmission Rate
        transmission_rate_2 = tx_rate_2
        
        # 8. Rain Attenuation Probability (New feature)
        # Simple model: higher probability if SNR is already low
        rain_attenuation_prob = 0.8 if snr < 5 else 0.1
        
        # 9. Available Bandwidth
        available_bw_2 = capacity_2 * (1 - load_2)
        
        # 10. Interference (Higher in Ka due to spot beams? or Lower? Let's assume dependent on load)
        interference_2 = load_2 * 10 
        
        # 11. Spectral Efficiency (bits/s/Hz - depends on SNR)
        spectral_efficiency_2 = np.log2(1 + 10**(snr_2/10)) if snr_2 > -90 else 0
        
        # 12. Link Margin
        link_margin_2 = snr_2 - 5.0 # Assuming 5dB required threshold
        
        # 13. Jitter (Variance in delay - simulated)
        jitter_2 = queuing_delay_2 * 0.1
        
        # 14. Packet Loss (Simulated based on SNR)
        packet_loss_2 = 0.0 if snr_2 > 10 else (0.1 if snr_2 > 5 else 0.5)

        # Construct full 29D state vector
        obs_vector = np.array([
            # Band 1 (Original 15)
            lat, lon, alt,
            snr, load, handovers,
            allocated_bw, allocation_ratio, demand_MB,
            throughput_req, queuing_delay_s, propagation_latency_s,
            transmission_rate_mbps, latency_req_s, beam_capacity,
            
            # Band 2 (New 14)
            snr_2, load_2, capacity_2,
            allocated_bw_2, allocation_ratio_2, queuing_delay_2,
            transmission_rate_2, rain_attenuation_prob, available_bw_2,
            interference_2, spectral_efficiency_2, link_margin_2,
            jitter_2, packet_loss_2
        ], dtype=np.float32)

        return obs_vector

    def _get_reward(self):
        qoe = self.aircraft.get_qoe_metrics(self.aircraft.deltaT)
        if not qoe or "throughput_req_mbps" not in qoe or "latency_req_s" not in qoe:
            return 0.0

        deltaT = self.aircraft.deltaT
        
        # --- Throughput satisfaction ---
        throughput_req = qoe["throughput_req_mbps"]
        allocated_MB   = qoe["allocated_bandwidth_MB"]
        allocated_mbps = (allocated_MB * 8.0) / deltaT

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
                ratio = total_latency_s / latency_req_s
                latency_satisfaction = max(0.0, 1.0 - (ratio - 1.0))
        else:
            latency_satisfaction = 1.0

        # --- Multiband Considerations in Reward ---
        # (Optional) We could boost reward if Band 2 features indicate superior potential
        # but for now we stick to realized QoE.

        w_thr = 0.7 
        w_lat = 0.3 

        reward = (
            w_thr * throughput_satisfaction +
            w_lat * latency_satisfaction
        )

        if self.handover_occurred == True:
            reward -= 0.1

        return float(reward)

    def render(self):
        if self.earth:
            self.earth.plotMap(plotSat=True, plotBeams=True, plotAircrafts=True, aircrafts=[self.aircraft])

    def close(self): 
        pass