# import libraries 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


# Beam handover environment 
class BeamHandoverEnv(gym.Env):
    """Custom Gymnasium environment for satellite beam handover."""
    def __init__(self, num_beams=5, episode_length=100):
        super().__init__()
        self.num_beams = num_beams
        self.episode_length = episode_length
        self.current_step = 0
        # Action space: pick one beam
        self.action_space = spaces.Discrete(self.num_beams)
        # Observation: [x_position, y_position, time_step, current_beam_id, aircraft demand]
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.beams = self._generate_beams()
        self.demand = self._generate_demand()
        self.position = (0, 0)
        self.current_beam = 0
        self.total_throughput = 0
    def _generate_beams(self):
        # Simulate static GEO beams
        return [
            {
                "capacity": random.uniform(10, 20),
                "MIR": random.uniform(5, 10),
                "congestion": random.uniform(0.1, 0.7)
            }
            for _ in range(self.num_beams)
        ] 
    def _generate_demand(self):
        return random.uniform(5, 15)  # or use a pattern: sinusoidal, bursty, etc.    
    def _get_observation(self):
        return np.array([
            self.position[0],
            self.position[1],
            self.current_step,
            self.current_beam, 
            self.demand
        ], dtype=np.float32)
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.position = (0.0, 0.0)
        self.current_beam = 0
        self.total_throughput = 0.0
        self.beams = self._generate_beams()
        obs = self._get_observation()
        return obs, {}
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        self.current_step += 1
        self.position = (self.position[0] + 1.0, self.position[1] + 0.5)
        # Select beam and calculate reward
        beam = self.beams[action]
        congestion_factor = 1.0 - beam["congestion"]
        throughput = min(beam["MIR"], beam["capacity"]) * congestion_factor
        handover_penalty = 1.0 if action != self.current_beam else 0.0
        reward = throughput - handover_penalty
        self.current_beam = action
        self.total_throughput += throughput
        obs = self._get_observation()
        done = self.current_step >= self.episode_length
        info = {}
        return obs, reward, done, False, info
    def render(self):
        print(f"Step {self.current_step}: Position={self.position}, Beam={self.current_beam}")
    def close(self):
        pass