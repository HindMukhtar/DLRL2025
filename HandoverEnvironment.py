import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
from LEOEnvironment import initialize

class LEOEnv(gym.Env):
    """
    Gymnasium environment wrapper for the LEO satellite handover simulation.
    """

    def __init__(self, img_path="PopMap_500.png", input_csv="input.csv", movementTime=20, deltaT=20, max_steps=100):
        super(LEOEnv, self).__init__()

        # Action space: 0 = stay, 1 = handover (example, can be expanded)
        self.action_space = spaces.Discrete(2)

        # Observation space: [aircraft_lat, aircraft_lon, snr, beam_load, beam_capacity]
        # You can expand this as needed
        low = np.array([-90, -180, -100, 0, 0], dtype=np.float32)
        high = np.array([90, 180, 100, 100, 1000], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.img_path = img_path
        self.input_csv = input_csv
        self.movementTime = movementTime
        self.deltaT = deltaT
        self.max_steps = max_steps

        self.env = None
        self.earth = None
        self.aircraft = None
        self.current_step = 0

        self._setup_simulation()

    def _setup_simulation(self):
        import pandas as pd
        import os

        # Ensure input.csv exists
        if not os.path.exists(self.input_csv):
            with open(self.input_csv, "w") as f:
                f.write("Test length,Constellation\n")
                f.write("100,OneWeb\n")

        inputParams = pd.read_csv(self.input_csv)

        # Ensure population map exists
        if not os.path.exists(self.img_path):
            from PIL import Image
            Image.new('L', (500, 250)).save(self.img_path)

        self.env = simpy.Environment()
        self.earth = initialize(self.env, self.img_path, inputParams, self.movementTime)
        self.aircraft = self.earth.aircraft[0]  # Assume single aircraft for now
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_simulation()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Example: action could be used to force a handover or stay
        # For now, the environment logic is handled by the simpy/earth step
        self.env.run(until=self.env.now + self.deltaT)
        self.current_step += 1

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Example observation: aircraft lat, lon, snr, beam load, beam capacity
        ac = self.aircraft
        lat = ac.latitude
        lon = ac.longitude
        snr = ac.current_snr if ac.current_snr is not None else 0
        load = ac.connected_beam.load if ac.connected_beam else 0
        cap = ac.connected_beam.capacity if ac.connected_beam else 0
        return np.array([lat, lon, snr, load, cap], dtype=np.float32)

    def _get_reward(self):
        # Example reward: allocation ratio (maximize throughput satisfaction)
        if self.aircraft.allocation_ratios:
            return self.aircraft.allocation_ratios[-1]
        return 0.0

    def render(self):
        # Optionally call earth.plotMap() or similar for visualization
        if self.earth:
            self.earth.plotMap(plotSat=True, plotBeams=True, plotAircrafts=True, aircrafts=[self.aircraft])

    def close(self): 
        pass 