import time
import pandas as pd
import math
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
import matplotlib
import simpy
import os
import random
from collections import deque

# TensorFlow and Keras for the DQN model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Cartopy for plotting maps
import cartopy.crs as ccrs
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from cartopy.feature import ShapelyFeature
from shapely.geometry import box
from shapely.affinity import scale, rotate, translate
from scipy.spatial import KDTree
from PIL import Image

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

###############################################################################
#################################    Simpy    #################################
###############################################################################

def simProgress(simTimelimit, env):
    """Monitors and prints the simulation progress."""
    timeSteps = 100
    timeStepSize = simTimelimit / timeSteps
    progress = 1
    startTime = time.time()
    yield env.timeout(timeStepSize)
    while True:
        elapsedTime = time.time() - startTime
        estimatedTimeRemaining = elapsedTime * (timeSteps / progress) - elapsedTime
        print(f"Simulation progress: {progress}% Estimated time remaining: {int(estimatedTimeRemaining)} seconds Current simulation time: {env.now}", end='\r')
        yield env.timeout(timeStepSize)
        progress += 1

###############################################################################
###############################    Constants    ###############################
###############################################################################

Re = 6378e3  # Radius of the earth [m]
G = 6.67259e-11  # Universal gravitational constant [m^3/kg s^2]
Me = 5.9736e24  # Mass of the earth
Te = 86164.28450576939  # Time required by Earth for 1 rotation
Vc = 299792458  # Speed of light [m/s]
k = 1.38e-23  # Boltzmann's constant
eff = 0.55  # Efficiency of the parabolic antenna

###############################################################################
#############################    DQN Classes    ###############################
###############################################################################

class ExperienceReplay:
    """A deque to store agent's experiences for replay."""
    def __init__(self, maxlen=2000):
        self.buffer = deque(maxlen=maxlen)

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def get_batch(self, batch_size):
        """Samples a random batch of experiences from the buffer."""
        return random.sample(self.buffer, batch_size)

    @property
    def buffer_size(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Learning Agent for beam selection."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ExperienceReplay(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Builds the Neural Network for the Q-value approximation."""
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        """Chooses an action based on the current state (epsilon-greedy)."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Trains the model using a batch of experiences from memory."""
        # Corrected line: Use self.state_size instead of accessing model.layers[0].input_shape[1]
        if self.memory.buffer_size < batch_size:
            return
        minibatch = self.memory.get_batch(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
###############################################################################
###############################     Classes    ################################
###############################################################################
class RFlink:
    def __init__(self, frequency, bandwidth, maxPtx, aDiameterTx, aDiameterRx, pointingLoss, noiseFigure,
                 noiseTemperature, min_rate):
        self.f = frequency
        self.B = bandwidth
        self.maxPtx = maxPtx
        self.maxPtx_db = 10 * math.log10(self.maxPtx)
        self.Gtx = 10 * math.log10(eff * ((math.pi * aDiameterTx * self.f / Vc) ** 2))
        self.Grx = 10 * math.log10(eff * ((math.pi * aDiameterRx * self.f / Vc) ** 2))
        self.G = self.Gtx + self.Grx - 2 * pointingLoss
        self.No = 10 * math.log10(self.B * k) + noiseFigure + 10 * math.log10(
            290 + (noiseTemperature - 290) * (10 ** (-noiseFigure / 10)))
        self.GoT = 10 * math.log10(eff * ((math.pi * aDiameterRx * self.f / Vc) ** 2)) - noiseFigure - 10 * math.log10(
            290 + (noiseTemperature - 290) * (10 ** (-noiseFigure / 10)))
        self.min_rate = min_rate

    def __repr__(self):
        return '\n Carrier frequency = {} GHz\n Bandwidth = {} MHz\n Transmission power = {} W\n Gain per antenna: Tx {}  Rx {}\n Total antenna gain = {} dB\n Noise power = {} dBW\n G/T = {} dB/K'.format(
            self.f / 1e9,
            self.B / 1e6,
            self.maxPtx,
            '%.2f' % self.Gtx,
            '%.2f' % self.Grx,
            '%.2f' % self.G,
            '%.2f' % self.No,
            '%.2f' % self.GoT,
        )
    
class OrbitalPlane:
    def __init__(self, ID, h, longitude, inclination, n_sat, min_elev, firstID, env):
        self.ID = ID 								# A unique ID given to every orbital plane = index in Orbital_planes, string
        self.h = h									# Altitude of deployment
        self.longitude = longitude					# Longitude angle where is intersects equator [radians]
        self.inclination = math.pi/2 - inclination	# Inclination of the orbit form [radians]
        self.n_sat = n_sat							# Number of satellites in plane
        self.period = 2 * math.pi * math.sqrt((self.h+Re)**3/(G*Me))	# Orbital period of the satellites in seconds
        self.v = 2*math.pi * (h + Re) / self.period						# Orbital velocity of the satellites in m/s
        self.min_elev = math.radians(min_elev)							# Minimum elevation angle for ground comm.
        self.max_alpha = math.acos(Re*math.cos(self.min_elev)/(self.h+Re))-self.min_elev	# Maximum angle at the center of the Earth w.r.t. yaw
        self.max_beta  = math.pi/2-self.max_alpha-self.min_elev								# Maximum angle at the satellite w.r.t. yaw
        self.max_distance_2_ground = Re*math.sin(self.max_alpha)/math.sin(self.max_beta)	# Maximum distance to a servable ground station
        
        # Adding satellites
        self.first_sat_ID = firstID # Unique ID of the first satellite in the orbital plane
        
        self.sats = []              # List of satellites in the orbital plane
        for i in range(n_sat):
            self.sats.append(Satellite(self.first_sat_ID + str(i), int(self.ID), int(i), self.h, self.longitude, self.inclination, self.n_sat, env))
        
        self.last_sat_ID = self.first_sat_ID + str(len(self.sats) - 1) # Unique ID of the last satellite in the orbital plane
        
    def __repr__(self):
        return '\nID = {}\n altitude= {} km\n longitude= {} deg\n inclination= {} deg\n number of satellites= {}\n period= {} hours\n satellite speed= {} km/s'.format(
            self.ID,
            self.h/1e3,
            '%.2f' % math.degrees(self.longitude),
            '%.2f' % math.degrees(self.inclination),
            '%.2f' % self.n_sat,
            '%.2f' % (self.period/3600),
            '%.2f' % (self.v/1e3))

    def rotate(self, delta_t, env_time):
        """
        Rotates the orbit according to the elapsed time by adjusting the longitude. The amount the longitude is adjusted
        is based on the fraction the elapsed time makes up of the time it takes the Earth to complete a full rotation.
        """
        # Change in longitude and phi due to Earth's rotation
        self.longitude = self.longitude + 2*math.pi*delta_t/Te 
        self.longitude = self.longitude % (2*math.pi)
        # Rotating every satellite in the orbital plane
        for sat in self.sats:
            sat.rotate(delta_t, self.longitude, self.period)
            sat.update_beam_loads(env_time)  # Update beam loads based on time

class Beam:
    def __init__(self, center_lat, center_lon, width_deg, height_deg, 
                 load=1, snr=0, id=None, constellation = 'OneWeb'):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.width_deg = width_deg
        self.height_deg = height_deg
        self.load = load 
        self.snr = snr 
        self.id = id
        self.load_amplitude = 0.5 
        self.load_frequency = 2 * math.pi / 900 
        self.load_phase = random.uniform(0, 2 * math.pi)
        self.base_load = 0.5 
        
        self._calculate_ellipse_parameters()

        if constellation == 'OneWeb': 
            self.max_capacity = 0.1 #Gbps
            self.capacity = self.max_capacity * self.load
            self.bw = 250e6 #Hz 
            self.frequency = 15 #GHz 
            self.Pt = 40 #dBm
            self.Gt = 30 #dBi
        
    def _calculate_ellipse_parameters(self):
        """Calculates semi_major_axis, semi_minor_axis, and orientation_angle."""
        semi_axis_x_half = abs(self.width_deg / 2.0)
        semi_axis_y_half = abs(self.height_deg / 2.0)

        self.semi_major_axis = max(semi_axis_x_half, semi_axis_y_half)
        self.semi_minor_axis = min(semi_axis_x_half, semi_axis_y_half)

        self.orientation_angle = 0.0 if semi_axis_x_half >= semi_axis_y_half else 90.0

    def get_footprint_eclipse(self, num_segments=100):
        """Generates a Shapely Polygon representing the elliptical footprint."""
        circle_points = [(math.cos(2 * math.pi * i / num_segments), math.sin(2 * math.pi * i / num_segments)) for i in range(num_segments)]
        ellipse_base = Polygon(circle_points)
        
        if self.orientation_angle == 0.0:
            scaled_ellipse = scale(ellipse_base, self.semi_major_axis, self.semi_minor_axis, origin=(0,0))
        else:
            scaled_ellipse = scale(ellipse_base, self.semi_minor_axis, self.semi_major_axis, origin=(0,0))
            scaled_ellipse = rotate(scaled_ellipse, self.orientation_angle, origin=(0,0))

        return translate(scaled_ellipse, xoff=self.center_lon, yoff=self.center_lat)

    def get_load_at_time(self, time_seconds):
        """Calculate the load at a given time using a sinusoidal function."""
        sinusoidal_component = self.load_amplitude * math.sin(self.load_frequency * time_seconds + self.load_phase)
        return self.base_load + sinusoidal_component
    
    def update_load(self, time_seconds):
        """Update the beam's current load and capacity."""
        self.load = self.get_load_at_time(time_seconds)
        self.capacity = self.max_capacity * self.load

class Satellite:
    def __init__(self, ID, in_plane, i_in_plane, h, longitude, inclination, n_sat, env, quota = 500, power = 10):
        self.ID = ID
        self.in_plane = in_plane
        self.i_in_plane = i_in_plane
        self.h = h
        self.constellationType = "OneWeb"
        self.r = Re + self.h 
        self.theta = 2 * math.pi * self.i_in_plane / n_sat
        self.phi = longitude
        self.inclination = inclination
        self.x = self.r * (math.sin(self.theta)*math.cos(self.phi) - math.cos(self.theta)*math.sin(self.phi)*math.sin(self.inclination))
        self.y = self.r * (math.sin(self.theta)*math.sin(self.phi) + math.cos(self.theta)*math.cos(self.phi)*math.sin(self.inclination))
        self.z = self.r * math.cos(self.theta)*math.cos(self.inclination)
        self.polar_angle = self.theta
        self.latitude = math.asin(self.z/self.r)
        if self.x > 0: self.longitude = math.atan(self.y/self.x)
        elif self.x < 0 and self.y >= 0: self.longitude = math.pi + math.atan(self.y/self.x)
        elif self.x < 0 and self.y < 0: self.longitude = math.atan(self.y/self.x) - math.pi
        else: self.longitude = math.pi/2 if self.y > 0 else -math.pi/2
        self.env = env
        self.beams = []
        self.init_beams()

    def init_beams(self):
        """Initializes the satellite's beams based on its constellation type."""
        side_km = 1310
        n_beams = 16
        beam_height_km = side_km / n_beams
        deg_per_km = 1 / 111.0
        beam_width_deg = side_km * deg_per_km
        beam_height_deg = beam_height_km * deg_per_km
        sat_lat = math.degrees(self.latitude)
        sat_lon = math.degrees(self.longitude)
        top_lat = sat_lat + (side_km / 2) * deg_per_km
        for i in range(n_beams):
            center_lat = top_lat - (i + 0.5) * beam_height_deg
            beam_id = f"{self.ID}_beam_{i+1}"
            self.beams.append(Beam(center_lat, sat_lon, beam_width_deg, beam_height_deg, id=beam_id))

    def update_beams(self):
        """Updates beam positions based on the satellite's current location."""
        side_km = 1310
        n_beams = 16
        beam_height_km = side_km / n_beams
        deg_per_km = 1 / 111.0
        beam_width_deg = side_km * deg_per_km
        beam_height_deg = beam_height_km * deg_per_km
        sat_lat = math.degrees(self.latitude)
        sat_lon = math.degrees(self.longitude)
        top_lat = sat_lat + (side_km / 2) * deg_per_km
        for i, beam in enumerate(self.beams):
            beam.center_lat = top_lat - (i + 0.5) * beam_height_deg
            beam.center_lon = sat_lon
            beam._calculate_ellipse_parameters()

    def update_beam_loads(self, time_seconds):
        """Updates the load for all beams on this satellite."""
        for beam in self.beams:
            beam.update_load(time_seconds)

    def rotate(self, delta_t, longitude, period):
        """Rotates the satellite and updates its coordinates."""
        self.phi = longitude
        self.theta = (self.theta + 2 * math.pi * delta_t / period) % (2 * math.pi)
        self.x = self.r * (math.sin(self.theta)*math.cos(self.phi) - math.cos(self.theta)*math.sin(self.phi)*math.sin(self.inclination))
        self.y = self.r * (math.sin(self.theta)*math.sin(self.phi) + math.cos(self.theta)*math.cos(self.phi)*math.sin(self.inclination))
        self.z = self.r * math.cos(self.theta)*math.cos(self.inclination)
        self.polar_angle = self.theta
        self.latitude = math.asin(self.z/self.r)
        if self.x > 0: self.longitude = math.atan(self.y/self.x)
        elif self.x < 0 and self.y >= 0: self.longitude = math.pi + math.atan(self.y/self.x)
        elif self.x < 0 and self.y < 0: self.longitude = math.atan(self.y/self.x) - math.pi
        else: self.longitude = math.pi/2 if self.y > 0 else -math.pi/2
        self.update_beams()

class AircraftSNR: 
    """Aircraft class implementing the original SNR-based beam selection."""
    def __init__(self, env, aircraft_id, start_lat, start_lon, height, speed_kmph, direction_deg, update_interval=1):
        self.env = env
        self.id = aircraft_id
        self.latitude = start_lat
        self.longitude = start_lon
        self.height = height
        self.Gr = 5 #dBi
        self.connected_satellite = None
        self.connected_beam = None
        self.current_snr = 0
        self.current_latency = 0
        self.handover_count = 0
        self.total_allocated_bandwidth = 0.0
        self.allocation_ratios = []
        self.demand = 0
        print(f"Aircraft {self.id} initialized at ({self.latitude:.2f}, {self.longitude:.2f}) with SNR-based logic.")

    def calculate_snr(self, beam, distance_km):
        """Calculates SNR based on beam and aircraft parameters."""
        T = 290
        Pt_dBm = beam.Pt
        Gt_dBi = beam.Gt
        Gr_dBi = self.Gr
        f_Hz = beam.frequency * 1e9
        B_Hz = beam.bw
        Pt_dBW = Pt_dBm - 30
        d_m = distance_km * 1000
        if d_m == 0: return float('inf')
        FSPL_dB = 20 * math.log10(d_m) + 20 * math.log10(f_Hz) - 147.55
        Pr_dBW = Pt_dBW + Gt_dBi + Gr_dBi - FSPL_dB
        N_dBW = 10 * math.log10(k * T * B_Hz)
        return Pr_dBW - N_dBW

    def scan_nearby_fast(self, earth, threshold_km=1500):
        """Efficiently scans for the best beam based on SNR using a KDTree."""
        beam_coords = []
        beam_refs = []
        for plane in earth.LEO:
            for sat in plane.sats:
                for beam in sat.beams:
                    beam_coords.append([beam.center_lat, beam.center_lon])
                    beam_refs.append((sat, beam))
        
        if not beam_refs:
            return None, None, -np.inf

        tree = KDTree(np.array(beam_coords))
        threshold_deg = threshold_km / 111.0
        idxs = tree.query_ball_point([self.latitude, self.longitude], r=threshold_deg)

        best_snr = -np.inf
        best_candidate_beam = None
        best_candidate_sat = None

        for idx in idxs:
            sat, beam = beam_refs[idx]
            aircraft_point = Point(self.longitude, self.latitude)
            if beam.get_footprint_eclipse().contains(aircraft_point):
                dist_3d = self._calculate_3d_distance(sat)
                snr = self.calculate_snr(beam, dist_3d / 1000)
                if snr > best_snr:
                    best_snr = snr
                    best_candidate_beam = beam
                    best_candidate_sat = sat
        return best_candidate_sat, best_candidate_beam, best_snr

    def get_demand(self, deltaT):
        """Calculates the aircraft's data demand for a timestep."""
        num_users = random.randint(1, 10)
        demand_per_user = [random.uniform(0, 25) for _ in range(num_users)]
        total_demand_mbps = sum(demand_per_user)
        return (total_demand_mbps * deltaT) / 8

    def set_demand(self, demand_value):
        """Sets the aircraft's data demand."""
        self.demand = demand_value
    
    
    def allocation_ratio(self, deltaT):
        """Calculates the ratio of allocated throughput to demand."""
        # self.demand = self.get_demand(deltaT) # REMOVED THIS LINE
        beam_capacity_MB = (self.connected_beam.capacity * 125 * deltaT) if self.connected_beam else 0
        allocated = min(self.demand, beam_capacity_MB)
        ratio = allocated / self.demand if self.demand > 0 else 0
        self.total_allocated_bandwidth += allocated
        self.allocation_ratios.append(ratio)
        return ratio, allocated, self.demand, beam_capacity_MB

    def move_and_connect_aircraft(self, earth_instance):
        """Connects to the best beam based on SNR."""
        best_candidate_sat, best_candidate_beam, best_snr = self.scan_nearby_fast(earth_instance)

        if best_candidate_beam and best_candidate_beam != self.connected_beam:
            if self.connected_beam:
                self.handover_count += 1
            self.connected_beam = best_candidate_beam
            self.connected_satellite = best_candidate_sat
            self.current_snr = best_snr
            self.current_latency = self._calculate_latency(self.connected_satellite)
        elif not best_candidate_beam and self.connected_beam:
            self.connected_beam = None
            self.connected_satellite = None
            self.current_snr = 0
            self.current_latency = 0
        elif best_candidate_beam and best_candidate_beam == self.connected_beam:
            self.current_snr = best_snr
            self.current_latency = self._calculate_latency(self.connected_satellite)

    def _calculate_3d_distance(self, satellite):
        """Calculates the 3D slant range from aircraft to satellite."""
        dist_2d_km = geopy.distance.geodesic((self.latitude, self.longitude), 
                                            (math.degrees(satellite.latitude), 
                                            math.degrees(satellite.longitude))).km
        altitude_diff_m = satellite.h - self.height
        return math.sqrt((dist_2d_km * 1000)**2 + altitude_diff_m**2)

    def _calculate_latency(self, satellite):
        """Calculates propagation latency to a satellite."""
        if satellite:
            return self._calculate_3d_distance(satellite) / Vc
        return 0

class AircraftDQN: 
    """Aircraft class implementing the DQN-based beam selection."""
    def __init__(self, env, aircraft_id, start_lat, start_lon, height, speed_kmph, direction_deg, update_interval=1):
        self.env = env
        self.id = aircraft_id
        self.latitude = start_lat
        self.longitude = start_lon
        self.height = height
        self.Gr = 5 #dBi
        self.connected_satellite = None
        self.connected_beam = None
        self.current_snr = 0
        self.current_latency = 0
        self.handover_count = 0
        self.total_allocated_bandwidth = 0.0
        self.allocation_ratios = []
        self.total_reward = 0
        self.demand = 0

        # DQN Agent Initialization
        self.N_CANDIDATE_SATELLITES = 4  # Number of nearest satellites to consider
        self.BEAMS_PER_SATELLITE = 16    # Number of beams per satellite
        self.N_CANDIDATE_BEAMS = self.N_CANDIDATE_SATELLITES * self.BEAMS_PER_SATELLITE # Total candidate beams
        self.N_FEATURES_PER_BEAM = 5     # (rel_lat, rel_lon, snr, capacity, latency)

        state_size = self.N_CANDIDATE_BEAMS * self.N_FEATURES_PER_BEAM
        action_size = self.N_CANDIDATE_BEAMS
        self.agent = DQNAgent(state_size, action_size)
        self.state = np.zeros((1, state_size))
        self.last_action = None

        print(f"Aircraft {self.id} initialized at ({self.latitude:.2f}, {self.longitude:.2f}) with DQN agent.")

    def calculate_snr(self, beam, distance_km):
        """Calculates SNR based on beam and aircraft parameters."""
        T = 290
        Pt_dBm = beam.Pt
        Gt_dBi = beam.Gt
        Gr_dBi = self.Gr
        f_Hz = beam.frequency * 1e9
        B_Hz = beam.bw
        Pt_dBW = Pt_dBm - 30
        d_m = distance_km * 1000
        if d_m == 0: return float('inf')
        FSPL_dB = 20 * math.log10(d_m) + 20 * math.log10(f_Hz) - 147.55
        Pr_dBW = Pt_dBW + Gt_dBi + Gr_dBi - FSPL_dB
        N_dBW = 10 * math.log10(k * T * B_Hz)
        return Pr_dBW - N_dBW

    def _calculate_3d_distance(self, satellite):
        """Calculates the 3D slant range from aircraft to satellite."""
        dist_2d_km = geopy.distance.geodesic((self.latitude, self.longitude), 
                                            (math.degrees(satellite.latitude), 
                                            math.degrees(satellite.longitude))).km
        altitude_diff_m = satellite.h - self.height
        return math.sqrt((dist_2d_km * 1000)**2 + altitude_diff_m**2)

    def _calculate_latency(self, satellite):
        """Calculates propagation latency to a satellite."""
        if satellite:
            return self._calculate_3d_distance(satellite) / Vc
        return 0

    def observe_state_and_candidates(self, earth):
        """
        Observes the environment to create a state vector for the DQN agent by
        considering all beams from the N_CANDIDATE_SATELLITES nearest satellites.
        """
        # 1. Find all satellites and calculate their distances from the aircraft.
        all_satellites_with_dist = []
        for plane in earth.LEO:
            for sat in plane.sats:
                dist_3d = self._calculate_3d_distance(sat)
                all_satellites_with_dist.append((dist_3d, sat))
        
        # 2. Sort satellites by distance and select the N_CANDIDATE_SATELLITES nearest ones.
        all_satellites_with_dist.sort(key=lambda x: x[0])
        nearest_satellites = [sat for dist, sat in all_satellites_with_dist[:self.N_CANDIDATE_SATELLITES]]

        # 3. Collect information from ALL beams of these nearest satellites.
        # This guarantees a consistent number of potential candidates.
        candidate_beams_info = []
        for sat in nearest_satellites:
            dist_3d_to_sat = self._calculate_3d_distance(sat)
            latency = self._calculate_latency(sat)
            for beam in sat.beams:
                # Calculate metrics for every beam, regardless of footprint containment.
                # The agent will learn that beams with very low SNR are poor choices.
                snr = self.calculate_snr(beam, dist_3d_to_sat / 1000)
                candidate_beams_info.append({
                    "beam": beam,
                    "snr": snr,
                    "latency": latency,
                    "rel_lat": beam.center_lat - self.latitude,
                    "rel_lon": beam.center_lon - self.longitude
                })

        # 4. Construct the state vector and the list of candidate beams for the action space.
        state_vector = []
        final_candidate_beams = [] # This will be the list of beam objects for the agent to choose from.
        
        for i in range(self.N_CANDIDATE_BEAMS): # Should be 64
            if i < len(candidate_beams_info):
                info = candidate_beams_info[i]
                final_candidate_beams.append(info['beam'])
                # Append features to the state vector
                state_vector.extend([
                    info['rel_lat'],
                    info['rel_lon'],
                    info['snr'],
                    info['beam'].capacity,
                    info['latency'] * 1e3
                ])
            else:
                # Pad with dummy/neutral values if total beams are less than N_CANDIDATE_BEAMS.
                dummy_beam = Beam(center_lat=0, center_lon=0, width_deg=0, height_deg=0, 
                                  load=0, snr=-np.inf, id=f"dummy_beam_{i}")
                final_candidate_beams.append(dummy_beam)
                # Append corresponding dummy features to the state vector
                state_vector.extend([0, 0, -np.inf, 0, 9999])

        # 5. Return the fixed-size list of beam objects and the state vector.
        return final_candidate_beams, np.array(state_vector).reshape(1, -1)


    def get_demand(self, deltaT):
        """Calculates the aircraft's data demand for a timestep."""
        num_users = random.randint(1, 10)
        demand_per_user = [random.uniform(0, 25) for _ in range(num_users)]
        total_demand_mbps = sum(demand_per_user)
        return (total_demand_mbps * deltaT) / 8
    
    def set_demand(self, demand_value):
        """Sets the aircraft's data demand."""
        self.demand = demand_value

    def step(self, earth, deltaT):
        """Executes one step of the agent's interaction with the environment."""
        candidate_beams, new_state = self.observe_state_and_candidates(earth)
        action = self.agent.act(self.state)

        handover = False
        # Check if the chosen action corresponds to a valid (non-dummy) beam
        if action < len(candidate_beams) and candidate_beams[action].snr != -np.inf:
            chosen_beam = candidate_beams[action]
            if self.connected_beam and self.connected_beam.id != chosen_beam.id:
                handover = True
                self.handover_count += 1
            
            self.connected_beam = chosen_beam
            # Find the satellite associated with the chosen beam
            self.connected_satellite = next((sat for p in earth.LEO for sat in p.sats if chosen_beam in sat.beams), None)
            
            # Recalculate SNR and latency for the chosen beam
            if self.connected_satellite:
                self.current_snr = self.calculate_snr(self.connected_beam, self._calculate_3d_distance(self.connected_satellite)/1000)
                self.current_latency = self._calculate_latency(self.connected_satellite)
            else: # Should not happen if chosen_beam is valid and associated with a sat
                self.current_snr = 0
                self.current_latency = 0
        else: # Action chose a dummy beam or an invalid index
            self.connected_beam = None
            self.connected_satellite = None
            self.current_snr = 0
            self.current_latency = 0

        beam_capacity_MB = (self.connected_beam.capacity * 125 * deltaT) if self.connected_beam else 0
        allocated = min(self.demand, beam_capacity_MB)
        alloc_ratio = allocated / self.demand if self.demand > 0 else 0
        
        reward = (alloc_ratio * 100) - (self.current_latency * 1000) - (20 if handover else 0)
        if not self.connected_beam:
            reward -= 50 # Penalize heavily for no connection
        
        self.total_reward += reward
        self.total_allocated_bandwidth += allocated
        self.allocation_ratios.append(alloc_ratio)
        
        self.agent.memory.remember(self.state, action, reward, new_state, False)
        self.state = new_state
        self.agent.replay(32)
        return alloc_ratio, allocated, self.demand, beam_capacity_MB

class Earth:
    def __init__(self, env, img_path, constellation, aircrafts, inputParams, deltaT, window=None):
        pop_count_data = Image.open(img_path)
        [self.total_x, self.total_y] = [1920, 906]
        self.constellationType = constellation
        self.LEO = create_Constellation(constellation, env)
        self.aircrafts = aircrafts
        self.moveConstellation = env.process(self.moveConstellation(env, deltaT))
        self.step_aircraft = env.process(self.step_aircraft(env, deltaT))

    def moveConstellation(self, env, deltaT=10):
        """Simpy process to move the satellite constellation."""
        while True:
            for constellation in self.LEO:
                constellation.rotate(deltaT, env.now)
            yield env.timeout(deltaT)

    def _generate_common_demand(self, deltaT):
        """Generates a common data demand for all aircraft at a given timestep."""
        num_users = random.randint(1, 10)
        demand_per_user = [random.uniform(0, 25) for _ in range(num_users)]
        total_demand_mbps = sum(demand_per_user)
        return (total_demand_mbps * deltaT) / 8
    

    def step_aircraft(self, env, deltaT=10):
        """
        SimPy process: At each interval, all aircraft scan, update demand, and calculate allocation.
        """
        while True:
            current_demand = self._generate_common_demand(deltaT) # Generate common demand
            for ac in self.aircrafts:
                ac.set_demand(current_demand) # Set the same demand for both aircraft
                if isinstance(ac, AircraftSNR):
                    ac.move_and_connect_aircraft(self)
                    ratio, allocated, demand, beam_capacity_MB = ac.allocation_ratio(deltaT)
                    print(f"\n[SimTime {env.now:.2f}] Aircraft {ac.id} (SNR-based):")
                    print(f"  Allocation ratio: {ratio:.2f} | Allocated: {allocated:.2f} MB | Demand: {demand:.2f} MB | Beam cap: {beam_capacity_MB:.2f} MB")
                    if ac.connected_beam:
                        print(f"  Connected to {ac.connected_beam.id}. SNR: {ac.current_snr:.2f} dB, Latency: {ac.current_latency*1e3:.2f} ms")
                    else:
                        print("  No connection.")
                elif isinstance(ac, AircraftDQN):
                    ratio, allocated, demand, beam_capacity_MB = ac.step(self, deltaT)
                    print(f"\n[SimTime {env.now:.2f}] Aircraft {ac.id} (DQN-based):")
                    print(f"  Allocation ratio: {ratio:.2f} | Allocated: {allocated:.2f} MB | Demand: {demand:.2f} MB | Beam cap: {beam_capacity_MB:.2f} MB")
                    if ac.connected_beam:
                        print(f"  Connected to {ac.connected_beam.id}. SNR: {ac.current_snr:.2f} dB, Latency: {ac.current_latency*1e3:.2f} ms")
                    else:
                        print("  No connection.")
                    print(f"  total reward is {ac.total_reward}")
            yield env.timeout(deltaT)

    def plotMap(self, plotSat=True, plotBeams=True, plotAircrafts=True, aircrafts=None, selected_beam_id=None):
        """Plots the world map with satellites, beams, and aircraft."""
        print("Plotting map...")
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_global()
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(self.LEO)))
        
        if plotSat:
            for plane, c in zip(self.LEO, colors):
                for sat in plane.sats:
                    ax.scatter(math.degrees(sat.longitude), math.degrees(sat.latitude), color=c, s=10, transform=ccrs.PlateCarree())

        if plotBeams:
            for plane in self.LEO:
                for sat in plane.sats:
                    for beam in sat.beams:
                        is_selected = selected_beam_id and beam.id == selected_beam_id
                        edge_color, line_width, alpha_val, z_order = ('red', 1.5, 1.0, 10) if is_selected else ('gray', 0.2, 0.3, 2)
                        feature = ShapelyFeature([beam.get_footprint_eclipse()], ccrs.PlateCarree(), edgecolor=edge_color, facecolor='none', linewidth=line_width, alpha=alpha_val)
                        ax.add_feature(feature, zorder=z_order)
        
        if plotAircrafts and aircrafts:
            ax.scatter([ac.longitude for ac in aircrafts], [ac.latitude for ac in aircrafts], color='black', marker='x', s=50, transform=ccrs.PlateCarree(), label='Aircraft', zorder=20)
            for aircraft in aircrafts:
                if aircraft.connected_satellite:
                    line_lons = [aircraft.longitude, math.degrees(aircraft.connected_satellite.longitude)]
                    line_lats = [aircraft.latitude, math.degrees(aircraft.connected_satellite.latitude)]
                    ax.plot(line_lons, line_lats, color='green', linewidth=1.0, linestyle='--', transform=ccrs.Geodetic(), zorder=15)

    def save_plot_at_intervals(self, env, interval=1, aircrafts_dqn=None):
        """Saves map plots at regular intervals, highlighting the DQN aircraft's connection."""
        img_count = 0
        if not os.path.exists("simulationImages_DQN"):
            os.makedirs("simulationImages_DQN")
        while True:
            print(f"\nSaving plot {img_count} at simulation time {env.now}")
            selected_beam_id = aircrafts_dqn[0].connected_beam.id if aircrafts_dqn and aircrafts_dqn[0].connected_beam else None
            self.plotMap(plotSat=True, plotBeams=True, plotAircrafts=True, aircrafts=aircrafts_dqn, selected_beam_id=selected_beam_id)
            plt.savefig(f"simulationImages_DQN/sat_positions_{img_count}.png")
            plt.close()
            img_count += 1
            yield env.timeout(interval)
    
    def save_plot_at_intervals_snr(self, env, interval=1, aircrafts_snr=None):
        """Saves map plots at regular intervals, highlighting the DQN aircraft's connection."""
        img_count2 = 0
        if not os.path.exists("simulationImages_SNR"):
            os.makedirs("simulationImages_SNR")
        while True:
            print(f"\nSaving plot {img_count2} at simulation time {env.now}")
            selected_beam_id = aircrafts_snr[0].connected_beam.id if aircrafts_snr and aircrafts_snr[0].connected_beam else None
            self.plotMap(plotSat=True, plotBeams=True, plotAircrafts=True, aircrafts=aircrafts_snr, selected_beam_id=selected_beam_id)
            plt.savefig(f"simulationImages_SNR/sat_positions_{img_count2}.png")
            plt.close()
            img_count2 += 1
            yield env.timeout(interval)

###############################################################################
###############################    Functions    ###############################
###############################################################################

def initialize(env, img_path, inputParams, movementTime):
    """Initializes the Earth, constellation, and both types of aircraft."""
    constellationType = inputParams['Constellation'][0]
    aircraft_snr = AircraftSNR(env, "A-380-SNR", start_lat=37.77, start_lon=-122.41, height=10000, speed_kmph=800, direction_deg=345)
    aircraft_dqn = AircraftDQN(env, "A-380-DQN", start_lat=37.77, start_lon=-122.41, height=10000, speed_kmph=800, direction_deg=345)
    aircrafts = [aircraft_snr, aircraft_dqn]
    earth = Earth(env, img_path, constellationType, aircrafts, inputParams, movementTime)
    print("Initialized Earth with both SNR and DQN Aircraft for comparison.")
    return earth

def create_Constellation(specific_constellation, env):
    """Creates a satellite constellation based on predefined parameters."""
    if specific_constellation == "OneWeb":
        print("Using OneWeb constellation design")
        P, N, height, inclination_angle = 18, 648, 1200e3, 86.4
    else: # Default to a smaller test constellation
        print("Using small walker Star constellation")
        P, N, height, inclination_angle = 3, 12, 1000e3, 53
    
    N_p = int(N / P)
    min_elevation_angle = 30
    distribution_angle = math.pi
    orbital_planes = []
    for i in range(P):
        orbital_planes.append(OrbitalPlane(str(i), height, i * distribution_angle / P, math.radians(inclination_angle), N_p,
                                           min_elevation_angle, str(i) + '_', env))
    return orbital_planes

def main():
    """Main function to set up and run the comparative simulation."""
    if not os.path.exists("input.csv"):
        with open("input.csv", "w") as f:
            f.write("Test length,Constellation\n")
            f.write("100,OneWeb\n")

    inputParams = pd.read_csv("input.csv")
    testLength = inputParams['Test length'][0]
    constellation_name = inputParams['Constellation'][0]
    movementTime = 20

    print(f"Constellation: {constellation_name}")
    print(f"Simulation test length: {testLength} seconds")

    env = simpy.Environment()
    
    img_path = "PopMap_500.png"
    if not os.path.exists(img_path):
        Image.new('L', (500, 250)).save(img_path)

    earth_instance = initialize(env, img_path, inputParams, movementTime)
    all_aircrafts = earth_instance.aircrafts

    # Pass only the DQN aircraft to the plotting function for highlighting its connection
    snr_aircrafts = [ac for ac in all_aircrafts if isinstance(ac, AircraftSNR)]
    env.process(earth_instance.save_plot_at_intervals_snr(env, interval=movementTime, aircrafts_snr=snr_aircrafts))
    dqn_aircrafts = [ac for ac in all_aircrafts if isinstance(ac, AircraftDQN)]
    env.process(earth_instance.save_plot_at_intervals(env, interval=movementTime, aircrafts_dqn=dqn_aircrafts))
    env.process(simProgress(testLength, env))
    
    startTime = time.time()
    env.run(until=testLength)
    timeToSim = time.time() - startTime

    for aircraft in all_aircrafts:
        if isinstance(aircraft, AircraftSNR):
            print("\n\n--- SNR-based Simulation Summary ---")
            print(f"Total simulation run time: {timeToSim:.2f} seconds")
            print(f"Aircraft '{aircraft.id}' total handovers: {aircraft.handover_count}")
            if aircraft.connected_beam:
                print(f"Aircraft '{aircraft.id}' final connected beam: {aircraft.connected_beam.id}")
                print(f"Aircraft '{aircraft.id}' final SNR: {aircraft.current_snr:.2f} dB")
                print(f"Aircraft '{aircraft.id}' total allocated BW: {aircraft.total_allocated_bandwidth:.2f} MB")
                if aircraft.allocation_ratios:
                    print(f"Aircraft '{aircraft.id}' Average Allocation to demand: {sum(aircraft.allocation_ratios)/len(aircraft.allocation_ratios):.2f}")
                print(f"Aircraft '{aircraft.id}' final Latency: {aircraft.current_latency*1e3:.2f} ms")
            else:
                print(f"Aircraft '{aircraft.id}' ended the simulation with no connection.")
        
        elif isinstance(aircraft, AircraftDQN):
            print("\n\n--- DQN Simulation Summary ---")
            print(f"Total simulation run time: {timeToSim:.2f} seconds")
            print(f"Aircraft '{aircraft.id}' total handovers: {aircraft.handover_count}")
            print(f"Aircraft '{aircraft.id}' total reward: {aircraft.total_reward:.2f}")
            if aircraft.allocation_ratios:
                 print(f"Aircraft '{aircraft.id}' Average Allocation to demand: {sum(aircraft.allocation_ratios)/len(aircraft.allocation_ratios):.2f}")
            if aircraft.connected_beam:
                print(f"Aircraft '{aircraft.id}' final connected beam: {aircraft.connected_beam.id}")
                print(f"Aircraft '{aircraft.id}' final SNR: {aircraft.current_snr:.2f} dB")
            else:
                print(f"Aircraft '{aircraft.id}' ended the simulation with no connection.")
            
            aircraft.agent.model.save(f"dqn_beam_selection_model_{aircraft.id}.h5")
            print(f"Saved trained model for aircraft {aircraft.id}")

if __name__ == '__main__':
    main()