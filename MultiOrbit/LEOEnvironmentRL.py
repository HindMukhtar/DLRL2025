import time
import pandas as pd
import math
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
import matplotlib
import simpy
import numba
import networkx as nx
from PIL import Image
from scipy.optimize import linear_sum_assignment
import os
import cartopy.crs as ccrs
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from cartopy.feature import ShapelyFeature
import matplotlib.patches as mpatches
from shapely.geometry import box
from shapely.affinity import scale, rotate, translate
from scipy.spatial import KDTree
import random
import csv

# Teleport gateway location (Ashburn, VA)
GATEWAY_LAT = 39.0438
GATEWAY_LON = -77.4874


###############################################################################
#################################    Simpy    #################################
###############################################################################

receivedDataBlocks = []
createdBlocks = []
seed = np.random.seed(1)

upGSLRates = []
downGSLRates = []
interRates = []
intraRate = []

def simProgress(simTimelimit, env):
    timeSteps = 100
    timeStepSize = simTimelimit/timeSteps
    progress = 1
    startTime = time.time()
    yield env.timeout(timeStepSize)
    while True:
        elapsedTime = time.time() - startTime
        estimatedTimeRemaining = elapsedTime * (timeSteps/progress) - elapsedTime
        print("Simulation progress: {}% Estimated time remaining: {} seconds Current simulation time: {}".format(progress, int(estimatedTimeRemaining), env.now), end='\r')
        yield env.timeout(timeStepSize)
        progress += 1


###############################################################################
###############################    Constants    ###############################
###############################################################################

Re  = 6378e3            # Radius of the earth [m]
G    = 6.67259e-11      # Universal gravitational constant [m^3/kg s^2]
Me  = 5.9736e24         # Mass of the earth
Te  = 86164.28450576939 # Time required by Earth for 1 rotation
Vc  = 299792458         # Speed of light [m/s]
k   = 1.38e-23          # Boltzmann's constant
eff = 0.55              # Efficiency of the parabolic antenna

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
    def __init__(self, ID, h, longitude, inclination, n_sat, min_elev, firstID, env, constellation_type="OneWeb"):
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
            self.sats.append(
                Satellite(
                    self.first_sat_ID + str(i),
                    int(self.ID),
                    int(i),
                    self.h,
                    self.longitude,
                    self.inclination,
                    self.n_sat,
                    env,
                    constellation_type=constellation_type,
                )
            )
        
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
    scenario = None
    _pop_density = None
    _pop_shape = None
    start_utc_hour = 0.0
    load_cycle_s = 5400.0
    def __init__(self, center_lat, center_lon, width_deg, height_deg, 
                 load=1, capacity=10, snr=0, id=None, constellation = 'OneWeb'): # Default capacity added
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.width_deg = width_deg
        self.height_deg = height_deg
        self.load = load # Current number of users connected
        self.snr = snr # Base SNR, will be adjusted for user position
        self.id = id  # Optional: unique identifier for the beam
        self.load_amplitude = 0.5 
        self.load_frequency = 2 * math.pi / 100 
        self.load_phase = random.uniform(0, 2 * math.pi)
        self.base_load = 0.5
        self._last_load_time = None
        self.noise_state = 0.0
        self.noise_tau_s = 900.0
        self.noise_sigma = 0.08
        self.peak_local_hour = 18.0
        
        # Calculate ellipse parameters based on width_deg and height_deg
        self._calculate_ellipse_parameters()

        if constellation == 'OneWeb': 
            #self.max_capacity = 7.2 #Gbps 
            self.max_capacity = 0.48 #Gbps reduce for now as we only have a single agent 
            self.max_ds_speed = 150 #Mbps 
            self.max_us_speed = 30 #Mbps 
            self.max_latency = 70 #ms 
            self.bw = 250e6 #Hz 
            self.effective_bw = self.bw
            self.capacity = self.max_capacity
            self.frequency = 15 #GHz 
            self.Pt = 40 #dBm - transmit power 
            self.Gt = 30 #dBi - antenna gain 
        elif constellation == 'Intelsat':
            # Coarse GEO spot-beam profile for simulation experiments.
            self.max_capacity = 0.50 # Gbps
            self.max_ds_speed = 50 # Mbps
            self.max_us_speed = 20 # Mbps
            self.max_latency = 700 # ms
            self.bw = 200e6 # Hz
            self.effective_bw = self.bw
            self.capacity = self.max_capacity
            self.frequency = 12 # GHz
            # Typical Intelsat multi-spot EIRP range: 50.7..62.3 dBW -> midpoint 56.5 dBW.
            # EIRP(dBW) = Pt(dBW) + Gt(dBi). We fix Gt and derive Pt for consistency.
            self.eirp_dbw = (50.7 + 62.3) / 2.0
            self.Gt = 42.0 # dBi (assumed beam gain)
            self.Pt = self.eirp_dbw - self.Gt + 30.0 # dBm
        else:
            # Safe defaults for other constellation types.
            self.max_capacity = 0.30
            self.max_ds_speed = 75
            self.max_us_speed = 15
            self.max_latency = 300
            self.bw = 100e6
            self.effective_bw = self.bw
            self.capacity = self.max_capacity
            self.frequency = 15
            self.Pt = 40
            self.Gt = 30
        
    def _calculate_ellipse_parameters(self):
        """Calculates semi_major_axis, semi_minor_axis, and orientation_angle."""
        semi_axis_x_half = abs(self.width_deg / 2.0)
        semi_axis_y_half = abs(self.height_deg / 2.0)

        self.semi_major_axis = max(semi_axis_x_half, semi_axis_y_half)
        self.semi_minor_axis = min(semi_axis_x_half, semi_axis_y_half)

        if semi_axis_x_half >= semi_axis_y_half:
            self.orientation_angle = 0.0  # Major axis is along the x-direction (longitude)
        else:
            self.orientation_angle = 90.0 # Major axis is along the y-direction (latitude)

    def get_footprint(self):
        # This method is kept for compatibility but get_footprint_eclipse is preferred
        min_lon = self.center_lon - self.width_deg / 2
        max_lon = self.center_lon + self.width_deg / 2
        min_lat = self.center_lat - self.height_deg / 2
        max_lat = self.center_lat + self.height_deg / 2
        return box(min_lon, min_lat, max_lon, max_lat)
    
    def get_footprint_eclipse(self, num_segments=100):
        """
        Generates a Shapely Polygon representing the elliptical footprint.
        """
        circle_points = []
        for i in range(num_segments):
            angle = 2 * math.pi * i / num_segments
            x = math.cos(angle)
            y = math.sin(angle)
            circle_points.append((x, y))
        
        ellipse_base = Polygon(circle_points)
        
        if self.orientation_angle == 0.0:
            scaled_ellipse = scale(ellipse_base, self.semi_major_axis, self.semi_minor_axis, origin=(0,0))
        else: # self.orientation_angle == 90.0
            scaled_ellipse = scale(ellipse_base, self.semi_minor_axis, self.semi_major_axis, origin=(0,0))
            scaled_ellipse = rotate(scaled_ellipse, self.orientation_angle, origin=(0,0))

        final_ellipse = translate(scaled_ellipse, xoff=self.center_lon, yoff=self.center_lat)
        return final_ellipse

    @classmethod
    def _load_pop_map(cls):
        if cls._pop_density is not None:
            return
        pop_map_path = os.path.join(os.path.dirname(__file__), "PopMap_500.png")
        try:
            img = Image.open(pop_map_path).convert("L")
            arr = np.array(img, dtype=np.float32)
            max_val = np.max(arr)
            if max_val > 0:
                arr = arr / max_val
            cls._pop_density = arr
            cls._pop_shape = arr.shape
        except Exception:
            cls._pop_density = None
            cls._pop_shape = None

    def _pop_density_at(self, lat, lon):
        Beam._load_pop_map()
        if Beam._pop_density is None:
            return None
        height, width = Beam._pop_shape
        x = (lon + 180.0) / 360.0 * (width - 1)
        y = (90.0 - lat) / 180.0 * (height - 1)
        xi = int(min(max(round(x), 0), width - 1))
        yi = int(min(max(round(y), 0), height - 1))
        return float(Beam._pop_density[yi, xi])
    
    def _update_noise(self, dt_seconds):
        if dt_seconds <= 0:
            return
        decay = math.exp(-dt_seconds / self.noise_tau_s)
        std = self.noise_sigma * math.sqrt(1 - decay * decay)
        self.noise_state = self.noise_state * decay + random.gauss(0.0, std)

    def off_axis_gain_db(self, lat, lon, edge_loss_db=12.0):
        dx = lon - self.center_lon
        dy = lat - self.center_lat
        a = max(self.semi_major_axis, 1e-6)
        b = max(self.semi_minor_axis, 1e-6)
        u = (dx / a) ** 2 + (dy / b) ** 2  # u=1 at beam edge
        if u <= 1.0:
            return -edge_loss_db * u
        return -edge_loss_db * 1.5  # outside footprint, extra penalty
    
    # def get_load_at_time(self, time_seconds):
    #     """
    #     Calculate the load at a given time using a sinusoidal function.
        
    #     Args:
    #         time_seconds: Current simulation time in seconds
            
    #     Returns:
    #         Load as a percentage (0-100)
    #     """
    #     sinusoidal_component = self.load_amplitude * math.sin(self.load_frequency * time_seconds + self.load_phase)
    #     current_load = self.base_load + sinusoidal_component
        
    #     # Ensure load stays within reasonable bounds (0-100%)
    #     return current_load

    def get_load_at_time(self, time_seconds):
        """
        Calculate the load at a given time using population density and a full-cycle
        episode sinusoid (low -> high -> low), plus correlated noise.
        
        Args:
            time_seconds: Current simulation time in seconds
            
        Returns:
            Load as a percentage (0-100)
        """
        base_density = self._pop_density_at(self.center_lat, self.center_lon)
        if base_density is None:
            base_density = self.base_load

        cycle_s = max(1.0, Beam.load_cycle_s)
        # Deterministic per-beam phase offset (longitude + beam-id jitter) to avoid full synchronization.
        lon_phase = (self.center_lon + 180.0) / 360.0
        id_hash = sum(ord(c) for c in str(self.id)) if self.id is not None else 0
        id_phase = (id_hash % 997) / 997.0
        beam_phase = (lon_phase + 0.1 * id_phase) % 1.0
        phase = 0.5 + 0.5 * math.sin(2 * math.pi * ((time_seconds / cycle_s) + beam_phase) - math.pi / 2.0)
        load_min = 0.05 + 0.15 * base_density
        # Allow slight overshoot above 1.0 so clipping produces full saturation windows.
        load_max = 0.82 + 0.28 * base_density
        current_load = load_min + (load_max - load_min) * phase + self.noise_state
        
        # Ensure load stays within reasonable bounds (0-100%)
        return min(max(current_load, 0.0), 1.0)
    
    def update_load(self, time_seconds):
        """
        Update the beam's current load based on the spatio-temporal model.
        
        Args:
            time_seconds: Current simulation time in seconds
        """
        if self._last_load_time is None:
            self._last_load_time = time_seconds
        else:
            dt = max(0.0, time_seconds - self._last_load_time)
            self._last_load_time = time_seconds
            self._update_noise(dt)
        self.load = self.get_load_at_time(time_seconds)
        self.effective_bw = self.bw * (1 - self.load)
        self.capacity = self.max_capacity

class Satellite:
    def __init__(self, ID, in_plane, i_in_plane, h, longitude, inclination, n_sat, env, quota = 500, power = 10, constellation_type="OneWeb"):
        self.ID = ID                    # A unique ID given to every satellite
        self.in_plane = in_plane        # Orbital plane where the satellite is deployed
        self.i_in_plane = i_in_plane    # Index in orbital plane
        self.quota = quota              # Quota of the satellite
        self.h = h                      # Altitude of deployment
        self.power = power              # Transmission power
        self.minElevationAngle = 30     # Value is taken from NGSO constellation design chapter.
        self.constellationType = constellation_type  # Type of constellation, used for beam initialization

        # Spherical Coordinates before inclination (r,theta,phi)
        self.r = Re+self.h 
        self.theta = 2 * math.pi * self.i_in_plane / n_sat
        self.phi = longitude
        
        # Inclination of the orbital plane
        self.inclination = inclination
        
        # Cartesian coordinates  (x,y,z)
        self.x = self.r * (math.sin(self.theta)*math.cos(self.phi) - math.cos(self.theta)*math.sin(self.phi)*math.sin(self.inclination))
        self.y = self.r * (math.sin(self.theta)*math.sin(self.phi) + math.cos(self.theta)*math.cos(self.phi)*math.sin(self.inclination))
        self.z = self.r * math.cos(self.theta)*math.cos(self.inclination)
        
        self.polar_angle = self.theta               # Angle within orbital plane [radians]
        self.latitude = math.asin(self.z/self.r)   # latitude corresponding to the satellite
        # longitude corresponding to satellite
        if self.x > 0:
            self.longitude = math.atan(self.y/self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y/self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y/self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi/2
        elif self.y < 0:
            self.longitude = -math.pi/2
        else:
            self.longitude = 0

        self.waiting_list = {}
        self.applications = []
        self.n_sat = n_sat
        # Intelsat 0_0 payload model: satellite-level configurable bandwidth pool.
        # Value comes from 326 equivalent 36 MHz units.
        self.payload_total_bw_hz = 326.0 * 36e6
        self.payload_min_bw_per_beam_hz = 1e6
        self.payload_alloc_alpha = 1.5

        # downlink params
        f = 20e9  # Carrier frequency GEO to ground (Hz)
        B = 500e6  # Maximum bandwidth
        maxPtx = 10  # Maximum transmission power in W
        Adtx = 0.26  # Transmitter antenna diameter in m
        Adrx = 0.33  # Receiver antenna diameter in m
        pL = 0.3  # Pointing loss in dB
        Nf = 1.5  # Noise figure in dB
        Tn = 50  # Noise temperature in K
        min_rate = 10e3  # Minimum rate in kbps
        self.ngeo2gt = RFlink(f, B, maxPtx, Adtx, Adrx, pL, Nf, Tn, min_rate)
        self.downRate = 0
        self.base_snr_at_center = 0 # To store the base SNR for beams

        # simpy
        self.env = env

        # Satellite beams - assuming OneWeb beams right now 
        self.beams = []
        self.init_beams()
        self._apply_intelsat_payload_allocation()

    def _apply_intelsat_payload_allocation(self):
        """
        Allocate Intelsat satellite-level payload bandwidth across beams.
        Applied only to Intelsat satellite 0_0 (i_in_plane == 0), as requested.
        """
        if self.constellationType != "Intelsat" or self.i_in_plane != 0 or not self.beams:
            return

        # Use inverse-load weighting so less congested beams receive more available BW.
        weights = [max(0.0, 1.0 - beam.load) ** self.payload_alloc_alpha for beam in self.beams]
        weight_sum = float(sum(weights))
        if weight_sum <= 0.0:
            weights = [1.0] * len(self.beams)
            weight_sum = float(len(self.beams))

        n_beams = len(self.beams)
        base_bw = max(0.0, self.payload_min_bw_per_beam_hz)
        remaining_bw = max(0.0, self.payload_total_bw_hz - (base_bw * n_beams))

        for beam, w in zip(self.beams, weights):
            share = (w / weight_sum) if weight_sum > 0.0 else (1.0 / n_beams)
            beam.effective_bw = base_bw + remaining_bw * share
            # Keep reporting field populated; throughput still uses Shannon + max DS cap + demand.
            beam.capacity = beam.max_capacity

    def _set_geo_slot(self, lon_deg):
        """
        Place satellite at GEO equatorial slot longitude (degrees).
        """
        self.theta = math.pi / 2.0  # equatorial
        self.phi = math.radians(lon_deg)
        self.x = self.r * math.cos(self.phi)
        self.y = self.r * math.sin(self.phi)
        self.z = 0.0
        self.polar_angle = self.theta
        self.latitude = 0.0
        self.longitude = self.phi

    def _generate_na_spot_centers(self, count=50):
        """
        Synthetic North America beam center lattice.
        """
        rows = 5
        cols = 10
        # Use a tighter NA window so adjacent spot beams can overlap.
        lon_min, lon_max = -150.0, -70.0
        lat_min, lat_max = 20.0, 60.0
        lons = np.linspace(lon_min, lon_max, cols)
        lats = np.linspace(lat_min, lat_max, rows)
        centers = []
        lon_step = (lon_max - lon_min) / max(cols - 1, 1)
        for r, lat in enumerate(lats):
            offset = 0.5 * lon_step if (r % 2 == 1) else 0.0
            for lon in lons:
                centers.append((float(lat), float(lon + offset)))
        return centers[:count]

    def init_beams(self):
        # OneWeb: coverage area is a square, divided into 16 horizontal strips (vertically stacked)
        total_area_km2 = 1_718_000
        side_km = total_area_km2 ** 0.5  # ≈ 1310 km
        n_beams = 16
        beam_width_km = side_km
        beam_height_km = side_km / n_beams  # ≈ 81.9 km

        deg_per_km = 1 / 111  # Approximate conversion
        # Add tolerance to beam size
        beam_width_deg = (beam_width_km + beam_width_km*0.4) * deg_per_km
        beam_height_deg = (beam_height_km + beam_height_km*0.4) * deg_per_km

        sat_lat = math.degrees(self.latitude)
        sat_lon = math.degrees(self.longitude)

        if self.constellationType =="OneWeb":  

            # The top edge of the coverage square
            top_lat = sat_lat + (side_km / 2) * deg_per_km

            for i in range(n_beams):
                # Each beam's center is halfway between its top and bottom edge
                beam_top = top_lat - i * beam_height_deg
                beam_bottom = beam_top - beam_height_deg
                center_lat = (beam_top + beam_bottom) / 2
                center_lon = sat_lon
                beam_id = f"{self.ID}_beam_{i+1}"  # Number beams 1 to 16
                self.beams.append(Beam(center_lat, center_lon, beam_width_deg, beam_height_deg, id=beam_id, constellation=self.constellationType))
        elif self.constellationType == "Intelsat":
            if self.i_in_plane == 0:
                # First Intelsat GEO satellite: 50 NA-focused synthetic spot beams.
                beam_width_deg = 13.0
                beam_height_deg = 13.0
                for i, (center_lat, center_lon) in enumerate(self._generate_na_spot_centers(count=50), start=1):
                    beam_id = f"{self.ID}_beam_{i}"
                    self.beams.append(
                        Beam(
                            center_lat,
                            center_lon,
                            beam_width_deg,
                            beam_height_deg,
                            id=beam_id,
                            constellation=self.constellationType,
                        )
                    )
            elif self.i_in_plane == 1:
                # Second Intelsat GEO satellite: one wide North America beam.
                # Approximate NA bounding box: lon [-165, -55], lat [15, 70].
                na_center_lat = (15.0 + 70.0) / 2.0
                na_center_lon = (-165.0 + -55.0) / 2.0
                na_width_deg = 110.0
                na_height_deg = 55.0
                wide_beam = Beam(
                    na_center_lat,
                    na_center_lon,
                    na_width_deg,
                    na_height_deg,
                    id=f"{self.ID}_beam_1",
                    constellation=self.constellationType,
                )
                # 0_1 specific payload/RF profile (Ku band, wide beam).
                # Configurable capacity: 28 equivalent 36 MHz units.
                wide_beam.bw = 28.0 * 36e6
                wide_beam.effective_bw = wide_beam.bw
                # Ku band representative frequency.
                wide_beam.frequency = 12.0
                # Given EIRP and assumed wide-beam gain; derive Pt for SNR model.
                wide_beam.eirp_dbw = 41.0
                wide_beam.Gt = 35.0
                wide_beam.Pt = wide_beam.eirp_dbw - wide_beam.Gt + 30.0
                self.beams.append(wide_beam)
            else:
                # Other GEO satellites use a compact generic pattern until explicit footprints are added.
                self.beams.append(Beam(sat_lat, sat_lon, beam_width_deg, beam_height_deg, constellation=self.constellationType))
                n_circle_beams = n_beams - 1
                radius_deg = beam_height_deg * 4
                for i in range(n_circle_beams):
                    angle = 2 * math.pi * i / n_circle_beams
                    dlat = radius_deg * math.cos(angle)
                    dlon = radius_deg * math.sin(angle) / max(math.cos(math.radians(sat_lat)), 1e-6)
                    center_lat = sat_lat + dlat
                    center_lon = sat_lon + dlon
                    self.beams.append(Beam(center_lat, center_lon, beam_width_deg, beam_height_deg, constellation=self.constellationType))
        else:
            # Center beam
            self.beams.append(Beam(sat_lat, sat_lon, beam_width_deg, beam_height_deg, constellation=self.constellationType))

            # Beams on circle
            n_circle_beams = n_beams - 1
            radius_deg = beam_height_deg * 4 # Circle radius in degrees

            for i in range(n_circle_beams):
                angle = 2 * math.pi * i / n_circle_beams
                dlat = radius_deg * math.cos(angle)
                dlon = radius_deg * math.sin(angle) / math.cos(math.radians(sat_lat))  # adjust for latitude distortion
                center_lat = sat_lat + dlat
                center_lon = sat_lon + dlon
                self.beams.append(Beam(center_lat, center_lon, beam_width_deg, beam_height_deg, constellation=self.constellationType))


    def maxSlantRange(self):
        """
        Maximum distance from satellite to edge of coverage area is calculated using the following formula:
        D_max(minElevationAngle, h) = sqrt(Re**2*sin**2(minElevationAngle) + 2*Re*h + h**2) - Re*sin(minElevationAngle)
        This formula is based on the NGSO constellation design chapter page 16.
        """
        eps = math.radians(self.minElevationAngle)

        distance = math.sqrt((Re+self.h)**2-(Re*math.cos(eps))**2) - Re*math.sin(eps)

        return distance

    def __repr__(self):
        return '\nID = {}\n orbital plane= {}, index in plane= {}, h={}\n pos r = {}, pos theta = {},' \
               ' pos phi = {},\n pos x= {}, pos y= {}, pos z= {}\n inclination = {}\n polar angle = {}' \
               '\n latitude = {}\n longitude = {}'.format(
                self.ID,
                self.in_plane,
                self.i_in_plane,
                '%.2f' % self.h,
                '%.2f' % self.r,
                '%.2f' % self.theta,
                '%.2f' % self.phi,
                '%.2f' % self.x,
                '%.2f' % self.y,
                '%.2f' % self.z,
                '%.2f' % math.degrees(self.inclination),
                '%.2f' % math.degrees(self.polar_angle),
                '%.2f' % math.degrees(self.latitude),
                '%.2f' % math.degrees(self.longitude))

    def update_beams(self):
        if self.constellationType == "Intelsat":
            # GEO synthetic spot beams are static in Earth frame.
            return
        # Recalculate beam centers based on current satellite position
        total_area_km2 = 1_718_000
        side_km = total_area_km2 ** 0.5  # ≈ 1310 km
        n_beams = 16
        beam_width_km = side_km
        beam_height_km = side_km / n_beams  # ≈ 81.9 km

        deg_per_km = 1 / 111  # Approximate conversion
        beam_width_deg = beam_width_km * deg_per_km
        beam_height_deg = beam_height_km * deg_per_km


        sat_lat = math.degrees(self.latitude)
        sat_lon = math.degrees(self.longitude)
        if self.constellationType =="OneWeb":  

            # The top edge of the coverage square
            top_lat = sat_lat + (side_km / 2) * deg_per_km

            for i in range(n_beams):
                beam_top = top_lat - i * beam_height_deg
                beam_bottom = beam_top - beam_height_deg
                center_lat = (beam_top + beam_bottom) / 2
                center_lon = sat_lon
                self.beams[i].center_lat = center_lat
                self.beams[i].center_lon = center_lon
                self.beams[i].width_deg = beam_width_deg
                self.beams[i].height_deg = beam_height_deg
                self.beams[i]._calculate_ellipse_parameters() # Recalculate ellipse params
        else:
                # Center beam
            self.beams[0].center_lat = sat_lat
            self.beams[0].center_lon = sat_lon
            self.beams[0]._calculate_ellipse_parameters() # Recalculate ellipse params
            # 15 beams around the center in a circle
            radius_deg = beam_height_deg * 4
            n_beams_side = n_beams - 1
            for i in range(0, n_beams_side):
                angle = 2 * math.pi * i / n_beams_side
                dlat = radius_deg * math.cos(angle)
                dlon = radius_deg * math.sin(angle) / math.cos(math.radians(sat_lat))  # adjust for latitude
                center_lat = sat_lat + dlat
                center_lon = sat_lon + dlon
                self.beams[i].center_lat = center_lat
                self.beams[i].center_lon = center_lon
                self.beams[i]._calculate_ellipse_parameters() # Recalculate ellipse params

    def update_beam_loads(self, time_seconds):
        """
        Update the load for all beams on this satellite.
        
        Args:
            time_seconds: Current simulation time in seconds
        """
        for beam in self.beams:
            beam.update_load(time_seconds)
        self._apply_intelsat_payload_allocation()

    def rotate(self, delta_t, longitude, period):
        """
        Rotates the satellite by re-calculating the sperical coordinates, Cartesian coordinates, and longitude and
        latitude adjusted for the new longitude of the orbit, and fraction the elapsed time makes up of the orbit time
        of the satellite.
        """
        if self.constellationType == "Intelsat":
            # GEO satellites remain fixed in Earth-centered longitude/latitude.
            self.update_beams()
            return

        # Updating spherical coordinates upon rotation (these are phi, theta before inclination)
        self.phi = longitude
        self.theta = self.theta + 2*math.pi*delta_t/period
        self.theta = self.theta % (2*math.pi)
        
        # Calculating x,y,z coordinates with inclination
        self.x = self.r * (math.sin(self.theta)*math.cos(self.phi) - math.cos(self.theta)*math.sin(self.phi)*math.sin(self.inclination))
        self.y = self.r * (math.sin(self.theta)*math.sin(self.phi) + math.cos(self.theta)*math.cos(self.phi)*math.sin(self.inclination))
        self.z = self.r * math.cos(self.theta)*math.cos(self.inclination)
        
        self.polar_angle = self.theta               # Angle within orbital plane [radians]
        self.latitude = math.asin(self.z/self.r)   # latitude corresponding to the satellite
        # longitude corresponding to satellite
        if self.x > 0:
            self.longitude = math.atan(self.y/self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y/self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y/self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi/2
        elif self.y < 0:
            self.longitude = -math.pi/2
        else:
            self.longitude = 0
        self.update_beams()



def load_route_from_csv(filename, skip_rows=0):
    """
    Loads the aircraft route from a CSV file, skipping the first and last N rows.
    Returns a list of dicts with keys: 'lat', 'lon', 'speed_mph'
    Supports both original route format and interpolated route format.
    """
    route = []
    # Try utf-8-sig first, fallback to latin1 if error
    print("Loading flight route from csv")
    try:
        with open(filename, newline='', encoding='utf-8') as csvfile:
            reader = list(csv.DictReader(csvfile))
    except UnicodeDecodeError as e:
        print(f"UTF-8 decoding error: {e}. Trying latin1 encoding.")
        with open(filename, newline='', encoding='latin1') as csvfile:
            reader = list(csv.DictReader(csvfile))
    
    # Ignore first and last skip_rows
    data = reader[skip_rows:-skip_rows] if skip_rows > 0 else reader
    
    # Detect format based on column names
    if not data:
        print("No data found in CSV file")
        return [], 0
        
    first_row = data[0]
    is_interpolated_format = 'seconds' in first_row and 'latitude' in first_row
    is_original_format = 'Latitude' in first_row and 'Time (EDT)' in first_row
    
    print(f"Detected format: {'Interpolated' if is_interpolated_format else 'Original' if is_original_format else 'Unknown'}")
    
    for row in data:
        try:
            if is_interpolated_format:
                # Handle interpolated format (lowercase column names, different datetime format)
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                speed = float(row['speed_mph'])
                altitude = float(row['altitude_ft'])
                # Handle interpolated datetime format: "1900-01-01 10:48:57"
                time = pd.to_datetime(row['datetime'], format="%Y-%m-%d %H:%M:%S")
                route.append({'lat': lat, 'lon': lon, 'speed_mph': speed, 'alt': altitude, 'time': time})
                
            elif is_original_format:
                # Handle original format (uppercase column names, different datetime format)
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
                speed = float(row['mph'])
                # Clean altitude field (remove commas and quotes)
                altitude_str = row['feet'].replace(',', '').replace('"', '')
                altitude = float(altitude_str)
                # Handle original datetime format: "Tue 10:48:57 AM"
                time = pd.to_datetime(row['Time (EDT)'], format="%a %I:%M:%S %p")
                route.append({'lat': lat, 'lon': lon, 'speed_mph': speed, 'alt': altitude, 'time': time})
                
            else:
                print(f"Unknown CSV format. Available columns: {list(row.keys())}")
                break
                
        except Exception as e:
            print(f"Skipping malformed row due to error: {e}")
            print(f"Row data: {row}")
            continue  # skip malformed rows
    
    if len(route) > 0:
        route_duration = (route[-1]['time'] - route[0]['time']).total_seconds()
        print(f"Loaded {len(route)} route points. Duration: {route_duration:.1f} seconds")
    else:
        route_duration = 0
        print("No valid route points loaded")
        
    return route, route_duration


class Aircraft: 
    scenario = None
    def __init__(self, env, aircraft_id, route=None, height=10000, update_interval=1, passengers=None, handover_rng=None, supported_constellations=None):
        self.passengers = passengers 
        self.time = 0 # in seconds 
        self.deltaT = 1 
        self.env = env
        self.id = aircraft_id
        self.update_interval = update_interval
        # Aircraft receive antennas: separate terminals for LEO and GEO links.
        self.Gr_leo = 40 # dBi (kept as-is for LEO)
        self.Gr_geo = 40 # dBi (GEO terminal; tune as needed)
        # If None, the aircraft can scan all constellations present in the Earth model.
        self.supported_constellations = set(supported_constellations) if supported_constellations else None

        self.route = route or []
        self.route_index = 0

        # Initialize position from route if available
        if self.route:
            self.latitude = self.route[0]['lat']
            self.longitude = self.route[0]['lon']
            self.height = self.route[0]['alt'] * 0.3048  # Convert feet to meters
            self.speed_kmph = self.route[0]['speed_mph'] * 1.60934
        else:
            self.latitude = 0
            self.longitude = 0
            self.speed_kmph = 0
            self.height = height  # in meters

        self.connected_satellite = None
        self.connected_beam = None
        self.current_snr = 0
        self.current_latency = 0
        self.handover_count = 0
        self.min_dwell_s = 5.0
        self.time_to_trigger_s = 5.0
        self.handover_rng = handover_rng or random.Random()
        self.last_handover_time = -float("inf")
        self.ttt_candidate = None
        self.ttt_candidate_sat = None
        self.ttt_start_time = None
        self.service_drop_until = 0.0
        # Latency model terms (RTT-level additions): fixed processing + stochastic jitter
        self.fixed_processing_latency_s = 0.030
        self.jitter_std_s = 0.005
        self.snr_congested_top_k = 2
        self.snr_congested_bias = 0.35
        self._snr_congestion_last_time = None
        self._snr_congestion_applied_ids = set()

        self.demand = 0 
        self.total_allocated_bandwidth = 0.0  # Total allocated bandwidth (MB)
        self.allocation_ratios = []
        self.total_demand = 0 
        self.latest_candidates = []
        self.passenger_step_cache = []
        self.traffic_steering_enabled = True
        self.last_service_drop_s = 0.0
        self.last_latency_violation_rate = 0.0
        self.last_avg_latency_s = 0.0
        self.last_transmission_rate_mbps = 0.0
        self.last_throughput_req_mbps = 0.0
        self.last_steering_assignments = []
        self.last_constellation_metrics = {}
        self.last_satellite_metrics = {}
        self.last_beam_metrics = {}
        # Traffic-steering RL interface (fixed-size masked action).
        self.max_steer_apps = 25
        self.max_steer_candidates = 12
        self.steering_action = None  # Backward-compatible dict: {passenger_id: "sat_id::beam_id"}
        self.steering_action_indices = None  # Preferred interface: length-M list with candidate-slot indices
        self.current_steering_candidates = []
        self.current_passenger_order = []

        print(f"Aircraft {self.id} initialized at ({self.latitude:.2f}, {self.longitude:.2f})")

    def __repr__(self):
        return 'Aircraft ID: {}, Latitude: {}, Longitude: {}, Altitude: {}'.format(
            self.id, self.latitude, self.longitude, self.height)

    def _get_rx_gain_dbi(self, sat=None):
        if sat is not None and getattr(sat, "constellationType", "") == "Intelsat":
            return self.Gr_geo
        return self.Gr_leo

    def calculate_snr(self, beam, distance_km, sat=None):
        # Supervisor's SNR Calculation Method
        # Constants
        T = 290  # Noise temperature (K)
        
        # Beam parameters (assumed to be in dBm, dBi, Hz, etc.)
        Pt_dBm = beam.Pt  # Transmit power in dBm
        Gt_dBi = beam.Gt  # Transmit antenna gain in dBi
        Gr_dBi = self._get_rx_gain_dbi(sat)  # Aircraft receive antenna gain in dBi
        f_Hz = beam.frequency * 1e9  # Frequency in Hz (beam.frequency in GHz)
        B_Hz = max(beam.effective_bw, 1.0)  # Bandwidth in Hz

        # Convert dBm to dBW
        Pt_dBW = Pt_dBm - 30

        # Free-space path loss (FSPL) in dB
        d_m = distance_km * 1000
        if d_m == 0: return float('inf') # Avoid log(0)
        FSPL_dB = 20 * math.log10(d_m) + 20 * math.log10(f_Hz) - 147.55

        # Received power in dBW
        off_axis = beam.off_axis_gain_db(self.latitude, self.longitude)
        Gt_dBi_effective = Gt_dBi + off_axis
        Pr_dBW = Pt_dBW + Gt_dBi_effective + Gr_dBi - FSPL_dB

        # Noise power in dBW
        N_dBW = 10 * math.log10(k * T * B_Hz)

        # SNR in dB
        SNR_dB = Pr_dBW - N_dBW
        return SNR_dB

    def scan_nearby_fast(self, constellation, threshold_km=1000):
        """
        Returns a list of all beams where the aircraft is inside the footprint.
        Each entry is a dict with beam, satellite, SNR, load, capacity, etc.
        """
        leo_beam_coords = []
        leo_beam_refs = []
        geo_beam_refs = []
        for plane in constellation:
            for sat in plane.sats:
                for beam in sat.beams:
                    if getattr(sat, "constellationType", "") == "Intelsat":
                        geo_beam_refs.append((sat, beam))
                    else:
                        leo_beam_coords.append([beam.center_lat, beam.center_lon])
                        leo_beam_refs.append((sat, beam))

        if not leo_beam_refs and not geo_beam_refs:
            return []

        aircraft_point = Point(self.longitude, self.latitude)
        candidates = []

        # LEO path: unchanged fast KDTree neighborhood scan.
        if leo_beam_refs:
            beam_coords = np.array(leo_beam_coords)
            tree = KDTree(beam_coords)
            threshold_deg = threshold_km / 111.0
            idxs = tree.query_ball_point([self.latitude, self.longitude], r=threshold_deg)
            for idx in idxs:
                sat, beam = leo_beam_refs[idx]
                dist_3d = self._calculate_3d_distance(sat)
                snr = self.calculate_snr(beam, dist_3d / 1000, sat=sat) # distance in km
                if beam.get_footprint().intersects(aircraft_point) or beam.get_footprint().contains(aircraft_point):
                    candidates.append({
                        'sat': sat,
                        'beam': beam,
                        'snr': snr,
                        'distance_km': dist_3d / 1000,
                        'load': beam.load,
                        'capacity': beam.capacity,
                        'beam_id': beam.id,
                        'sat_id': sat.ID
                    })

        # GEO path: evaluate GEO beams directly by footprint (no LEO thresholding).
        for sat, beam in geo_beam_refs:
            if not (beam.get_footprint().intersects(aircraft_point) or beam.get_footprint().contains(aircraft_point)):
                continue
            dist_3d = self._calculate_3d_distance(sat)
            snr = self.calculate_snr(beam, dist_3d / 1000, sat=sat) # distance in km
            candidates.append({
                'sat': sat,
                'beam': beam,
                'snr': snr,
                'distance_km': dist_3d / 1000,
                'load': beam.load,
                'capacity': beam.capacity,
                'beam_id': beam.id,
                'sat_id': sat.ID
            })
        if Aircraft.scenario == "snr_congested" and candidates:
            self._apply_snr_congestion(candidates)
        print(f"candidates found: {len(candidates)}")
        return candidates

    def _apply_snr_congestion(self, candidates):
        sim_time = self.env.now if hasattr(self.env, "now") else self.time
        if self._snr_congestion_last_time != sim_time:
            self._snr_congestion_last_time = sim_time
            self._snr_congestion_applied_ids.clear()
        ranked = sorted(candidates, key=lambda c: c["snr"], reverse=True)
        top_k = min(self.snr_congested_top_k, len(ranked))
        for cand in ranked[:top_k]:
            beam = cand["beam"]
            if beam.id in self._snr_congestion_applied_ids:
                cand["load"] = beam.load
                cand["capacity"] = beam.capacity
                continue
            beam.load = min(1.0, beam.load + self.snr_congested_bias)
            self._snr_congestion_applied_ids.add(beam.id)
            sat = cand.get("sat")
            if sat is not None and hasattr(sat, "_apply_intelsat_payload_allocation"):
                sat._apply_intelsat_payload_allocation()
            else:
                beam.effective_bw = beam.bw * (1 - beam.load)
                beam.capacity = beam.max_capacity
            cand["load"] = beam.load
            cand["capacity"] = beam.capacity

    def scan_nearby(self, constellation, threshold_km=500):
        results = []
        for plane in constellation:
            for sat in plane.sats:
                sat_dist = geopy.distance.distance(
                    (self.latitude, self.longitude),
                    (math.degrees(sat.latitude), math.degrees(sat.longitude))
                ).km
                if sat_dist < threshold_km:
                    results.append({'type': 'satellite', 'id': sat.ID, 'distance_km': sat_dist})
                for beam in sat.beams:
                    beam_dist = geopy.distance.distance(
                        (self.latitude, self.longitude),
                        (beam.center_lat, beam.center_lon)
                    ).km
                    if beam_dist < threshold_km:
                        snr = self.calculate_snr(beam, beam_dist, sat=sat)
                        results.append({
                            'type': 'beam',
                            'sat_id': sat.ID,
                            'beam_id': beam.id,
                            'distance_km': beam_dist,
                            'snr_db': snr, 
                            'load': sat.load, 
                            'capacity': sat.capacity
                        })
        return results

    def get_demand_at_time(self, deltaT):
        """
        Calculate the aircraft's demand at a given time using a random function.
        Returns:
            Demand in Mbps
        """
        num_users = random.randint(1, 10)  # 1 to 10 users
        demand_per_user = [random.uniform(0, 25) for _ in range(num_users)]  # 2–25 Mbps per user
        total_demand = sum(demand_per_user)
        self.total_demand += (total_demand*deltaT)/8
        return (total_demand*deltaT)/8  

    def update_demand(self, deltaT):
        """
        Update the aircraft's current demand based on the sinusoidal function.
        Args:
            time_seconds: Current simulation time in seconds
        """
        self.demand = self.get_demand_at_time(deltaT)   

    def step_passengers(self, deltaT):
        total_throughput = 0.0
        min_latency = float('inf')
        self.demand = 0.0
        self.passenger_step_cache = []
        if self.passengers:
            for passenger in self.passengers:
                demand, throughput, latency = passenger.step_application(deltaT) # Convert Mbps to MB for deltaT
                self.demand += demand*deltaT/8 
                total_throughput += throughput
                min_latency = min(min_latency, latency)
                self.passenger_step_cache.append(
                    {
                        "passenger_id": passenger.ID,
                        "demand_mbps": float(max(demand, 0.0)),
                        "latency_req_s": float(max(latency, 0.0)) / 1000.0,
                    }
                )

        return self.demand, total_throughput, min_latency

    def _candidate_available_rate_mbps(self, candidate):
        beam = candidate["beam"]
        snr_db = candidate["snr"]
        if beam.load >= 1.0:
            return 0.0
        shannon_capacity_mbps = beam.effective_bw * math.log2(1 + 10 ** (snr_db / 10)) / 1e6
        return max(0.0, min(shannon_capacity_mbps, beam.max_ds_speed))

    def _candidate_latency_s(self, sat):
        aircraft_to_sat_m = self._calculate_3d_distance(sat)
        sat_to_gateway_m = self._calculate_3d_distance_to_gateway(sat)
        propagation_rtt_s = 2.0 * ((aircraft_to_sat_m + sat_to_gateway_m) / Vc)
        return propagation_rtt_s + self.fixed_processing_latency_s

    def _steer_passenger_traffic(self, deltaT):
        """
        Greedy per-step traffic steering:
        map each passenger/application demand to the best candidate beam (LEO or GEO)
        while respecting per-beam instantaneous capacity.
        """
        demands = [d for d in self.passenger_step_cache if d["demand_mbps"] > 1e-9]
        if not demands:
            self.last_steering_assignments = []
            self.last_service_drop_s = 0.0
            self.last_latency_violation_rate = 0.0
            self.last_avg_latency_s = 0.0
            self.last_transmission_rate_mbps = 0.0
            self.last_throughput_req_mbps = 0.0
            self.last_constellation_metrics = {}
            self.last_satellite_metrics = {}
            self.last_beam_metrics = {}
            return {
                "allocated_mb": 0.0,
                "demand_mb": 0.0,
                "beam_capacity_mb": 0.0,
                "served_mbps_total": 0.0,
                "service_drop_s": 0.0,
                "violation_rate": 0.0,
                "avg_latency_s": 0.0,
            }

        if not self.latest_candidates:
            total_demand_mbps = sum(d["demand_mbps"] for d in demands)
            service_drop = deltaT
            self.last_steering_assignments = []
            self.last_service_drop_s = service_drop
            self.last_latency_violation_rate = 1.0
            self.last_avg_latency_s = 0.0
            self.last_transmission_rate_mbps = 0.0
            self.last_throughput_req_mbps = total_demand_mbps
            self.last_constellation_metrics = {}
            self.last_satellite_metrics = {}
            self.last_beam_metrics = {}
            return {
                "allocated_mb": 0.0,
                "demand_mb": (total_demand_mbps * deltaT) / 8.0,
                "beam_capacity_mb": 0.0,
                "served_mbps_total": 0.0,
                "service_drop_s": service_drop,
                "violation_rate": 1.0,
                "avg_latency_s": 0.0,
            }

        # Build a fixed-size candidate set across both networks (LEO + GEO).
        ranked_candidates = sorted(
            self.latest_candidates,
            key=lambda c: c.get("snr", -1e9),
            reverse=True,
        )[: self.max_steer_candidates]
        self.current_steering_candidates = ranked_candidates
        ordered_demands = sorted(demands, key=lambda x: (x["latency_req_s"], -x["demand_mbps"]))[: self.max_steer_apps]
        self.current_passenger_order = [d["passenger_id"] for d in ordered_demands]

        remaining = {}
        latencies = {}
        beam_caps_mbps = {}
        for idx, cand in enumerate(ranked_candidates):
            key = f"{cand['sat_id']}::{cand['beam_id']}::{idx}"
            cap = self._candidate_available_rate_mbps(cand)
            remaining[key] = cap
            beam_caps_mbps[key] = cap
            latencies[key] = self._candidate_latency_s(cand["sat"])
        candidate_meta = {
            f"{cand['sat_id']}::{cand['beam_id']}::{idx}": cand
            for idx, cand in enumerate(ranked_candidates)
        }
        user_to_key = {}
        for k in remaining.keys():
            sat_id, beam_id, _ = k.split("::")
            user_to_key[f"{sat_id}::{beam_id}"] = k

        total_demand_mbps = sum(d["demand_mbps"] for d in ordered_demands)
        served_mbps_total = 0.0
        latency_samples = []
        violations = 0
        unmet_count = 0
        assignments = []

        for app_slot, d in enumerate(ordered_demands):
            pid = d["passenger_id"]
            req = d["demand_mbps"]
            req_lat_s = d["latency_req_s"]
            best_key = None
            best_score = -float("inf")
            best_rate = 0.0
            best_lat = 0.0

            # Optional external action: passenger_id -> "sat_id::beam_id"
            forced_choice = None
            if isinstance(self.steering_action_indices, (list, tuple, np.ndarray)) and app_slot < len(self.steering_action_indices):
                chosen_idx = self.steering_action_indices[app_slot]
                if isinstance(chosen_idx, (int, np.integer)) and 0 <= int(chosen_idx) < len(ranked_candidates):
                    cand = ranked_candidates[int(chosen_idx)]
                    forced_choice = f"{cand['sat_id']}::{cand['beam_id']}"
            if forced_choice is None and isinstance(self.steering_action, dict):
                forced_choice = self.steering_action.get(pid)
            if isinstance(forced_choice, str):
                forced_key = user_to_key.get(forced_choice)
                if forced_key and remaining.get(forced_key, 0.0) > 1e-9:
                    forced_rate = min(req, remaining[forced_key])
                    forced_lat = latencies[forced_key]
                    if forced_rate > 1e-9:
                        best_key = forced_key
                        best_rate = forced_rate
                        best_lat = forced_lat

            if best_key is None:
                for key, rem in remaining.items():
                    if rem <= 1e-9:
                        continue
                    cand_rate = min(req, rem)
                    cand_lat = latencies[key]
                    latency_penalty = max(0.0, cand_lat - req_lat_s)
                    throughput_term = cand_rate / max(req, 1e-9)
                    score = throughput_term - 2.0 * latency_penalty
                    if score > best_score:
                        best_score = score
                        best_key = key
                        best_rate = cand_rate
                        best_lat = cand_lat

            if best_key is None or best_rate <= 1e-9:
                unmet_count += 1
                assignments.append(
                    {
                        "passenger_id": pid,
                        "beam_key": None,
                        "served_mbps": 0.0,
                        "demand_mbps": req,
                        "latency_s": None,
                        "latency_req_s": req_lat_s,
                    }
                )
                continue

            remaining[best_key] = max(0.0, remaining[best_key] - best_rate)
            served_mbps_total += best_rate
            latency_samples.append(best_lat)
            if best_lat > req_lat_s:
                violations += 1
            if best_rate + 1e-9 < req:
                unmet_count += 1
            assignments.append(
                {
                    "passenger_id": pid,
                    "beam_key": best_key,
                    "served_mbps": best_rate,
                    "demand_mbps": req,
                    "latency_s": best_lat,
                    "latency_req_s": req_lat_s,
                }
            )

        allocated_mb = (served_mbps_total * deltaT) / 8.0
        demand_mb = (total_demand_mbps * deltaT) / 8.0
        beam_capacity_mb = (sum(beam_caps_mbps.values()) * deltaT) / 8.0
        violation_rate = violations / len(demands) if demands else 0.0
        avg_latency_s = sum(latency_samples) / len(latency_samples) if latency_samples else 0.0
        service_drop_s = deltaT if unmet_count > 0 else 0.0

        self.last_steering_assignments = assignments
        self.last_service_drop_s = service_drop_s
        self.last_latency_violation_rate = violation_rate
        self.last_avg_latency_s = avg_latency_s
        self.last_transmission_rate_mbps = served_mbps_total
        self.last_throughput_req_mbps = total_demand_mbps
        self._update_hierarchical_qoe_metrics(assignments, candidate_meta)

        return {
            "allocated_mb": allocated_mb,
            "demand_mb": demand_mb,
            "beam_capacity_mb": beam_capacity_mb,
            "served_mbps_total": served_mbps_total,
            "service_drop_s": service_drop_s,
            "violation_rate": violation_rate,
            "avg_latency_s": avg_latency_s,
        }

    @staticmethod
    def _empty_group_metrics():
        return {
            "demand_mbps": 0.0,
            "served_mbps": 0.0,
            "unmet_mbps": 0.0,
            "allocation_ratio": 1.0,
            "avg_latency_s": 0.0,
            "latency_violation_rate": 0.0,
            "passenger_count": 0,
            "served_passenger_count": 0,
            "dropped_passenger_count": 0,
            "_latency_sum_s": 0.0,
            "_latency_violations": 0,
        }

    def _finalize_group_metrics(self, m):
        if m["demand_mbps"] > 1e-9:
            m["allocation_ratio"] = m["served_mbps"] / m["demand_mbps"]
        else:
            m["allocation_ratio"] = 1.0
        if m["served_passenger_count"] > 0:
            m["avg_latency_s"] = m["_latency_sum_s"] / m["served_passenger_count"]
        else:
            m["avg_latency_s"] = 0.0
        if m["passenger_count"] > 0:
            m["latency_violation_rate"] = m["_latency_violations"] / m["passenger_count"]
        else:
            m["latency_violation_rate"] = 0.0
        m.pop("_latency_sum_s", None)
        m.pop("_latency_violations", None)
        return m

    def _update_hierarchical_qoe_metrics(self, assignments, candidate_meta):
        constellation_metrics = {}
        satellite_metrics = {}
        beam_metrics = {}

        for a in assignments:
            demand = float(a.get("demand_mbps", 0.0))
            served = float(a.get("served_mbps", 0.0))
            unmet = max(0.0, demand - served)
            lat = a.get("latency_s", None)
            lat_req = float(a.get("latency_req_s", 0.0))
            beam_key = a.get("beam_key", None)
            if beam_key is None or beam_key not in candidate_meta:
                continue

            cand = candidate_meta[beam_key]
            sat = cand["sat"]
            constellation = getattr(sat, "constellationType", "Unknown")
            sat_id = cand["sat_id"]
            beam_id = cand["beam_id"]
            beam_compact_key = f"{sat_id}::{beam_id}"

            if constellation not in constellation_metrics:
                constellation_metrics[constellation] = self._empty_group_metrics()
            if sat_id not in satellite_metrics:
                satellite_metrics[sat_id] = self._empty_group_metrics()
            if beam_compact_key not in beam_metrics:
                beam_metrics[beam_compact_key] = self._empty_group_metrics()

            for grp in (
                constellation_metrics[constellation],
                satellite_metrics[sat_id],
                beam_metrics[beam_compact_key],
            ):
                grp["passenger_count"] += 1
                grp["demand_mbps"] += demand
                grp["served_mbps"] += served
                grp["unmet_mbps"] += unmet
                if served > 1e-9:
                    grp["served_passenger_count"] += 1
                    if lat is not None:
                        grp["_latency_sum_s"] += float(lat)
                        if float(lat) > lat_req:
                            grp["_latency_violations"] += 1
                else:
                    grp["dropped_passenger_count"] += 1

        self.last_constellation_metrics = {
            k: self._finalize_group_metrics(v)
            for k, v in constellation_metrics.items()
        }
        self.last_satellite_metrics = {
            k: self._finalize_group_metrics(v)
            for k, v in satellite_metrics.items()
        }
        self.last_beam_metrics = {
            k: self._finalize_group_metrics(v)
            for k, v in beam_metrics.items()
        }

    def _get_action_mask_from_ordered(self, ordered_demands, ranked_candidates):
        m = self.max_steer_apps
        k = self.max_steer_candidates
        mask = np.zeros((m, k), dtype=np.float32)
        for i, _ in enumerate(ordered_demands[:m]):
            for j, cand in enumerate(ranked_candidates[:k]):
                if self._candidate_available_rate_mbps(cand) > 1e-9:
                    mask[i, j] = 1.0
        return mask

    def get_steering_observation(self):
        """
        Returns fixed-size steering tensors and a mask for RL:
        - passenger_features: [M, 3] = [demand_mbps, latency_req_s, active_flag]
        - candidate_features: [K, 6] = [snr_db, load, eff_bw_hz_norm, max_ds_norm, est_latency_s, is_geo]
        - action_mask: [M, K] valid assignment mask
        """
        ordered_demands = sorted(
            [d for d in self.passenger_step_cache if d["demand_mbps"] > 1e-9],
            key=lambda x: (x["latency_req_s"], -x["demand_mbps"]),
        )[: self.max_steer_apps]
        ranked_candidates = sorted(
            self.latest_candidates,
            key=lambda c: c.get("snr", -1e9),
            reverse=True,
        )[: self.max_steer_candidates]
        self.current_passenger_order = [d["passenger_id"] for d in ordered_demands]
        self.current_steering_candidates = ranked_candidates

        M = self.max_steer_apps
        K = self.max_steer_candidates
        pfeat = np.zeros((M, 3), dtype=np.float32)
        cfeat = np.zeros((K, 6), dtype=np.float32)

        for i, d in enumerate(ordered_demands):
            pfeat[i, 0] = float(d["demand_mbps"])
            pfeat[i, 1] = float(d["latency_req_s"])
            pfeat[i, 2] = 1.0

        for j, c in enumerate(ranked_candidates):
            beam = c["beam"]
            sat = c["sat"]
            cfeat[j, 0] = float(c.get("snr", -100.0))
            cfeat[j, 1] = float(c.get("load", 1.0))
            cfeat[j, 2] = float(beam.effective_bw / 1e9)
            cfeat[j, 3] = float(beam.max_ds_speed / 1000.0)
            cfeat[j, 4] = float(self._candidate_latency_s(sat))
            cfeat[j, 5] = 1.0 if getattr(sat, "constellationType", "") == "Intelsat" else 0.0

        mask = self._get_action_mask_from_ordered(ordered_demands, ranked_candidates)
        candidate_ids = [f"{c['sat_id']}::{c['beam_id']}" for c in ranked_candidates]
        passenger_ids = [d["passenger_id"] for d in ordered_demands]
        return {
            "passenger_features": pfeat,
            "candidate_features": cfeat,
            "action_mask": mask,
            "passenger_ids": passenger_ids,
            "candidate_ids": candidate_ids,
        }

    def set_steering_action(self, action_map=None, action_indices=None):
        """
        Optional external action interfaces:
        - action_map format: {passenger_id: "sat_id::beam_id"}
        - action_indices format: length-M list of candidate slot indices (preferred RL path)
        If not set, steering uses greedy fallback each step.
        """
        self.steering_action = action_map
        self.steering_action_indices = action_indices

    def queing_delay(self, requested, deltaT):
        """
        Calculate queuing delay based on current load and data size.
        
        Args:
            data_size_mb: Size of the data to be transmitted in megabytes  """ 
        if self.connected_beam is None: 
            return 1000, 0   # No connection means infinite delay
        
        if self.connected_beam.load >= 1.0:
            return 1000, 0   # Infinite delay if load is at or above capacity
        
        shannon_capacity = self.connected_beam.effective_bw * math.log2(1 + 10**(self.current_snr/10)) / 1e6  # in Mbps
        demand_mbps = (self.demand * 8) / max(deltaT, 1e-9)
        transmission_rate_mbps = min(shannon_capacity, self.connected_beam.max_ds_speed, demand_mbps)
        
        if transmission_rate_mbps <= 0.0:
            return 1000, 0.0
    
        service_Mb = transmission_rate_mbps * deltaT  # Mb that can be served this step
        demand_Mb = self.demand * 8  # Convert MB to Mb for calculation
        
        if demand_Mb <= service_Mb:
            return 0, transmission_rate_mbps  # No delay if demand can be served immediately
        else: 
            backlog_Mb    = demand_Mb - service_Mb
            delay_seconds = backlog_Mb / (transmission_rate_mbps)   # Convert MB to Mb for calculation

        return delay_seconds, transmission_rate_mbps
    
    def allocation_ratio(self, deltaT):
        """
        Returns the ratio of allocated throughput to total demand for the current time step.
        Throughput is limited by the available beam capacity.
        """
        # Update demand for this timestep
        #self.update_demand(deltaT)
        #demand = self.demand  # in MB (as per your get_demand_at_time)

        if self.traffic_steering_enabled:
            steered = self._steer_passenger_traffic(deltaT)
            allocated = steered["allocated_mb"]
            demand = steered["demand_mb"]
            beam_capacity_MB = steered["beam_capacity_mb"]
            ratio = (allocated / demand) if demand > 1e-9 else 1.0
            self.total_allocated_bandwidth += allocated
            self.allocation_ratios.append(ratio)
            return ratio, allocated, demand, beam_capacity_MB

        # Get available capacity from the current beam (in Gbps, convert to MB for deltaT)
        if self.connected_beam:
            if self.connected_beam.load >= 1.0:
                allocated = 0.0
                beam_capacity_MB = self.connected_beam.max_capacity * 125 * deltaT  # MB for reporting
                shannon_capacity = 0.0
            else:
            # Convert beam capacity from Gbps to MB for this timestep
            # 1 Gbps = 125 MB/s
                beam_capacity_MB = self.connected_beam.max_capacity * 125 * deltaT  # MB for this timestep (for reporting)
                shannon_capacity = self.connected_beam.effective_bw * math.log2(1 + 10**(self.current_snr/10)) / 1e6  # in Mbps
        else:
            beam_capacity_MB = 0
            shannon_capacity = 0 

        # Allocated bandwidth is bounded by shannon capacity, demand, and max DS speed.
        if self.connected_beam and self.connected_beam.load < 1.0: 
            demand_mbps = (self.demand * 8) / max(deltaT, 1e-9)
            served_mbps = min(shannon_capacity, self.connected_beam.max_ds_speed, demand_mbps)
            allocated = (served_mbps * deltaT) / 8
        else: 
            allocated = 0 

        # No unmet demand when demand is zero, so treat ratio as fully satisfied.
        if self.demand > 0:
            ratio = allocated / self.demand
        else:
            ratio = 1.0

        # Track stats for the episode
        self.total_allocated_bandwidth += allocated
        self.allocation_ratios.append(ratio)            

        return ratio, allocated, self.demand, beam_capacity_MB   

    def get_qoe_metrics(self, deltaT):
        """
        Returns qoe metrics based on passenger usage
        """
        demand, total_throughput_required, max_latency_allowed = self.step_passengers(deltaT)
        ratio, allocated, demand, beam_capacity_MB = self.allocation_ratio(deltaT) 
        if self.traffic_steering_enabled:
            queing_delay = 0.0 if allocated > 0.0 else (1000.0 if demand > 0.0 else 0.0)
            transmission_rate_mbps = self.last_transmission_rate_mbps
            total_throughput_required = self.last_throughput_req_mbps
        else:
            queing_delay, transmission_rate_mbps = self.queing_delay(total_throughput_required, deltaT)
        # RTT latency = propagation RTT + fixed processing + stochastic jitter
        jitter_s = abs(self.handover_rng.gauss(0.0, self.jitter_std_s))
        propagation_latency = self._calculate_latency() * 2 + self.fixed_processing_latency_s + jitter_s
        sim_time = self.env.now if hasattr(self.env, "now") else self.time
        if self.traffic_steering_enabled:
            service_drop_s = self.last_service_drop_s
        else:
            service_drop_s = deltaT if demand > 0 and allocated <= 0 else 0.0
        dwell_remaining_s = max(0.0, self.min_dwell_s - (sim_time - self.last_handover_time))
        if self.ttt_start_time is None:
            ttt_remaining_s = 0.0
        else:
            ttt_remaining_s = max(0.0, self.time_to_trigger_s - (sim_time - self.ttt_start_time))
        return {
            'allocation_ratio': ratio,
            'allocated_bandwidth_MB': allocated,
            'demand_MB': demand,
            'beam_capacity_MB': beam_capacity_MB,
            'queuing_delay_s': queing_delay,
            'transmission_rate_mbps': transmission_rate_mbps,
            'propagation_latency_s': propagation_latency, 
            'throughput_req_mbps': total_throughput_required, 
            'latency_req_s': max_latency_allowed/1000, # convert ms to s 
            'SNR_dB': self.current_snr,
            'service_drop_s': service_drop_s,
            'latency_violation_rate': self.last_latency_violation_rate if self.traffic_steering_enabled else 0.0,
            'avg_assigned_latency_s': self.last_avg_latency_s if self.traffic_steering_enabled else propagation_latency,
            'constellation_qoe': self.last_constellation_metrics if self.traffic_steering_enabled else {},
            'satellite_qoe': self.last_satellite_metrics if self.traffic_steering_enabled else {},
            'beam_qoe': self.last_beam_metrics if self.traffic_steering_enabled else {},
            'dwell_remaining_s': dwell_remaining_s,
            'ttt_remaining_s': ttt_remaining_s
        }

    def time_between_points(self, lat1, lon1, lat2, lon2, speed_kmph):
        # Calculate distance in kilometers
        coords_1 = (lat1, lon1)
        coords_2 = (lat2, lon2)
        distance_km = geopy.distance.geodesic(coords_1, coords_2).km
        # Calculate time in hours
        if speed_kmph == 0:
            return float('inf')  # Avoid division by zero
        time_hours = distance_km / speed_kmph
        return time_hours
     
    def _update_position(self):
        """
        Updates aircraft's latitude, longitude, and speed from the route.
        If using a real route, just step to the next point.
        """
        if self.route and self.route_index < len(self.route):
            point = self.route[self.route_index]
            prevpoint = self.route[self.route_index - 1] if self.route_index > 0 else point
            self.latitude = point['lat']
            self.longitude = point['lon']
            self.height = point['alt'] * 0.3048  # feet to meters
            self.speed_kmph = point['speed_mph'] * 1.60934  # mph to km/h
            self.route_index += 1  # Move to next point for next update
            print(f"route index {self.route_index} of {len(self.route)}")
            deltaT = self.time_between_points(prevpoint['lat'], prevpoint['lon'], point['lat'], point['lon'], self.speed_kmph)
            deltaT_seconds = deltaT * 3600 if deltaT and deltaT > 0 else 1  # At least 1 second
            self.time += deltaT_seconds
            print(f"new posotion {self.longitude} {self.latitude} speed {self.speed_kmph} deltaT {deltaT_seconds}")
            return deltaT_seconds
        # If not using a route, you could implement the old logic here (with direction), but skip for real route.

    def move_and_connect_aircraft(self, constellation):

        # 1. Move the aircraft
        deltaT = self._update_position()
        self.deltaT = deltaT 
        print(f"self.deltaT: {self.deltaT}")
        sim_time = self.env.now if hasattr(self.env, "now") else self.time

        # 2. Find the best beam using the efficient KDTree scan
        candidates = self.scan_nearby_fast(constellation)  
        self.latest_candidates = candidates
        if not candidates:
            if self.connected_beam:
                print(f"Time {sim_time:.2f}: Aircraft {self.id} LOST connection from {self.connected_beam.id}")
            self.connected_beam = None
            self.connected_satellite = None
            self.current_snr = -100
            self.current_latency = 0
            self.ttt_candidate = None
            self.ttt_candidate_sat = None
            self.ttt_start_time = None
            self.latest_candidates = []
            return deltaT

        best = max(candidates, key=lambda c: c["snr"])
        best_candidate_sat = best["sat"]
        best_candidate_beam = best["beam"]
        best_snr = best["snr"]

        if self.connected_beam is None:
            print(f"Time {sim_time:.2f}: Aircraft {self.id} CONNECTED to {best_candidate_beam.id} (Initial connection)")
            self.connected_beam = best_candidate_beam
            self.connected_satellite = best_candidate_sat
            self.current_snr = best_snr
            self.current_latency = self._calculate_latency()
            self.last_handover_time = sim_time
            return deltaT

        if best_candidate_beam == self.connected_beam:
            self.current_snr = best_snr
            self.current_latency = self._calculate_latency()
            self.ttt_candidate = None
            self.ttt_candidate_sat = None
            self.ttt_start_time = None
            return deltaT

        current_snr_now = self.calculate_snr(
            self.connected_beam,
            self._calculate_3d_distance(self.connected_satellite) / 1000,
            sat=self.connected_satellite,
        )
        self.current_snr = current_snr_now
        self.current_latency = self._calculate_latency()

        if sim_time - self.last_handover_time < self.min_dwell_s:
            return deltaT

        if self.ttt_candidate != best_candidate_beam:
            self.ttt_candidate = best_candidate_beam
            self.ttt_candidate_sat = best_candidate_sat
            self.ttt_start_time = sim_time
            return deltaT

        if sim_time - self.ttt_start_time < self.time_to_trigger_s:
            return deltaT

        self.handover_count += 1
        print(f"Time {sim_time:.2f}: Aircraft {self.id} HANDOVER from {self.connected_beam.id} to {best_candidate_beam.id}. Handovers: {self.handover_count}")
        sat_changed = self.connected_satellite is not None and best_candidate_sat != self.connected_satellite
        self.connected_beam = best_candidate_beam
        self.connected_satellite = best_candidate_sat
        self.current_snr = best_snr
        self.current_latency = self._calculate_latency()
        self.last_handover_time = sim_time
        self.service_drop_until = sim_time
        self.ttt_candidate = None
        self.ttt_candidate_sat = None
        self.ttt_start_time = None

        return deltaT 

    def _calculate_3d_distance(self, satellite):
        """Calculates the 3D slant range from aircraft to satellite."""
        # Calculate 2D distance on Earth's surface
        dist_2d_km = geopy.distance.geodesic((self.latitude, self.longitude), 
                                            (math.degrees(satellite.latitude), 
                                            math.degrees(satellite.longitude))).km

        # Altitude difference
        altitude_diff_m = (satellite.h - self.height) # Satellite altitude - aircraft height

        # Total 3D distance using Pythagorean theorem (approximation)
        total_distance_m = math.sqrt((dist_2d_km * 1000)**2 + (altitude_diff_m)**2)
        return total_distance_m

    def _calculate_3d_distance_to_gateway(self, satellite):
        """Calculates the 3D slant range from satellite to gateway."""
        dist_2d_km = geopy.distance.geodesic(
            (GATEWAY_LAT, GATEWAY_LON),
            (math.degrees(satellite.latitude), math.degrees(satellite.longitude))
        ).km
        altitude_diff_m = satellite.h
        total_distance_m = math.sqrt((dist_2d_km * 1000)**2 + (altitude_diff_m)**2)
        return total_distance_m

    def _calculate_latency(self):
        """
        Calculates the propagation latency from the aircraft to the gateway via satellite.
        """
        if self.connected_satellite:
            aircraft_to_sat_m = self._calculate_3d_distance(self.connected_satellite)
            sat_to_gateway_m = self._calculate_3d_distance_to_gateway(self.connected_satellite)
            propagation_delay = (aircraft_to_sat_m + sat_to_gateway_m) / Vc
            return propagation_delay
        return 0           

class Passenger: 
    scenario = None
    def __init__(self, ID, application = None, rng=None):
        self.ID = ID
        self.application = application 
        self.rng = rng or random.Random()
        self.APP_PROBS = {
                        "streaming":      0.25,
                        "voip":           0.10,
                        "email":          0.10,
                        "web":            0.30,
                        "file_transfer":  0.10,
                        "video_conference": 0.15,
                    }
        if Passenger.scenario == "demand_aware":
            self.APP_PROBS = {
                        "streaming":      0.45,
                        "voip":           0.03,
                        "email":          0.02,
                        "web":            0.10,
                        "file_transfer":  0.15,
                        "video_conference": 0.25,
                    }
        
        self.APP_CLASSES = {
                            "streaming":        lambda: StreamingApp(),
                            "voip":             lambda: VoIPApp(),
                            "email":            lambda: EmailApp(),
                            "web":              lambda: WebBrowsingApp(),
                            "file_transfer":    lambda: FileTransferApp(),
                            "video_conference": lambda: VideoConferenceApp(),  
                        }

    def step_application(self, dt_seconds=10):
        # 1) If active, advance session time
        if self.application is not None:
            self.application.step_session_time(dt_seconds)
            if not self.application.session_active():
                self.application = None
        
        # 2) If idle, maybe start a new app
        if self.application is None:
            start_prob = 0.5  # 50% chance per 10 seconds
            if self.rng.random() < start_prob:
                apps  = list(self.APP_PROBS.keys())
                probs = list(self.APP_PROBS.values())
                app_name = self.rng.choices(apps, weights=probs, k=1)[0]
                self.application = self.APP_CLASSES[app_name]()  # Each app now uses seconds

        # 3) Demand
        if self.application is None: 
            print("No application active")
        return (0.0, 0.0, 0.0) if self.application is None else (self.application.get_demand(dt_seconds), self.application.base_demand_mbps, self.application.latency_ms)

        
class StreamingApp:
    def __init__(self, app_type='video', base_demand_mbps=5, latency_ms=300, duration=40*60, sessionid = None):
        self.sessionid = sessionid
        self.app_type = app_type
        self.base_demand_mbps = base_demand_mbps
        self.latency_ms = latency_ms
        self.duration = duration
        self.session_time = 0 

    def step_session_time(self, time_increment): 
        self.session_time += time_increment

    def session_active(self):
        return self.session_time < self.duration

    def get_demand(self, time):
        # Simple model: demand is constant over duration
        if self.session_active():
            return self.base_demand_mbps
        else:
            return 0

class VoIPApp:
    def __init__(self, app_type='voip', base_demand_mbps=0.1, latency_ms=100, duration=5*60, sessionid = None):
        self.sessionid = sessionid
        self.app_type = app_type
        self.base_demand_mbps = base_demand_mbps
        self.latency_ms = latency_ms
        self.duration = duration
        self.session_time = 0 

    def step_session_time(self, time_increment): 
        self.session_time += time_increment

    def session_active(self):
        return self.session_time < self.duration
    
    def get_demand(self, time):
        if self.session_active():
            return self.base_demand_mbps
        else:
            return 0

class EmailApp:
    def __init__(self, app_type='email', base_demand_mbps=0.5, latency_ms=200, duration=3*60, sessionid = None):
        self.sessionid = sessionid
        self.app_type = app_type
        self.base_demand_mbps = base_demand_mbps
        self.latency_ms = latency_ms
        self.duration = duration
        self.session_time = 0 

    def step_session_time(self, time_increment): 
        self.session_time += time_increment

    def session_active(self):
        return self.session_time < self.duration

    def get_demand(self, time):
        if self.session_active():
            return self.base_demand_mbps
        else:
            return 0

class WebBrowsingApp:
    def __init__(self, app_type='web', base_demand_mbps=1, latency_ms=500, duration=10*60, sessionid = None):
        self.sessionid = sessionid
        self.app_type = app_type
        self.base_demand_mbps = base_demand_mbps    
        self.latency_ms = latency_ms
        self.duration = duration
        self.session_time = 0 

    def step_session_time(self, time_increment): 
        self.session_time += time_increment

    def session_active(self):
        return self.session_time < self.duration

    def get_demand(self, time):
        if self.session_active():
            return self.base_demand_mbps
        else:
            return 0

class FileTransferApp:
    def __init__(self, app_type='file_transfer', base_demand_mbps=10, latency_ms=1000, duration=10*60, sessionid = None):
        self.sessionid = sessionid
        self.app_type = app_type
        self.base_demand_mbps = base_demand_mbps
        self.latency_ms = latency_ms
        self.duration = duration
        self.session_time = 0 

    def step_session_time(self, time_increment): 
        self.session_time += time_increment

    def session_active(self): 
        return self.session_time < self.duration

    def get_demand(self, time):
        if self.session_active():
            return self.base_demand_mbps
        else:
            return 0

class VideoConferenceApp:
    def __init__(self, app_type='video_conference', base_demand_mbps=2, latency_ms=150, duration=45*60, sessionid = None):
        self.sessionid = sessionid
        self.app_type = app_type
        self.base_demand_mbps = base_demand_mbps
        self.latency_ms = latency_ms
        self.duration = duration
        self.session_time = 0 

    def step_session_time(self, time_increment): 
        self.session_time += time_increment

    def session_active(self): 
        return self.session_time < self.duration

    def get_demand(self, time):
        if self.session_active():
            return self.base_demand_mbps
        else:
            return 0            

# Earth consisting of cells
class Earth:
    def __init__(self, env, constellation, aircraft, window=None):
        self.deltaT = 0 
        
        [self.total_x, self.total_y] = [1920, 906]

        self.total_cells = self.total_x * self.total_y
        self.constellationType = constellation  # Backward-compatible storage

        # window is a list with the coordinate bounds of our window of interest
        # format for window = [western longitude, eastern longitude, southern latitude, northern latitude]
        if window is not None:  # if window provided
            # latitude, longitude bounds:
            self.lati = [window[2], window[3]]
            self.longi = [window[0], window[1]]
            # dataset pixel bounds:
            self.windowx = (
            (int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
            self.windowy = (
            (int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))
        else:  # set window size as entire world if no window provided
            self.lati = [-90, 90]
            self.longi = [-179, 180]
            self.windowx = (0, self.total_x)
            self.windowy = (0, self.total_y)

        # Create one or more constellations.
        self.constellation_names = self._normalize_constellation_names(constellation)
        self.constellations = {}
        for cname in self.constellation_names:
            planes = create_Constellation(cname, env)
            # Tag satellites with source constellation name for per-aircraft filtering.
            for plane in planes:
                for sat in plane.sats:
                    sat.constellation_name = cname
            self.constellations[cname] = planes
        # Backward-compatible flattened list of orbital planes.
        self.LEO = [plane for planes in self.constellations.values() for plane in planes]
        # create aircrafts 
        self.aircraft = aircraft 
        self.env = env  # simpy environment
        self.img_count = 0
        self.Training = True

    @staticmethod
    def _normalize_constellation_names(constellation):
        if isinstance(constellation, str):
            # Supports: "OneWeb", "OneWeb,Intelsat", "OneWeb+Intelsat"
            tokens = constellation.replace("+", ",").split(",")
            names = [tok.strip() for tok in tokens if tok.strip()]
            return names if names else [constellation]
        if isinstance(constellation, (list, tuple, set)):
            names = [str(c).strip() for c in constellation if str(c).strip()]
            return names if names else ["OneWeb"]
        return [str(constellation)]

    def _get_constellation_planes_for_aircraft(self, aircraft):
        allowed = getattr(aircraft, "supported_constellations", None)
        if not allowed:
            return self.LEO
        return [
            plane
            for cname, planes in self.constellations.items()
            if cname in allowed
            for plane in planes
        ]

    def set_window(self, window):  # function to change/set window for the earth
        """
        Unused function
        """
        self.lati = [window[2], window[3]]
        self.longi = [window[0], window[1]]
        self.windowx = ((int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
        self.windowy = ((int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))

    def advance_constellation(self, deltaT, env_time):
        # rotate constellation and satellites
        for constellation in self.LEO:
            constellation.rotate(deltaT, env_time)
  
    def step_aircraft(self, folder="PPO", actions=None):
        qoe_metrics = []
        for ac_idx, ac in enumerate(self.aircraft):
            planes_for_ac = self._get_constellation_planes_for_aircraft(ac)
            deltaT = ac.move_and_connect_aircraft(planes_for_ac)
            self.deltaT = deltaT 
            if isinstance(actions, dict):
                if ac_idx in actions:
                    ac.set_steering_action(action_indices=actions[ac_idx])
                elif ac.id in actions:
                    ac.set_steering_action(action_indices=actions[ac.id])
                else:
                    ac.set_steering_action(action_indices=None)
            elif actions is not None:
                ac.set_steering_action(action_indices=actions)
            else:
                ac.set_steering_action(action_indices=None)
            qoe_metrics.append(ac.get_qoe_metrics(deltaT))
            #print(f"Aircraft {ac.id} scan:")
            #print(f"Allocation ratio: {ratio:.2f} | Allocated: {allocated:.2f} MB | Demand: {demand:.2f} MB | Beam cap: {beam_capacity_MB:.2f} MB")
        if self.Training == False: 
            self.save_plot(self.env, plotSat=True, plotBeams=True, plotAircrafts=True, aircrafts=self.aircraft, folder=folder)
        return qoe_metrics

    def plotMap(self, plotSat = True, plotBeams = True, plotAircrafts = True, aircrafts=None, selected_beam_id=None, path = None, bottleneck = None):
        print("Plotting map")
        fig = plt.figure(figsize=(15, 8))  # Make figure slightly wider for text
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_global()

        # Generate a unique color for each orbital plane
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(self.LEO)))
        
        # Plot satellites with color per plane
        if plotSat:
            plotted_sat_label = False
            for plane, c in zip(self.LEO, colors):
                for sat in plane.sats:
                    lon = math.degrees(sat.longitude)
                    lat = math.degrees(sat.latitude)
                    # Use the color 'c' for the current plane
                    if not plotted_sat_label:
                        ax.scatter(lon, lat, color=c, s=10, transform=ccrs.PlateCarree(), label='Satellites')
                        plotted_sat_label = True
                    else:
                        ax.scatter(lon, lat, color=c, s=10, transform=ccrs.PlateCarree())

        # Plot beams
        if plotBeams:
            for plane in self.LEO:
                for sat in plane.sats:
                    for beam in sat.beams:
                        # Determine color and linewidth based on whether it's the selected beam
                        is_selected = selected_beam_id and beam.id == selected_beam_id
                        edge_color = 'red' if is_selected else 'gray'
                        line_width = 1.5 if is_selected else 0.2
                        alpha_val = 1.0 if is_selected else 0.3
                        z_order = 10 if is_selected else 2 # Draw selected beam on top

                        footprint = beam.get_footprint_eclipse()
                        feature = ShapelyFeature([footprint], ccrs.PlateCarree(), edgecolor=edge_color, facecolor='none', linewidth=line_width, alpha=alpha_val)
                        ax.add_feature(feature, zorder=z_order)
        
        # Plot Aircrafts and connection lines
        if plotAircrafts and aircrafts:
            aircraft_lons = [ac.longitude for ac in aircrafts]
            aircraft_lats = [ac.latitude for ac in aircrafts]
            # ax.scatter(aircraft_lons, aircraft_lats, color='black', marker='x', s=50, transform=ccrs.PlateCarree(), label='Aircraft', zorder=20)
            for lon, lat in zip(aircraft_lons, aircraft_lats):
                ax.text(lon, lat, '✈', fontsize=18, color='black', transform=ccrs.PlateCarree(), ha='center', va='center', zorder=20)
            
            # Optionally plot aircraft connection lines
            plotted_conn_label = False
            for aircraft in aircrafts:
                if aircraft.connected_satellite:
                    # Convert to degrees for plotting
                    line_lons = [aircraft.longitude, math.degrees(aircraft.connected_satellite.longitude)]
                    line_lats = [aircraft.latitude, math.degrees(aircraft.connected_satellite.latitude)]
                    if not plotted_conn_label:
                        ax.plot(line_lons, line_lats, color='green', linewidth=1.0, linestyle='--', transform=ccrs.Geodetic(), label='Connection', zorder=15)
                        plotted_conn_label = True
                    else:
                        ax.plot(line_lons, line_lats, color='green', linewidth=1.0, linestyle='--', transform=ccrs.Geodetic(), zorder=15)

            # Add aircraft metrics as text annotations
            for i, aircraft in enumerate(aircrafts):
                # Calculate metrics
                avg_allocation = sum(aircraft.allocation_ratios) / len(aircraft.allocation_ratios) if aircraft.allocation_ratios else 0
                total_handovers = aircraft.handover_count
                total_data = aircraft.total_allocated_bandwidth
                total_demand = aircraft.total_demand

                # Position text near aircraft (offset to avoid overlap)
                text_lon = aircraft.longitude + 10 + (i * 20)  # Offset each aircraft's text
                text_lat = aircraft.latitude + 5
                
                # Create metrics text
                metrics_text = f"Aircraft {aircraft.id}:\n" \
                            f"Avg Allocation: {avg_allocation:.2%}\n" \
                            f"Handovers: {total_handovers}\n" \
                            f"Data: {total_data:.1f} MB \n" \
                            f"Total Demand: {total_demand:.1f} MB"

                # Add text annotation with background box
                ax.text(text_lon, text_lat, metrics_text,
                    transform=ccrs.PlateCarree(),
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='black'),
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    zorder=25)

        # Create summary statistics text box
        if aircrafts:
            # Calculate aggregate statistics
            total_aircraft = len(aircrafts)
            total_handovers_all = sum(ac.handover_count for ac in aircrafts)
            total_data_all = sum(ac.total_allocated_bandwidth for ac in aircrafts)
            total_demand_all = sum(ac.total_demand for ac in aircrafts)

            # Calculate overall average allocation ratio
            all_ratios = []
            for ac in aircrafts:
                all_ratios.extend(ac.allocation_ratios)
            overall_avg_allocation = sum(all_ratios) / len(all_ratios) if all_ratios else 0
            
            summary_text = f"SUMMARY STATISTICS\n" \
                        f"Total Aircraft: {total_aircraft}\n" \
                        f"Total Handovers: {total_handovers_all}\n" \
                        f"Total Data Transmitted: {total_data_all:.1f} MB\n" \
                        f"Overall Avg Allocation: {overall_avg_allocation:.2%} \n" \
                        f"Total Demand: {total_demand_all:.1f} MB"

            # Add summary text box in top-left corner
            ax.text(0.02, 0.98, summary_text,
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.9, edgecolor='navy'),
                verticalalignment='top',
                horizontalalignment='left',
                zorder=30,
                weight='bold')

        # Create a single legend for all plotted elements
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            plt.legend(by_label.values(), by_label.keys(), loc='lower left', prop={'size': 8})
        
        # Add title with timestamp
        plt.title(f"LEO Satellite Network - Time: {self.env.now:.1f}s", fontsize=14, weight='bold')

    def plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs = []
        ys = []
        zs = []
        for con in self.LEO:
            for sat in con.sats:
                xs.append(sat.x)
                ys.append(sat.y)
                zs.append(sat.z)
        ax.scatter(xs, ys, zs, marker='o')
        plt.show()

    def save_plot(self, env, plotSat=True, plotBeams=True, plotAircrafts=False, aircrafts=None, folder = "PPO"):
        if not os.path.exists("simulationImages"):
            os.makedirs("simulationImages")
        print(f"\nSaving plot {self.img_count} at simulation time {env.now}")
        # Get the ID of the connected beam for highlighting
        selected_beam_id = aircrafts[0].connected_beam.id if aircrafts and aircrafts[0].connected_beam else None
        self.plotMap(plotSat=plotSat, plotBeams=plotBeams, plotAircrafts=plotAircrafts, aircrafts=aircrafts, selected_beam_id=selected_beam_id)
        plt.savefig(f"simulationImages/{folder}/sat_positions_{self.img_count}.png")
        plt.close()
        self.img_count += 1          

    def __repr__(self):
        return 'total divisions in x = {}\n total divisions in y = {}\n total cells = {}\n window of operation ' \
               '(longitudes) = {}\n window of operation (latitudes) = {}'.format(
                self.total_x,
                self.total_y,
                self.total_cells,
                self.windowx,
                self.windowy)


###############################################################################
###############################    Functions    ###############################
###############################################################################


def initialize(env, constellationType, route, scenario=None, demand_seed=42, handover_seed=123, aircraft_constellations=None):
    """
    Initializes an instance of the earth with cells from a population map and gateways from a csv file.
    During initialisation, several steps are performed to prepare for simulation:
        - GTs find the cells that within their ground coverage areas and "link" to them.
        - A certain LEO Constellation with a given architecture is created.
        - Satellites are distributed out to GTs so each GT connects to one satellite (if possible) and each satellite
        only has one connected GT.
        - A graph is created from all the GSLs and ISLs
        - Paths are created from each GT to all other GTs
        - Buffers and processes are created on all GTs and satellites used for sending the blocks throughout the network
    """

    # Load earth, aircraft, passengers and consellations 
    Beam.scenario = scenario
    Beam.start_utc_hour = 0.0
    load_cycles = 1.0
    if scenario == "load_cycle_2":
        load_cycles = 2.0
    elif scenario == "load_cycle_5":
        load_cycles = 5.0
    elif scenario == "load_cycle_1":
        load_cycles = 1.0
    if route and len(route) > 1:
        route_duration_s = (route[-1]["time"] - route[0]["time"]).total_seconds()
        Beam.load_cycle_s = max(1.0, route_duration_s / load_cycles)
    else:
        Beam.load_cycle_s = 5400.0
    if scenario == "peak_hour":
        Beam.start_utc_hour = 17.75
    Passenger.scenario = scenario
    Aircraft.scenario = scenario
    passenger_count = 5
    if scenario == "medium_aircraft":
        passenger_count = 15
    elif scenario == "large_aircraft":
        passenger_count = 25
    passengers = [
        Passenger(f"P-{i:03d}", rng=random.Random(demand_seed + i))
        for i in range(1, passenger_count + 1)
    ]
    handover_rng = random.Random(handover_seed)
    if aircraft_constellations is None:
        if isinstance(constellationType, str):
            aircraft_constellations = [c.strip() for c in constellationType.replace("+", ",").split(",") if c.strip()]
        elif isinstance(constellationType, (list, tuple, set)):
            aircraft_constellations = [str(c).strip() for c in constellationType if str(c).strip()]
    aircraft1 = Aircraft(
        env,
        "A-380",
        route=route,
        height=10000,
        update_interval=10,
        passengers=passengers,
        handover_rng=handover_rng,
        supported_constellations=aircraft_constellations,
    )
    # aircraft1 = Aircraft(env, "A-380", start_lat=37.77, start_lon=-122.41, height=10000, speed_kmph=8000000, direction_deg=345, update_interval=10)  # Example aircraft at San Francisco
    earth = Earth(env, constellationType, [aircraft1])

    print("Initialized Earth")
    print(earth)
    print()

    return earth


def create_Constellation(specific_constellation, env):

    if specific_constellation == "small":               # Small Walker star constellation for tests.
        print("Using small walker Star constellation")
        P = 3					# Number of orbital planes
        N_p = 4 				# Number of satellites per orbital plane
        N = N_p*P				# Total number of satellites
        height = 1000e3			# Altitude of deployment for each orbital plane (set to the same altitude here)
        inclination_angle = 53	# Inclination angle for the orbital planes, set to 90 for Polar 
        Walker_star = True		# Set to True for Walker star and False for Walker Delta
        min_elevation_angle = 30

    elif specific_constellation =="Kepler":
        print("Using Kepler constellation design")
        P = 7
        N_p = 20
        N = N_p*P
        height = 600e3
        inclination_angle = 98.6
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="Iridium_NEXT":
        print("Using Iridium NEXT constellation design")
        P = 6
        N_p = 11
        N = N_p*P
        height = 780e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="OneWeb":
        print("Using OneWeb constellation design")
        P = 18
        N = 648	# Number of satellites 
        N_p = int(N/P)
        height = 1200e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30
        orbital_speed = 27000 #km/h 
        orbit_duration = 109 #minutes 
        orbits_per_day = 13 


    elif specific_constellation =="Starlink":			# Phase 1 550 km altitude orbit shell
        print("Using Starlink constellation design")
        P = 72
        N = 1584
        N_p = int(N/P)
        height = 550e3
        inclination_angle = 53
        Walker_star = False
        min_elevation_angle = 25

    elif specific_constellation == "Test":
        print("Using a test constellation design")
        P = 30                     # Number of orbital planes
        N = 1200                   # Total number of satellites
        N_p = int(N/P)             # Number of satellites per orbital plane
        height = 600e3             # Altitude of deployment for each orbital plane (set to the same altitude here)
        inclination_angle = 86.4   # Inclination angle for the orbital planes, set to 90 for Polar
        Walker_star = True         # Set to True for Walker star and False for Walker Delta
        min_elevation_angle = 30
    elif specific_constellation in ("intelsat", "Intelsat"):
        print("Using Intelsat GEO constellation design")
        P = 1
        N = 2
        N_p = 2
        height = 35786e3
        inclination_angle = 0
        Walker_star = False
        min_elevation_angle = 10
    else:
        print("Not valid Constellation Name")
        P = np.NaN
        N_p = np.NaN
        N = np.NaN
        height = np.NaN
        inclination_angle = np.NaN
        Walker_star = False
        exit()
    
    distribution_angle = 2*math.pi  # Angle in which the orbital planes are distributed in
    
    if Walker_star:
        distribution_angle /= 2
    orbital_planes = []

    # Add orbital planes and satellites
    # Orbital_planes.append(orbital_plane(0, height, 0, math.radians(inclination_angle), N_p, min_elevation_angle, 0))
    constellation_type = "Intelsat" if specific_constellation in ("intelsat", "Intelsat") else "OneWeb"
    for i in range(0, P):
        orbital_planes.append(
            OrbitalPlane(
                str(i),
                height,
                i * distribution_angle / P,
                math.radians(inclination_angle),
                N_p,
                min_elevation_angle,
                str(i) + '_',
                env,
                constellation_type=constellation_type,
            )
        )

    if specific_constellation in ("intelsat", "Intelsat"):
        # GEO slots (first two satellites requested so far).
        geo_slots_deg = [-91.0, -89.0]
        for idx, sat in enumerate(orbital_planes[0].sats):
            sat._set_geo_slot(geo_slots_deg[idx % len(geo_slots_deg)])
            sat.update_beams()

    return orbital_planes


class MultiOrbitSteeringEnv:
    """
    Lightweight RL wrapper for masked traffic-steering actions.
    Action: length-M vector of candidate indices (one per app slot).
    Observation: dict from Aircraft.get_steering_observation().
    """
    def __init__(
        self,
        route,
        constellation="OneWeb+Intelsat",
        scenario=None,
        aircraft_constellations=None,
        demand_seed=42,
        handover_seed=123,
        base_seed=1,
        reward_weights=None,
    ):
        self.route = route
        self.constellation = constellation
        self.scenario = scenario
        self.aircraft_constellations = aircraft_constellations
        self.demand_seed = demand_seed
        self.handover_seed = handover_seed
        self.base_seed = base_seed
        self.episode = 0

        self.reward_weights = reward_weights or {
            "alloc": 1.0,
            "drop": 0.8,
            "lat": 0.6,
        }

        self.env = None
        self.earth = None
        self.aircraft = None
        self.last_obs = None
        self.step_idx = 0
        self.max_steps = max(1, len(route) - 1) if route else 1

    def _seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _sanitize_action_with_mask(action_indices, action_mask):
        if action_mask is None:
            return action_indices
        mask = np.asarray(action_mask)
        m, k = mask.shape
        out = [0] * m
        for i in range(m):
            valid = np.where(mask[i] > 0.5)[0]
            if len(valid) == 0:
                out[i] = 0
                continue
            a = 0
            if isinstance(action_indices, (list, tuple, np.ndarray)) and i < len(action_indices):
                ai = action_indices[i]
                if isinstance(ai, (int, np.integer)):
                    a = int(ai)
            if a < 0 or a >= k or mask[i, a] <= 0.5:
                out[i] = int(valid[0])
            else:
                out[i] = a
        return out

    def _build_reward(self, qoe):
        alloc_ratio = float(qoe.get("allocation_ratio", 0.0))
        delta_t = max(1e-9, float(self.earth.deltaT if self.earth else 1.0))
        drop_norm = float(qoe.get("service_drop_s", 0.0)) / delta_t
        lat_violation = float(qoe.get("latency_violation_rate", 0.0))
        w = self.reward_weights
        reward = (
            w["alloc"] * alloc_ratio
            - w["drop"] * drop_norm
            - w["lat"] * lat_violation
        )
        return float(np.clip(reward, -1.0, 1.0))

    def reset(self, seed=None):
        if seed is None:
            seed = self.base_seed + self.episode
        self._seed_all(seed)
        self.episode += 1
        self.step_idx = 0

        self.env = simpy.Environment()
        self.earth = initialize(
            self.env,
            self.constellation,
            self.route,
            scenario=self.scenario,
            demand_seed=self.demand_seed + seed,
            handover_seed=self.handover_seed + seed,
            aircraft_constellations=self.aircraft_constellations,
        )
        self.aircraft = self.earth.aircraft[0]
        # Prime one decision state with no external action.
        qoe_list = self.earth.step_aircraft(actions=None)
        qoe = qoe_list[0] if qoe_list else {}
        self.last_obs = self.aircraft.get_steering_observation()
        info = {"qoe": qoe, "seed": seed}
        return self.last_obs, info

    def step(self, action):
        if self.earth is None:
            raise RuntimeError("Call reset() before step().")

        current_mask = self.last_obs.get("action_mask") if isinstance(self.last_obs, dict) else None
        safe_action = self._sanitize_action_with_mask(action, current_mask)
        qoe_list = self.earth.step_aircraft(actions=safe_action)
        qoe = qoe_list[0] if qoe_list else {}

        # Advance simpy time for deterministic timer progression.
        dt = max(1.0, float(self.earth.deltaT if self.earth else 1.0))
        self.env.run(until=self.env.now + dt)

        reward = self._build_reward(qoe)
        self.last_obs = self.aircraft.get_steering_observation()
        self.step_idx += 1
        done = self.step_idx >= self.max_steps
        truncated = False
        info = {"qoe": qoe}
        return self.last_obs, reward, done, truncated, info


# Global variable to store the earth instance, so Aircraft class can access it
earth_instance = None

def main():
    """
    This function is made to avoid problems with scope. everything in if __name__ = "__main__" is in global scope which
    can be an issue.
    """
    global earth_instance # Declare as global


    inputParams = pd.read_csv("input.csv")

    testLength = inputParams['Test length'][0]
    constellation_name = inputParams['Constellation'][0]

    # Time interval for constellation movement updates
    movementTime = 20
    print(f"Constellation: {constellation_name}")
    print(f"Constellation movement update interval: {movementTime} seconds")

    simulationTimelimit = testLength
    print(f"Simulation test length: {simulationTimelimit} seconds")

    env = simpy.Environment()
    route, route_duration = load_route_from_csv('route.csv', skip_rows=3)
    earth_instance = initialize(env, constellation_name, route)

    # --- Aircraft Simulation ---
    all_aircrafts = earth_instance.aircraft # List of all aircrafts to pass to plotting function

    # Start plotting process to save map images at intervals
    # plotAircrafts is set to True to show aircraft on the map
    #env.process(earth_instance.save_plot_at_intervals(env, interval=movementTime, plotSat=True, plotBeams=True, plotAircrafts=True, aircrafts=all_aircrafts))

    #progress = env.process(simProgress(simulationTimelimit, env))
    startTime = time.time()
    env.run(route_duration)
    timeToSim = time.time() - startTime

    for aircraft in all_aircrafts:
        print("\n\n--- Simulation Summary ---")
        print(f"Total simulation run time: {timeToSim:.2f} seconds")
        print(f"Aircraft '{aircraft.id}' total handovers: {aircraft.handover_count}")
        if aircraft.connected_beam:
            print(f"Aircraft '{aircraft.id}' final connected beam: {aircraft.connected_beam.id}")
            print(f"Aircraft '{aircraft.id}' final SNR: {aircraft.current_snr:.2f} dB")
            print(f"Aircraft '{aircraft.id}' total allocated BW: {aircraft.total_allocated_bandwidth:.2f} MB")
            print(f"Aircraft '{aircraft.id}' Average Allocation to demand: {sum(aircraft.allocation_ratios)/len(aircraft.allocation_ratios):.2f}")
            print(f"Aircraft '{aircraft.id}' final Latency: {aircraft.current_latency*1e3:.2f} ms")
        else:
            print(f"Aircraft '{aircraft.id}' ended the simulation with no connection.")


if __name__ == '__main__':
    main()
