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
from shapely.geometry import Polygon
from shapely.affinity import scale, rotate, translate
from scipy.spatial import KDTree
import numpy as np


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

    def rotate(self, delta_t):
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

class Beam:
    def __init__(self, center_lat, center_lon, width_deg, height_deg, 
                 load=0, capacity=10, snr=0, id=None, constellation = 'OneWeb'): # Default capacity added
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.width_deg = width_deg
        self.height_deg = height_deg
        self.load = load # Current number of users connected
        self.capacity = capacity # Max number of users or data rate
        self.snr = snr # Base SNR, will be adjusted for user position
        self.id = id  # Optional: unique identifier for the beam
        
        # Calculate ellipse parameters based on width_deg and height_deg
        self._calculate_ellipse_parameters()

        if constellation == 'OneWeb': 
            self.max_capacity = 7.2 #Gbps 
            self.max_ds_speed = 150 #Mbps 
            self.max_us_speed = 30 #Mbps 
            self.max_latency = 70 #ms 
            self.bw = 250e6 #Hz 
            self.frequency = 15 #GHz 
            self.Pt = 40 #dBm - transmit power 
            self.Gt = 30 #dBi - antenna gain 
        
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

class Satellite:
    def __init__(self, ID, in_plane, i_in_plane, h, longitude, inclination, n_sat, env, quota = 500, power = 10):
        self.ID = ID                    # A unique ID given to every satellite
        self.in_plane = in_plane        # Orbital plane where the satellite is deployed
        self.i_in_plane = i_in_plane    # Index in orbital plane
        self.quota = quota              # Quota of the satellite
        self.h = h                      # Altitude of deployment
        self.power = power              # Transmission power
        self.minElevationAngle = 30     # Value is taken from NGSO constellation design chapter.
        self.constellationType = "OneWeb"  # Type of constellation, used for beam initialization

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

    def init_beams(self):
        # OneWeb: coverage area is a square, divided into 16 horizontal strips (vertically stacked)
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
                # Each beam's center is halfway between its top and bottom edge
                beam_top = top_lat - i * beam_height_deg
                beam_bottom = beam_top - beam_height_deg
                center_lat = (beam_top + beam_bottom) / 2
                center_lon = sat_lon
                beam_id = f"{self.ID}_beam_{i+1}"  # Number beams 1 to 16
                self.beams.append(Beam(center_lat, center_lon, beam_width_deg, beam_height_deg, id=beam_id))
        else:
            # Center beam
            self.beams.append(Beam(sat_lat, sat_lon, beam_width_deg, beam_height_deg))

            # Beams on circle
            n_circle_beams = n_beams - 1
            radius_deg = beam_height_deg * 4 # Circle radius in degrees

            for i in range(n_circle_beams):
                angle = 2 * math.pi * i / n_circle_beams
                dlat = radius_deg * math.cos(angle)
                dlon = radius_deg * math.sin(angle) / math.cos(math.radians(sat_lat))  # adjust for latitude distortion
                center_lat = sat_lat + dlat
                center_lon = sat_lon + dlon
                self.beams.append(Beam(center_lat, center_lon, beam_width_deg, beam_height_deg))

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

    def adjustDownRate(self):

        speff_thresholds = np.array(
            [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
             1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
             2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
             3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
             5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
        lin_thresholds = np.array(
            [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
             1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
             3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
             8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
             16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
             45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])
        db_thresholds = np.array(
            [-100.00000, -2.85000, -2.35000, -2.03000, -1.24000, -0.30000, 0.22000, 1.00000, 1.45000, 2.23000, 3.10000,
             4.03000, 4.68000, 4.73000, 5.13000, 5.50000, 5.97000, 6.55000, 6.84000, 7.41000, 8.10000, 8.38000, 8.43000,
             8.97000, 9.27000, 9.71000, 10.21000, 10.65000, 11.03000, 11.10000, 11.61000, 11.75000, 12.17000, 12.73000,
             13.05000, 13.64000, 13.98000, 14.81000, 15.47000, 15.87000, 16.55000, 16.98000, 17.24000, 18.10000,
             18.59000, 18.84000, 19.57000])

        pathLoss = 10*np.log10((4*math.pi*self.linkedGT.linkedSat[0]*self.ngeo2gt.f/Vc)**2)
        snr = 10**((self.ngeo2gt.maxPtx_db + self.ngeo2gt.G - pathLoss - self.ngeo2gt.No)/10)
        shannonRate = self.ngeo2gt.B*np.log2(1+snr)

        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr)]
        speff = self.ngeo2gt.B * feasible_speffs[-1]

        self.downRate = speff


    def rotate(self, delta_t, longitude, period):
        """
        Rotates the satellite by re-calculating the sperical coordinates, Cartesian coordinates, and longitude and
        latitude adjusted for the new longitude of the orbit, and fraction the elapsed time makes up of the orbit time
        of the satellite.
        """
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

class Aircraft: 
    def __init__(self, env, aircraft_id, start_lat, start_lon, height, speed_kmph, direction_deg, update_interval=1):
        self.env = env
        self.id = aircraft_id
        self.latitude = start_lat  # in degrees
        self.longitude = start_lon  # in degrees
        self.height = height  # in meters
        self.speed_kmph = speed_kmph
        self.direction_rad = math.radians(direction_deg) # Direction in radians (0 = East, 90 = North)
        self.update_interval = update_interval # How often aircraft position and connection is updated
        self.Gr = 5 #dBi - Gain of the aircraft antenna (Supervisor's value)

        self.connected_satellite = None
        self.connected_beam = None
        self.current_snr = 0
        self.current_latency = 0
        self.handover_count = 0

        self.env.process(self.move_and_connect_aircraft())
        print(f"Aircraft {self.id} initialized at ({self.latitude:.2f}, {self.longitude:.2f})")

    def __repr__(self):
        return 'Aircraft ID: {}, Latitude: {}, Longitude: {}, Altitude: {}'.format(
            self.id, self.latitude, self.longitude, self.height)

    def calculate_snr(self, beam, distance_km):
        # Supervisor's SNR Calculation Method
        # Constants
        T = 290  # Noise temperature (K)
        
        # Beam parameters (assumed to be in dBm, dBi, Hz, etc.)
        Pt_dBm = beam.Pt  # Transmit power in dBm
        Gt_dBi = beam.Gt  # Transmit antenna gain in dBi
        Gr_dBi = self.Gr  # Aircraft receive antenna gain in dBi
        f_Hz = beam.frequency * 1e9  # Frequency in Hz (beam.frequency in GHz)
        B_Hz = beam.bw  # Bandwidth in Hz

        # Convert dBm to dBW
        Pt_dBW = Pt_dBm - 30

        # Free-space path loss (FSPL) in dB
        d_m = distance_km * 1000
        if d_m == 0: return float('inf') # Avoid log(0)
        FSPL_dB = 20 * math.log10(d_m) + 20 * math.log10(f_Hz) - 147.55

        # Received power in dBW
        Pr_dBW = Pt_dBW + Gt_dBi + Gr_dBi - FSPL_dB

        # Noise power in dBW
        N_dBW = 10 * math.log10(k * T * B_Hz)

        # SNR in dB
        SNR_dB = Pr_dBW - N_dBW
        return SNR_dB

    def scan_nearby_fast(self, earth, threshold_km=1500):
        # Supervisor's efficient scanning method using KDTree
        # Gather all beam centers and references
        beam_coords = []
        beam_refs = []
        for plane in earth.LEO:
            for sat in plane.sats:
                for beam in sat.beams:
                    beam_coords.append([beam.center_lat, beam.center_lon])
                    beam_refs.append((sat, beam))
        
        if not beam_refs:
            return None, -np.inf

        beam_coords = np.array(beam_coords)

        # Build KDTree
        tree = KDTree(beam_coords)

        # Query for all beams within threshold (in degrees, approx)
        threshold_deg = threshold_km / 111.0
        idxs = tree.query_ball_point([self.latitude, self.longitude], r=threshold_deg)

        best_snr = -np.inf
        best_candidate_beam = None
        best_candidate_sat = None

        print(f"\n[SimTime {self.env.now:.2f}] Aircraft {self.id} scan:")
        for idx in idxs:
            sat, beam = beam_refs[idx]
            
            # Use 3D distance for SNR calculation
            dist_3d = self._calculate_3d_distance(sat)
            snr = self.calculate_snr(beam, dist_3d / 1000) # distance in km
            
            # Check if aircraft is within the beam's elliptical footprint
            aircraft_point = Point(self.longitude, self.latitude)
            if beam.get_footprint_eclipse().contains(aircraft_point):
                 print(f"  - Beam {beam.id} on Sat {sat.ID} | Distance: {dist_3d/1000:.2f} km | SNR: {snr:.2f} dB (In Footprint)")
                 if snr > best_snr and (beam.load < beam.capacity):
                    best_snr = snr
                    best_candidate_beam = beam
                    best_candidate_sat = sat
            else:
                 print(f"  - Beam {beam.id} on Sat {sat.ID} | Distance: {dist_3d/1000:.2f} km | SNR: {snr:.2f} dB (Outside Footprint)")

        return best_candidate_sat, best_candidate_beam, best_snr

    def move_and_connect_aircraft(self):
        """
        SimPy process for aircraft movement and connection management.
        """
        while True:
            # 1. Move the aircraft
            self._update_position()

            # 2. Find the best beam using the efficient KDTree scan
            best_candidate_sat, best_candidate_beam, best_snr = self.scan_nearby_fast(earth_instance)
            
            # 3. Manage connection and handover
            if best_candidate_beam and best_candidate_beam != self.connected_beam:
                # Handover or initial connection
                if self.connected_beam:
                    # Disconnect from old beam
                    self.connected_beam.load -= 1
                    self.handover_count += 1
                    print(f"Time {self.env.now:.2f}: Aircraft {self.id} HANDOVER from {self.connected_beam.id} to {best_candidate_beam.id}. Handovers: {self.handover_count}")
                else:
                    print(f"Time {self.env.now:.2f}: Aircraft {self.id} CONNECTED to {best_candidate_beam.id} (Initial connection)")
                
                # Connect to new beam
                self.connected_beam = best_candidate_beam
                self.connected_satellite = best_candidate_sat
                self.connected_beam.load += 1
                self.current_snr = best_snr
                self.current_latency = self._calculate_latency() # Update latency
                print(f"  >> New Status: SNR: {self.current_snr:.2f} dB, Latency: {self.current_latency*1e3:.2f} ms, Beam Load: {self.connected_beam.load}/{self.connected_beam.capacity}")

            elif not best_candidate_beam and self.connected_beam:
                # Lost connection
                print(f"Time {self.env.now:.2f}: Aircraft {self.id} LOST connection from {self.connected_beam.id}")
                self.connected_beam.load -= 1
                self.connected_beam = None
                self.connected_satellite = None
                self.current_snr = 0
                self.current_latency = 0
            
            elif best_candidate_beam and best_candidate_beam == self.connected_beam:
                # Still connected to the same beam, just update SNR/latency
                self.current_snr = best_snr
                self.current_latency = self._calculate_latency()
                print(f"Time {self.env.now:.2f}: Aircraft {self.id} remains connected to {self.connected_beam.id}. SNR: {self.current_snr:.2f} dB, Latency: {self.current_latency*1e3:.2f} ms")


            yield self.env.timeout(self.update_interval)

    def _update_position(self):
        """
        Updates aircraft's latitude and longitude based on speed and direction.
        Uses Haversine formula for movement on a sphere.
        """
        # Convert speed from km/h to degrees per interval
        distance_km_per_interval = (self.speed_kmph / 3600) * self.update_interval

        # Convert distance in km to angular distance in radians on Earth's surface
        angular_distance_rad = distance_km_per_interval * 1000 / Re # Convert km to meters for Re

        # Current position in radians
        lat_rad = math.radians(self.latitude)
        lon_rad = math.radians(self.longitude)

        # Calculate new position using spherical trigonometry (Haversine formula for destination point)
        new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(angular_distance_rad) +
                                math.cos(lat_rad) * math.sin(angular_distance_rad) * math.cos(self.direction_rad))
        
        new_lon_rad = lon_rad + math.atan2(math.sin(self.direction_rad) * math.sin(angular_distance_rad) * math.cos(lat_rad),
                                           math.cos(angular_distance_rad) - math.sin(lat_rad) * math.sin(new_lat_rad))

        self.latitude = math.degrees(new_lat_rad)
        self.longitude = math.degrees(new_lon_rad)
        
        # Ensure longitude stays within -180 to +180
        self.longitude = (self.longitude + 180) % 360 - 180

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

    def _calculate_latency(self):
        """
        Calculates the propagation latency from the aircraft to the connected satellite.
        """
        if self.connected_satellite:
            total_distance_m = self._calculate_3d_distance(self.connected_satellite)
            # Propagation delay
            propagation_delay = total_distance_m / Vc # Vc is speed of light in m/s
            return propagation_delay
        return 0

# Earth consisting of cells
class Earth:
    def __init__(self, env, img_path, constellation, inputParams, deltaT, getRates = False, window=None):
        pop_count_data = Image.open(img_path)
        # total image sizes
        [self.total_x, self.total_y] = pop_count_data.size

        self.total_cells = self.total_x * self.total_y
        self.constellationType = "OneWeb"  # Type of constellation, used for beam initialization

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

        # create constellation of satellites
        self.LEO = create_Constellation(constellation, env)

        # Simpy process for handling moving the constellation and the satellites within the constellation
        self.moveConstellation = env.process(self.moveConstellation(env, deltaT, getRates))

    def set_window(self, window):  # function to change/set window for the earth
        """
        Unused function
        """
        self.lati = [window[2], window[3]]
        self.longi = [window[0], window[1]]
        self.windowx = ((int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
        self.windowy = ((int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))


    def moveConstellation(self, env, deltaT=10, getRates = False):
        """
        Simpy process function:

        Moves the constellations in terms of the Earth's rotation and moves the satellites within the constellations.
        The movement is based on the time that has passed since last constellation movement and is defined by the
        "deltaT" variable.

        After the satellites have been moved a process of re-linking all links, both GSLs and ISLs, is conducted where
        the paths for all blocks are re-made, the blocks are moved (if necessary) to the correct buffers, and all
        processes managing the send-buffers are checked to ensure they will still work correctly.
        """
        while True:
    
            # rotate constellation and satellites
            for constellation in self.LEO:
                constellation.rotate(deltaT)
            yield env.timeout(deltaT)    

    def plotMap(self, plotSat = True, plotBeams = True, plotAircrafts = True, aircrafts=None, selected_beam_id=None, path = None, bottleneck = None):
        print("Plotting map")
        fig = plt.figure(figsize=(12, 6))
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
            ax.scatter(aircraft_lons, aircraft_lats, color='black', marker='x', s=50, transform=ccrs.PlateCarree(), label='Aircraft', zorder=20)
            
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


        # Create a single legend for all plotted elements
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            plt.legend(by_label.values(), by_label.keys(), loc='lower left', prop={'size': 8})

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

    def save_plot_at_intervals(self, env, interval=1, plotSat=True, plotBeams=True, plotAircrafts=False, aircrafts=None):
        img_count = 0
        if not os.path.exists("simulationImages"):
            os.makedirs("simulationImages")
        while True:
            print(f"\nSaving plot {img_count} at simulation time {env.now}")
            # Get the ID of the connected beam for highlighting
            selected_beam_id = aircrafts[0].connected_beam.id if aircrafts and aircrafts[0].connected_beam else None
            self.plotMap(plotSat=plotSat, plotBeams=plotBeams, plotAircrafts=plotAircrafts, aircrafts=aircrafts, selected_beam_id=selected_beam_id)
            plt.savefig(f"simulationImages/sat_positions_{img_count}.png")
            plt.close()
            img_count += 1
            yield env.timeout(interval)        

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


def initialize(env, img_path, inputParams, movementTime):
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

    constellationType = inputParams['Constellation'][0]

    # Load earth and gateways
    earth = Earth(env, img_path,  constellationType, inputParams, movementTime)
    # Initialize an Aircraft instance here
    # Example aircraft: starting at SF, moving North towards Seattle at 800 km/h, 10km altitude
    aircraft = Aircraft(env, "A-380", start_lat=37.77, start_lon=-122.41, height=10000, speed_kmph=50000, direction_deg=345, update_interval=10)

    print("Initialized Earth")
    print(earth)
    print()

    return earth, aircraft 


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
    for i in range(0, P):
        orbital_planes.append(OrbitalPlane(str(i), height, i*distribution_angle/P, math.radians(inclination_angle), N_p,
                                           min_elevation_angle, str(i) + '_', env))

    return orbital_planes


# Global variable to store the earth instance, so Aircraft class can access it
earth_instance = None

def main():
    """
    This function is made to avoid problems with scope. everything in if __name__ = "__main__" is in global scope which
    can be an issue.
    """
    global earth_instance # Declare as global

    # Create a dummy input.csv if it doesn't exist
    if not os.path.exists("input.csv"):
        with open("input.csv", "w") as f:
            f.write("Test length,Constellation\n")
            f.write("100,OneWeb\n")


    inputParams = pd.read_csv("input.csv")

    testLength = inputParams['Test length'][0]
    constellation_name = inputParams['Constellation'][0]

    # Time interval for constellation movement updates
    movementTime = 10
    print(f"Constellation: {constellation_name}")
    print(f"Constellation movement update interval: {movementTime} seconds")

    simulationTimelimit = testLength
    print(f"Simulation test length: {simulationTimelimit} seconds")

    env = simpy.Environment()
    
    # Check for population map, create a dummy one if not found
    img_path = "PopMap_500.png"
    if not os.path.exists(img_path):
        print(f"'{img_path}' not found. Creating a dummy 500x250 black image.")
        Image.new('L', (500, 250)).save(img_path)


    earth_instance, aircraft1 = initialize(env, img_path, inputParams, movementTime)

    # --- Aircraft Simulation ---
    all_aircrafts = [aircraft1] # List of all aircrafts to pass to plotting function

    # Start plotting process to save map images at intervals
    # plotAircrafts is set to True to show aircraft on the map
    env.process(earth_instance.save_plot_at_intervals(env, interval=10, plotSat=True, plotBeams=True, plotAircrafts=True, aircrafts=all_aircrafts))

    progress = env.process(simProgress(simulationTimelimit, env))
    startTime = time.time()
    env.run(simulationTimelimit)
    timeToSim = time.time() - startTime

    print("\n\n--- Simulation Summary ---")
    print(f"Total simulation run time: {timeToSim:.2f} seconds")
    print(f"Aircraft '{aircraft1.id}' total handovers: {aircraft1.handover_count}")
    if aircraft1.connected_beam:
        print(f"Aircraft '{aircraft1.id}' final connected beam: {aircraft1.connected_beam.id}")
        print(f"Aircraft '{aircraft1.id}' final SNR: {aircraft1.current_snr:.2f} dB")
        print(f"Aircraft '{aircraft1.id}' final Latency: {aircraft1.current_latency*1e3:.2f} ms")
    else:
        print(f"Aircraft '{aircraft1.id}' ended the simulation with no connection.")


if __name__ == '__main__':
    main()