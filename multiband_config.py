"""
Multiband Configuration for LEO Satellite Constellations

This configuration file defines the frequency bands used by different
satellite constellations for multiband handover scenarios.

Author: Updated for multiband support
Date: 2026
"""

# =============================================================================
# FREQUENCY BAND CONFIGURATIONS
# =============================================================================

FREQUENCY_BANDS = {
    'OneWeb': {
        'band1': {
            'name': 'Ku-band Downlink',
            'fc': 12.5e9,      # Hz - Center frequency (10.7-14.5 GHz range)
            'fc_label': '12.5 GHz',
            'bw': 250e6,       # Hz - Bandwidth (250 MHz)
            'Pt': 40,          # dBm - Transmit power
            'Gt': 30,          # dBi - Antenna gain
            'frequency_range': '10.7-14.5 GHz',
            'typical_use': 'Primary downlink for user terminals'
        },
        'band2': {
            'name': 'Ka-band Downlink',
            'fc': 20.0e9,      # Hz - Center frequency (17.7-20.2 GHz range)
            'fc_label': '20.0 GHz',
            'bw': 500e6,       # Hz - Bandwidth (500 MHz)
            'Pt': 38,          # dBm - Transmit power
            'Gt': 32,          # dBi - Antenna gain
            'frequency_range': '17.7-20.2 GHz',
            'typical_use': 'High-capacity downlink, gateway links'
        },
        'constellation_info': {
            'altitude': 1200,  # km
            'inclination': 86.4,  # degrees
            'num_satellites': 648,
            'num_beams_per_satellite': 16,
            'coverage_area': '1,718,000 km²'
        }
    },
    
    'Starlink': {
        'band1': {
            'name': 'Ku-band Downlink',
            'fc': 12.0e9,      # Hz - Center frequency (10.7-12.7 GHz)
            'fc_label': '12.0 GHz',
            'bw': 240e6,       # Hz - Bandwidth (240 MHz)
            'Pt': 35,          # dBm - Transmit power
            'Gt': 28,          # dBi - Antenna gain
            'frequency_range': '10.7-12.7 GHz',
            'typical_use': 'Primary user downlink'
        },
        'band2': {
            'name': 'Ka-band Downlink',
            'fc': 18.5e9,      # Hz - Center frequency (17.8-19.3 GHz)
            'fc_label': '18.5 GHz',
            'bw': 500e6,       # Hz - Bandwidth (500 MHz)
            'Pt': 40,          # dBm - Transmit power
            'Gt': 35,          # dBi - Antenna gain
            'frequency_range': '17.8-19.3 GHz',
            'typical_use': 'High-throughput downlink, gateway'
        },
        'constellation_info': {
            'altitude': 550,   # km (Phase 1 shell)
            'inclination': 53,  # degrees
            'num_satellites': 1584,  # Phase 1
            'num_beams_per_satellite': 16,  # Estimated
            'coverage_area': 'Variable based on beam steering'
        }
    }
}

# =============================================================================
# MULTIBAND STATE SPACE CONFIGURATION
# =============================================================================

OBSERVATION_SPACE_CONFIG = {
    'dimension': 18,
    'features': [
        # Position and basic metrics (indices 0-2)
        {'index': 0, 'name': 'latitude', 'unit': 'degrees', 'range': [-90, 90]},
        {'index': 1, 'name': 'longitude', 'unit': 'degrees', 'range': [-180, 180]},
        {'index': 2, 'name': 'altitude', 'unit': 'meters', 'range': [0, 15000]},
        
        # Band 1 (Ku-band) metrics (indices 3-5)
        {'index': 3, 'name': 'snr_band1', 'unit': 'dB', 'range': [-100, 100], 'band': 'Ku'},
        {'index': 4, 'name': 'load_band1', 'unit': 'ratio', 'range': [0, 1], 'band': 'Ku'},
        {'index': 5, 'name': 'handovers', 'unit': 'count', 'range': [0, 1000]},
        
        # Connection quality metrics (indices 6-8)
        {'index': 6, 'name': 'allocated_bandwidth', 'unit': 'MB', 'range': [0, 1000]},
        {'index': 7, 'name': 'allocation_ratio', 'unit': 'ratio', 'range': [0, 1]},
        {'index': 8, 'name': 'demand', 'unit': 'MB', 'range': [0, 1000]},
        
        # Performance metrics (indices 9-11)
        {'index': 9, 'name': 'throughput_required', 'unit': 'Mbps', 'range': [0, 200]},
        {'index': 10, 'name': 'queuing_delay', 'unit': 'seconds', 'range': [0, 10]},
        {'index': 11, 'name': 'propagation_latency', 'unit': 'seconds', 'range': [0, 1]},
        
        # Additional metrics (indices 12-14)
        {'index': 12, 'name': 'transmission_rate', 'unit': 'Mbps', 'range': [0, 200]},
        {'index': 13, 'name': 'latency_required', 'unit': 'seconds', 'range': [0, 1]},
        {'index': 14, 'name': 'beam_capacity_band1', 'unit': 'MB', 'range': [0, 1000], 'band': 'Ku'},
        
        # Band 2 (Ka-band) metrics (indices 15-17) - NEW FOR MULTIBAND
        {'index': 15, 'name': 'snr_band2', 'unit': 'dB', 'range': [-100, 100], 'band': 'Ka'},
        {'index': 16, 'name': 'load_band2', 'unit': 'ratio', 'range': [0, 1], 'band': 'Ka'},
        {'index': 17, 'name': 'beam_capacity_band2', 'unit': 'MB', 'range': [0, 1000], 'band': 'Ka'},
    ]
}

# =============================================================================
# MODEL CONFIGURATION FOR MULTIBAND
# =============================================================================

MODEL_CONFIGS = {
    'ODT': {
        'name': 'Online Decision Transformer',
        'state_dim': 18,  # Updated from 15 to 18
        'embed_dim': 128,  # Increased from 64
        'num_layers': 3,   # Increased from 2
        'num_heads': 4,    # Added multi-head attention
        'max_length': 20,
        'target_return': 1.0,
        'supports_multiband': True,
        'band_selection_head': True,  # ODT has explicit band selection
        'description': 'Transformer-based model with multiband awareness'
    },
    'PPO': {
        'name': 'Proximal Policy Optimization',
        'state_dim': 18,
        'supports_multiband': True,
        'band_selection_head': False,
        'description': 'Policy gradient method with multiband state'
    },
    'DQN': {
        'name': 'Deep Q-Network',
        'state_dim': 18,
        'buffer_size': 50,
        'supports_multiband': True,
        'band_selection_head': False,
        'description': 'Value-based RL with multiband state'
    },
    'BASELINE': {
        'name': 'Baseline Greedy',
        'state_dim': 18,
        'supports_multiband': True,
        'band_selection_head': False,
        'description': 'Greedy beam selection with multiband tracking'
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_band_info(constellation, band_name):
    """
    Get frequency band information for a constellation
    
    Args:
        constellation: str - 'OneWeb' or 'Starlink'
        band_name: str - 'band1' or 'band2'
    
    Returns:
        dict - Band configuration
    """
    if constellation not in FREQUENCY_BANDS:
        raise ValueError(f"Unknown constellation: {constellation}")
    if band_name not in FREQUENCY_BANDS[constellation]:
        raise ValueError(f"Unknown band: {band_name}")
    
    return FREQUENCY_BANDS[constellation][band_name]


def print_multiband_summary(constellation):
    """
    Print a summary of multiband configuration for a constellation
    
    Args:
        constellation: str - 'OneWeb' or 'Starlink'
    """
    if constellation not in FREQUENCY_BANDS:
        print(f"Unknown constellation: {constellation}")
        return
    
    config = FREQUENCY_BANDS[constellation]
    
    print(f"\n{'='*70}")
    print(f"MULTIBAND CONFIGURATION: {constellation}")
    print(f"{'='*70}")
    
    # Constellation info
    info = config['constellation_info']
    print(f"\nConstellation Parameters:")
    print(f"  Altitude: {info['altitude']} km")
    print(f"  Inclination: {info['inclination']}°")
    print(f"  Number of Satellites: {info['num_satellites']}")
    print(f"  Beams per Satellite: {info['num_beams_per_satellite']}")
    print(f"  Coverage Area: {info['coverage_area']}")
    
    # Band 1 info
    print(f"\nBand 1 (Primary):")
    band1 = config['band1']
    print(f"  Name: {band1['name']}")
    print(f"  Center Frequency: {band1['fc_label']} ({band1['frequency_range']})")
    print(f"  Bandwidth: {band1['bw']/1e6:.0f} MHz")
    print(f"  Transmit Power: {band1['Pt']} dBm")
    print(f"  Antenna Gain: {band1['Gt']} dBi")
    print(f"  Use Case: {band1['typical_use']}")
    
    # Band 2 info
    print(f"\nBand 2 (Secondary/High-Capacity):")
    band2 = config['band2']
    print(f"  Name: {band2['name']}")
    print(f"  Center Frequency: {band2['fc_label']} ({band2['frequency_range']})")
    print(f"  Bandwidth: {band2['bw']/1e6:.0f} MHz")
    print(f"  Transmit Power: {band2['Pt']} dBm")
    print(f"  Antenna Gain: {band2['Gt']} dBi")
    print(f"  Use Case: {band2['typical_use']}")
    
    print(f"\n{'='*70}")


def print_observation_space_info():
    """Print information about the multiband observation space"""
    print(f"\n{'='*70}")
    print(f"MULTIBAND OBSERVATION SPACE")
    print(f"{'='*70}")
    print(f"Total Dimensions: {OBSERVATION_SPACE_CONFIG['dimension']}")
    print(f"\nFeature Breakdown:")
    
    # Group by category
    position_features = [f for f in OBSERVATION_SPACE_CONFIG['features'] if f['index'] < 3]
    band1_features = [f for f in OBSERVATION_SPACE_CONFIG['features'] 
                      if 'band' in f and f.get('band') == 'Ku']
    band2_features = [f for f in OBSERVATION_SPACE_CONFIG['features'] 
                      if 'band' in f and f.get('band') == 'Ka']
    other_features = [f for f in OBSERVATION_SPACE_CONFIG['features'] 
                      if 'band' not in f and f['index'] >= 3]
    
    print("\n  Position Features (3):")
    for f in position_features:
        print(f"    [{f['index']}] {f['name']}: {f['range']} {f['unit']}")
    
    print("\n  Band 1 (Ku-band) Features (3):")
    for f in band1_features:
        print(f"    [{f['index']}] {f['name']}: {f['range']} {f['unit']}")
    
    print("\n  Connection Quality Features (9):")
    for f in other_features:
        print(f"    [{f['index']}] {f['name']}: {f['range']} {f['unit']}")
    
    print("\n  Band 2 (Ka-band) Features (3) - NEW:")
    for f in band2_features:
        print(f"    [{f['index']}] {f['name']}: {f['range']} {f['unit']}")
    
    print(f"\n{'='*70}")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Print configuration for OneWeb
    print_multiband_summary('OneWeb')
    
    # Print configuration for Starlink
    print_multiband_summary('Starlink')
    
    # Print observation space info
    print_observation_space_info()
    
    # Test getting band info
    print("\n\nExample: Getting Band 2 info for Starlink:")
    band_info = get_band_info('Starlink', 'band2')
    print(f"Center Frequency: {band_info['fc_label']}")
    print(f"Bandwidth: {band_info['bw']/1e6:.0f} MHz")
