import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import re

def parse_time(time_str):
    """Parse time string like 'Tue 10:48:41 AM' to datetime"""
    try:
        # Remove day name and extract time
        time_part = time_str.split(' ', 1)[1]  # Remove 'Tue'
        # Parse time with AM/PM
        dt = datetime.strptime(time_part, '%I:%M:%S %p')
        # Set a reference date (doesn't matter for interpolation)
        dt = dt.replace(year=2024, month=1, day=1)
        return dt
    except:
        return None

def load_route_from_csv(filename, skip_rows=2):
    """
    Loads the aircraft route from a CSV file, skipping the first and last N rows.
    Returns a list of dicts with keys: 'lat', 'lon', 'speed_mph'
    """
    route = []
    # Try utf-8-sig first, fallback to latin1 if error
    print("Loading flight route from csv")
    try:
        with open(filename, newline='', encoding='utf-8-sig') as csvfile:
            reader = list(csv.DictReader(csvfile))
    except UnicodeDecodeError as e:
        print(f"UTF-8 decoding error: {e}. Trying latin1 encoding.")
        with open(filename, newline='', encoding='latin1') as csvfile:
            reader = list(csv.DictReader(csvfile))
    # Ignore first and last skip_rows
    data = reader[skip_rows:-skip_rows]
    print(data)
    for row in data:
        try:
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            speed = float(row['mph'])
            altitude = float(row['feet'].replace(',', ''))  # Ensure altitude is float
            time = pd.to_datetime(row['Time (EDT)'], format="%a %I:%M:%S %p")  # Time in seconds
            route.append({'lat': lat, 'lon': lon, 'speed_mph': speed, 'alt': altitude, 'time': time})
        except Exception as e:
            print(f"Skipping malformed row due to error: {e}")
            continue  # skip malformed rows
    route_duration = (route[-1]['time'] - route[0]['time']).total_seconds()
    return route, route_duration

def clean_numeric_value(value_str):
    """Clean and convert numeric values, handling commas and special characters"""
    if pd.isna(value_str) or value_str == '':
        return np.nan
    
    # Convert to string and clean
    cleaned = str(value_str)
    # Remove commas, quotes, and other unwanted characters including non-breaking spaces
    cleaned = re.sub(r'[,"�\xa0\u00a0\u2009\u200a]', '', cleaned)
    # Remove degree symbols and directional indicators
    cleaned = re.sub(r'[°�?]', '', cleaned)
    # Remove trailing characters that aren't numbers, decimal points, or minus signs
    cleaned = re.sub(r'[^\d.-]+$', '', cleaned)
    # Remove leading non-numeric characters except minus
    cleaned = re.sub(r'^[^\d.-]+', '', cleaned)
    
    # Handle empty strings after cleaning
    if cleaned.strip() == '':
        return np.nan
    
    try:
        return float(cleaned)
    except:
        return np.nan

def interpolate_route_to_10s(csv_file_path, output_file_path='route_10s_interpolated.csv'):
    """
    Read route.csv and interpolate to 10-second cadences using load_route_from_csv
    
    Args:
        csv_file_path: Path to input route.csv file
        output_file_path: Path for output interpolated file
    """
    
    print(f"Reading route data from {csv_file_path}...")
    
    # Use the existing load_route_from_csv function
    try:
        route_data, route_duration = load_route_from_csv(csv_file_path, skip_rows=3)
        print(f"Loaded {len(route_data)} data points from route")
    except Exception as e:
        print(f"Error loading route: {e}")
        return None
    
    if len(route_data) < 2:
        print("Error: Need at least 2 data points for interpolation!")
        return None
    
    # Convert to DataFrame for easier processing
    route_df = pd.DataFrame(route_data)
    
    # Convert times to seconds from start
    start_time = route_df['time'].min()
    route_df['seconds'] = (route_df['time'] - start_time).dt.total_seconds()
    
    print(f"Time range: {start_time} to {route_df['time'].max()}")
    print(f"Duration: {route_duration:.0f} seconds ({route_duration/60:.1f} minutes)")
    
    # Create 10-second time grid
    total_duration = route_df['seconds'].max()
    time_10s = np.arange(0, total_duration + 1, 10)  # 10-second intervals
    
    print(f"Interpolating to {len(time_10s)} 10-second intervals...")
    
    # Interpolate each numeric column
    interpolated_data = {'seconds': time_10s}
    
    # Map the route data columns to standard names
    column_mapping = {
        'lat': 'latitude',
        'lon': 'longitude', 
        'speed_mph': 'speed_mph',
        'alt': 'altitude_ft'
    }
    
    for route_col, output_col in column_mapping.items():
        if route_col in route_df.columns:
            # Remove NaN values for interpolation
            valid_mask = ~route_df[route_col].isna()
            if valid_mask.sum() >= 2:  # Need at least 2 points
                interp_func = interp1d(
                    route_df.loc[valid_mask, 'seconds'], 
                    route_df.loc[valid_mask, route_col], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                interpolated_data[output_col] = interp_func(time_10s)
            else:
                interpolated_data[output_col] = np.full(len(time_10s), np.nan)
    
    # Create interpolated dataframe
    interpolated_df = pd.DataFrame(interpolated_data)
    
    # Add back datetime column
    interpolated_df['datetime'] = start_time + pd.to_timedelta(interpolated_df['seconds'], unit='s')
    
    # Reorder columns
    available_cols = ['seconds', 'datetime']
    for col in ['latitude', 'longitude', 'speed_mph', 'altitude_ft']:
        if col in interpolated_df.columns:
            available_cols.append(col)
    
    interpolated_df = interpolated_df[available_cols]
    
    # Save to CSV
    interpolated_df.to_csv(output_file_path, index=False)
    
    print(f"\nInterpolation complete!")
    print(f"Output saved to: {output_file_path}")
    print(f"Original data points: {len(route_df)}")
    print(f"Interpolated data points: {len(interpolated_df)}")
    print(f"Total duration: {total_duration:.0f} seconds ({total_duration/60:.1f} minutes)")
    
    # Show sample of results
    print(f"\nSample interpolated data:")
    print(interpolated_df.head(10))
    
    return interpolated_df

def main():
    # Run the interpolation
    route_file = 'route.csv'
    output_file = 'route_10s_interpolated.csv'
    
    try:
        interpolated_data = interpolate_route_to_10s(route_file, output_file)
        
        # Basic validation
        if interpolated_data is not None:
            print(f"\nValidation:")
            print(f"Latitude range: {interpolated_data['latitude'].min():.4f} to {interpolated_data['latitude'].max():.4f}")
            print(f"Longitude range: {interpolated_data['longitude'].min():.4f} to {interpolated_data['longitude'].max():.4f}")
            print(f"Altitude range: {interpolated_data['altitude_ft'].min():.0f} to {interpolated_data['altitude_ft'].max():.0f} ft")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()