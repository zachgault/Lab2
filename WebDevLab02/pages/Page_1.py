import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
import requests
import json

# Add these imports if not already present
from mpl_toolkits.mplot3d import Axes3D

def fetch_tle_data():
    """Fetch TLE data from Celestrak API"""
    try:
        # Use a more general API endpoint that returns more satellites
        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        st.success(f"Fetched {len(data)} satellites from Celestrak")
        return data
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching TLE data: {e}")
        # Return sample data for testing
        return [{
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "EPOCH": "2025-07-19T14:43:28.168608",
            "MEAN_MOTION": 15.49309239,
            "ECCENTRICITY": 0.0001693,
            "INCLINATION": 51.6461,
            "RA_OF_ASC_NODE": 339.7939,
            "ARG_OF_PERICENTER": 92.8340,
            "MEAN_ANOMALY": 267.3124,
            "EPHEMERIS_TYPE": 0,
            "CLASSIFICATION_TYPE": "U",
            "NORAD_CAT_ID": 25544,
            "ELEMENT_SET_NO": 999,
            "REV_AT_EPOCH": 28996,
            "BSTAR": 0.000040768,
            "MEAN_MOTION_DOT": 0.00002182,
            "MEAN_MOTION_DDOT": 0
        }]

def orbital_elements_to_tle(sat_data):
    """Convert orbital elements from Celestrak JSON to TLE format"""
    try:
        # Extract data
        norad_id = str(sat_data.get('NORAD_CAT_ID', 99999))
        object_id = sat_data.get('OBJECT_ID', '00000A')
        
        # Parse epoch
        epoch_str = sat_data.get('EPOCH', '')
        if epoch_str:
            epoch_dt = datetime.fromisoformat(epoch_str.replace('Z', '+00:00'))
            # Convert to TLE epoch format (YYDDD.FFFFFFFF)
            year = epoch_dt.year % 100
            day_of_year = epoch_dt.timetuple().tm_yday
            fraction_of_day = (epoch_dt.hour * 3600 + epoch_dt.minute * 60 + epoch_dt.second + epoch_dt.microsecond/1e6) / 86400
            epoch_tle = f"{year:02d}{day_of_year:03d}.{fraction_of_day:08f}"
        else:
            epoch_tle = "25200.50000000"  # Default
        
        # Get orbital elements
        mean_motion = sat_data.get('MEAN_MOTION', 15.0)
        eccentricity = sat_data.get('ECCENTRICITY', 0.001)
        inclination = sat_data.get('INCLINATION', 51.6)
        raan = sat_data.get('RA_OF_ASC_NODE', 0.0)
        arg_perigee = sat_data.get('ARG_OF_PERICENTER', 0.0)
        mean_anomaly = sat_data.get('MEAN_ANOMALY', 0.0)
        
        # Get perturbation terms
        bstar = sat_data.get('BSTAR', 0.0)
        mean_motion_dot = sat_data.get('MEAN_MOTION_DOT', 0.0)
        mean_motion_ddot = sat_data.get('MEAN_MOTION_DDOT', 0.0)
        element_set_no = sat_data.get('ELEMENT_SET_NO', 999)
        
        # Format TLE Line 1
        # Format: 1 NNNNNc NNNNNaaa YYDDD.DDDDDDDD  .DDDDDDDD  DDDDD-D  DDDDD-D D NNNNN
        classification = sat_data.get('CLASSIFICATION_TYPE', 'U')
        
        # Format scientific notation for TLE
        def format_exponential(value, width=8):
            if value == 0:
                return ' 00000-0'
            
            # Convert to scientific notation
            exp_str = f"{value:.5e}"
            mantissa, exp = exp_str.split('e')
            mantissa = float(mantissa)
            exp = int(exp)
            
            # Format for TLE (no decimal point, sign for exponent)
            if mantissa < 0:
                sign = '-'
                mantissa = abs(mantissa)
            else:
                sign = ' '
            
            # Remove decimal point and format
            mantissa_str = f"{mantissa:.5f}".replace('.', '')[:5]
            exp_sign = '+' if exp >= 0 else '-'
            exp_str = f"{abs(exp)}"
            
            return f"{sign}{mantissa_str}{exp_sign}{exp_str}"
        
        mean_motion_dot_str = format_exponential(mean_motion_dot)
        mean_motion_ddot_str = format_exponential(mean_motion_ddot)
        bstar_str = format_exponential(bstar)
        
        line1 = f"1 {norad_id:>5}{'U'} {object_id:>8} {epoch_tle} {mean_motion_dot_str} {mean_motion_ddot_str} {bstar_str} 0 {element_set_no:>4}"
        
        # Calculate checksum for line 1
        checksum1 = calculate_tle_checksum(line1)
        line1 += str(checksum1)
        
        # Format TLE Line 2
        # Format: 2 NNNNN DDD.DDDD DDD.DDDD DDDDDDD DDD.DDDD DDD.DDDD DD.DDDDDDDDNNNNN
        ecc_str = f"{eccentricity:.7f}".split('.')[1]  # Remove leading "0."
        
        line2 = f"2 {norad_id:>5} {inclination:8.4f} {raan:8.4f} {ecc_str:>7} {arg_perigee:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f}{sat_data.get('REV_AT_EPOCH', 0):>5}"
        
        # Calculate checksum for line 2
        checksum2 = calculate_tle_checksum(line2)
        line2 += str(checksum2)
        
        return line1, line2
        
    except Exception as e:
        st.error(f"Error converting orbital elements to TLE: {e}")
        return None, None

def calculate_tle_checksum(line):
    """Calculate TLE checksum"""
    checksum = 0
    for char in line:
        if char.isdigit():
            checksum += int(char)
        elif char == '-':
            checksum += 1
    return checksum % 10

def filter_satellites(tle_data, object_name, eccentricity, inclination, ra_of_asc_node, arg_of_pericenter):
    """Filter satellites based on user criteria"""
    filtered = []
    
    for sat in tle_data:
        # Object name filter
        if object_name and object_name.strip() != '0' and object_name.strip() != '':
            if object_name.upper() not in sat.get('OBJECT_NAME', '').upper():
                continue
        
        # Eccentricity filter
        if eccentricity and eccentricity.strip() != '0' and eccentricity.strip() != '':
            try:
                ecc_value = float(eccentricity)
                sat_ecc = float(sat.get('ECCENTRICITY', 0))
                if abs(sat_ecc - ecc_value) > 0.01:  # Tolerance
                    continue
            except ValueError:
                continue
        
        # Inclination filter
        if inclination and inclination.strip() != '0' and inclination.strip() != '':
            try:
                inc_value = float(inclination)
                sat_inc = float(sat.get('INCLINATION', 0))
                if abs(sat_inc - inc_value) > 5.0:  # 5 degree tolerance
                    continue
            except ValueError:
                continue
                
        # RAAN filter
        if ra_of_asc_node and ra_of_asc_node.strip() != '0' and ra_of_asc_node.strip() != '':
            try:
                raan_value = float(ra_of_asc_node)
                sat_raan = float(sat.get('RA_OF_ASC_NODE', 0))
                if abs(sat_raan - raan_value) > 10.0:  # 10 degree tolerance
                    continue
            except ValueError:
                continue
                
        # Argument of perigee filter
        if arg_of_pericenter and arg_of_pericenter.strip() != '0' and arg_of_pericenter.strip() != '':
            try:
                arg_value = float(arg_of_pericenter)
                sat_arg = float(sat.get('ARG_OF_PERICENTER', 0))
                if abs(sat_arg - arg_value) > 10.0:  # 10 degree tolerance
                    continue
            except ValueError:
                continue
        
        filtered.append(sat)
    
    return filtered

def create_sgp4_satellite(sat_data):
    """Create SGP4 satellite object from orbital elements data"""
    try:
        # Convert orbital elements to TLE format
        tle_line1, tle_line2 = orbital_elements_to_tle(sat_data)
        
        if not tle_line1 or not tle_line2:
            st.warning(f"Could not create TLE for {sat_data.get('OBJECT_NAME', 'Unknown')}")
            return None
        
        # Debug: Show TLE lines
        with st.expander(f"TLE for {sat_data.get('OBJECT_NAME', 'Unknown')}"):
            st.code(f"Line 1: {tle_line1}\nLine 2: {tle_line2}")
        
        # Create satellite object using SGP4
        satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        
        # Check if satellite was created successfully
        if satellite.error != 0:
            st.warning(f"SGP4 error {satellite.error} for {sat_data.get('OBJECT_NAME', 'Unknown')}")
            return None
            
        return satellite
    except Exception as e:
        st.error(f"Error creating SGP4 satellite for {sat_data.get('OBJECT_NAME', 'Unknown')}: {e}")
        return None

def propagate_sgp4_orbit(satellite, hours=24, steps=100):
    """Propagate satellite orbit using SGP4 for specified time period"""
    try:
        # Time array
        start_time = datetime.utcnow()
        time_points = [start_time + timedelta(hours=i * hours / steps) for i in range(steps)]
        
        positions = []
        velocities = []
        error_count = 0
        
        for time_point in time_points:
            # Convert to Julian date
            jd, fr = jday(time_point.year, time_point.month, time_point.day, 
                         time_point.hour, time_point.minute, time_point.second)
            
            # Propagate satellite
            error, r, v = satellite.sgp4(jd, fr)
            
            if error == 0:  # No error
                positions.append(r)
                velocities.append(v)
            else:
                error_count += 1
                positions.append([np.nan, np.nan, np.nan])
                velocities.append([np.nan, np.nan, np.nan])
        
        if error_count > 0:
            st.warning(f"SGP4 propagation errors occurred in {error_count}/{steps} time steps")
        
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        # Check if we have valid positions
        valid_positions = ~np.isnan(positions[:, 0])
        
        if np.sum(valid_positions) == 0:
            st.error("No valid positions generated!")
            return None, None, None
        
        return positions, velocities, time_points
    except Exception as e:
        st.error(f"Error propagating orbit: {e}")
        return None, None, None

def plot_sgp4_orbits_3d(satellites_data, orbit_hours=24):
    """Create 3D plot of SGP4 satellite orbits"""
    try:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(satellites_data), 10)))
        
        # Plot Earth
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        earth_radius = 6371  # km
        earth_x = earth_radius * np.outer(np.cos(u), np.sin(v))
        earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
        earth_z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(earth_x, earth_y, earth_z, alpha=0.3, color='lightblue', edgecolor='none')
        
        orbit_plotted = False
        
        for i, sat_data in enumerate(satellites_data):
            satellite = create_sgp4_satellite(sat_data)
            if satellite is None:
                continue
            
            positions, velocities, times = propagate_sgp4_orbit(satellite, hours=orbit_hours, steps=150)
            if positions is None:
                continue
            
            # Filter out NaN values
            valid_indices = ~np.isnan(positions[:, 0])
            if not np.any(valid_indices):
                st.warning(f"No valid positions for {sat_data.get('OBJECT_NAME', 'Unknown')}")
                continue
            
            pos_valid = positions[valid_indices]
            
            object_name = sat_data.get('OBJECT_NAME', 'Unknown')
            color_idx = i % len(colors)
            
            # Plot orbit
            ax.plot(pos_valid[:, 0], pos_valid[:, 1], pos_valid[:, 2], 
                   label=object_name, color=colors[color_idx], linewidth=2)
            
            # Mark starting position
            if len(pos_valid) > 0:
                ax.scatter(pos_valid[0, 0], pos_valid[0, 1], pos_valid[0, 2], 
                          color=colors[color_idx], s=100, marker='o', alpha=1.0, edgecolors='black')
            
            orbit_plotted = True
        
        if not orbit_plotted:
            st.error("No orbits could be plotted. Check your satellite data.")
            return None
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(f'SGP4 Satellite Orbits - {orbit_hours}h Propagation\n(Earth-Centered Inertial Frame)')
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # Set reasonable axis limits
        max_range = 25000  # km
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Improve 3D visualization
        ax.view_init(elev=20, azim=45)
        
        return fig
    except Exception as e:
        st.error(f"Error creating 3D plot: {e}")
        return None

def plot_sgp4_ground_track(satellites_data, orbit_hours=24):
    """Plot ground track of satellites"""
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(satellites_data), 10)))
        
        track_plotted = False
        
        for i, sat_data in enumerate(satellites_data):
            satellite = create_sgp4_satellite(sat_data)
            if satellite is None:
                continue
            
            positions, velocities, times = propagate_sgp4_orbit(satellite, hours=orbit_hours, steps=200)
            if positions is None:
                continue
            
            # Convert ECEF to lat/lon
            valid_indices = ~np.isnan(positions[:, 0])
            if not np.any(valid_indices):
                continue
            
            pos_valid = positions[valid_indices]
            
            # Convert to lat/lon
            lats = []
            lons = []
            for pos in pos_valid:
                x, y, z = pos
                r = np.sqrt(x**2 + y**2 + z**2)
                lat = np.arcsin(z / r) * 180 / np.pi
                lon = np.arctan2(y, x) * 180 / np.pi
                lats.append(lat)
                lons.append(lon)
            
            if not lats:
                continue
            
            # Handle longitude wraparound for continuous plotting
            lons_plot = []
            lats_plot = []
            for j in range(len(lons)):
                if j > 0 and abs(lons[j] - lons[j-1]) > 180:
                    # Add break in line for longitude wraparound
                    lons_plot.append(np.nan)
                    lats_plot.append(np.nan)
                lons_plot.append(lons[j])
                lats_plot.append(lats[j])
            
            object_name = sat_data.get('OBJECT_NAME', 'Unknown')
            color_idx = i % len(colors)
            ax.plot(lons_plot, lats_plot, label=object_name, color=colors[color_idx], linewidth=2, alpha=0.8)
            
            # Mark starting position
            if lons:
                ax.scatter(lons[0], lats[0], color=colors[color_idx], s=100, marker='o', 
                          alpha=1.0, edgecolors='black')
            
            track_plotted = True
        
        if not track_plotted:
            st.error("No ground tracks could be plotted.")
            return None
        
        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        ax.set_title(f'SGP4 Satellite Ground Tracks - {orbit_hours}h Propagation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set world map limits
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        
        # Add gridlines for better reference
        ax.set_xticks(np.arange(-180, 181, 30))
        ax.set_yticks(np.arange(-90, 91, 30))
        
        return fig
    except Exception as e:
        st.error(f"Error creating ground track plot: {e}")
        return None

def page_sgp4_orbits():
    """Page 2: SGP4 orbit modeling"""
    st.header("🛰️ SGP4 Orbit Modeling (High Precision)")
    st.write("This page uses the SGP4 algorithm with real Celestrak orbital elements data.")
    st.write("Data is fetched from: https://celestrak.org/NORAD/elements/")
    st.write("---")
    
    # User input fields
    col1, col2 = st.columns(2)
    
    with col1:
        object_name = st.text_input("Enter Satellite name you want to search for", 'ISS', 
                                   help="Partial name search (e.g., 'ISS', 'STARLINK')", key="sgp4_name")
        eccentricity = st.text_input("Enter Eccentricity you want to search for", '0',
                                   help="Orbital eccentricity (0-1, e.g., '0.001')", key="sgp4_ecc")
        inclination = st.text_input("Enter Inclination you want to search for", '0',
                                  help="Orbital inclination in degrees (e.g., '51.6')", key="sgp4_inc")
    
    with col2:
        ra_of_asc_node = st.text_input("Enter RA_OF_ASC_NODE you want to search for", '0',
                                     help="Right ascension of ascending node in degrees", key="sgp4_ra")
        arg_of_pericenter = st.text_input("Enter ARG_OF_PERICENTER you want to search for", '0',
                                        help="Argument of pericenter in degrees", key="sgp4_arg")
    
    # SGP4 specific settings
    st.subheader("SGP4 Simulation Settings")
    col1, col2 = st.columns(2)
    with col1:
        orbit_hours = st.slider("Propagation Time (hours)", 1, 168, 24, help="How long to propagate the orbit")
    with col2:
        plot_type = st.selectbox("Plot Type", ["3D Orbit", "Ground Track", "Both"], help="Choose visualization type")
    
    # Search button
    if st.button("Run SGP4 Simulation", type="primary", key="sgp4_search"):
        with st.spinner("Fetching satellite data from Celestrak..."):
            tle_data = fetch_tle_data()
        
        if not tle_data:
            st.error("No satellite data available. Please try again later.")
            return
        
        # Filter satellites based on user criteria
        filtered_satellites = filter_satellites(
            tle_data, object_name, eccentricity, inclination, 
            ra_of_asc_node, arg_of_pericenter
        )
        
        if not filtered_satellites:
            st.warning("No satellites found matching your criteria. Try adjusting your search parameters.")
            return
        
        st.success(f"Found {len(filtered_satellites)} satellite(s) matching your criteria:")
        
        # Limit satellites for performance
        max_satellites = min(3, len(filtered_satellites))
        simulation_satellites = filtered_satellites[:max_satellites]
        
        if len(filtered_satellites) > max_satellites:
            st.info(f"Showing first {max_satellites} satellites for optimal performance.")
        
        # Display satellite information
        satellite_info = []
        for sat in simulation_satellites:
            satellite_info.append({
                'Name': sat.get('OBJECT_NAME', 'Unknown'),
                'NORAD ID': sat.get('NORAD_CAT_ID', 'Unknown'),
                'Epoch': sat.get('EPOCH', 'Unknown'),
                'Eccentricity': f"{float(sat.get('ECCENTRICITY', 0)):.6f}",
                'Inclination': f"{float(sat.get('INCLINATION', 0)):.2f}°",
                'Mean Motion': f"{float(sat.get('MEAN_MOTION', 0)):.8f} rev/day",
                'Altitude (approx)': f"{(((86400/float(sat.get('MEAN_MOTION', 15))/(2*np.pi))**(2/3) * (398600.4418)**(1/3)) - 6371):.1f} km"
            })
        
        st.dataframe(pd.DataFrame(satellite_info))
        
        # Run SGP4 simulation and plotting
        st.subheader("SGP4 Simulation Results")
        
        with st.spinner(f"Running SGP4 propagation for {orbit_hours} hours..."):
            if plot_type in ["3D Orbit", "Both"]:
                st.subheader("3D Orbital Trajectories")
                fig_3d = plot_sgp4_orbits_3d(simulation_satellites, orbit_hours)
                if fig_3d:
                    st.pyplot(fig_3d)
                    plt.close(fig_3d)
            
            if plot_type in ["Ground Track", "Both"]:
                st.subheader("Ground Track")
                fig_ground = plot_sgp4_ground_track(simulation_satellites, orbit_hours)
                if fig_ground:
                    st.pyplot(fig_ground)
                    plt.close(fig_ground)
        
        # SGP4 specific information
        st.subheader("SGP4 Algorithm Information")
        st.info("""
        **SGP4 Features:**
        - Uses real orbital elements from Celestrak/NORAD
        - Accounts for atmospheric drag effects
        - Includes gravitational perturbations from Earth's non-spherical shape  
        - Considers secular and periodic variations
        - Provides high accuracy for short-term predictions
        """)

def main():
    st.set_page_config(page_title="SGP4 Satellite Tracker", layout="wide")
    st.title("🛰️ SGP4 Satellite Orbit Tracker")
    
    page_sgp4_orbits()

if __name__ == "__main__":
    main()
