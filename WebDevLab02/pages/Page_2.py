import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import math as m
import json
import google.generativeai as genai

def intro():
    st.header("Satellite Orbit Modeling")
    st.write("This application allows you to search for satellites and visualize their orbital trajectories using TLE (Two-Line Element) data from CelesTrak.")
    st.write("Enter search criteria in the fields below to find and plot satellite orbits.")
    st.write("---")
intro()
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_tle_data():
    """Fetch TLE data from CelesTrak API"""
    try:
        # Using the general catalog endpoint for more satellites
        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching TLE data: {e}")
        return []

def filter_satellites(tle_data, object_name, eccentricity, inclination, ra_of_asc_node, arg_of_pericenter):
    """Filter satellites based on user input criteria. If no criteria provided, returns all satellites."""
    filtered_satellites = []
    
    # Check if any search criteria are provided (not '0' or empty)
    has_criteria = any([
        object_name and object_name.strip() != '0' and object_name.strip() != '',
        eccentricity and eccentricity.strip() != '0' and eccentricity.strip() != '',
        inclination and inclination.strip() != '0' and inclination.strip() != '',
        ra_of_asc_node and ra_of_asc_node.strip() != '0' and ra_of_asc_node.strip() != '',
        arg_of_pericenter and arg_of_pericenter.strip() != '0' and arg_of_pericenter.strip() != ''
    ])
    
    # If no criteria provided, return all satellites
    if not has_criteria:
        return tle_data
    
    for satellite in tle_data:
        # Check if satellite matches search criteria
        matches = True
        
        # Object name filter (case insensitive partial match)
        if object_name and object_name.strip() != '0' and object_name.strip() != '':
            if object_name.upper() not in satellite.get('OBJECT_NAME', '').upper():
                matches = False
        
        # Eccentricity filter (approximate match within 0.01)
        if eccentricity and eccentricity.strip() != '0' and eccentricity.strip() != '':
            try:
                target_ecc = float(eccentricity)
                sat_ecc = float(satellite.get('ECCENTRICITY', 0))
                if abs(sat_ecc - target_ecc) > 0.01:
                    matches = False
            except ValueError:
                matches = False
        
        # Inclination filter (approximate match within 5 degrees)
        if inclination and inclination.strip() != '0' and inclination.strip() != '':
            try:
                target_inc = float(inclination)
                sat_inc = float(satellite.get('INCLINATION', 0))
                if abs(sat_inc - target_inc) > 5:
                    matches = False
            except ValueError:
                matches = False
        
        # RA of ascending node filter (approximate match within 10 degrees)
        if ra_of_asc_node and ra_of_asc_node.strip() != '0' and ra_of_asc_node.strip() != '':
            try:
                target_ra = float(ra_of_asc_node)
                sat_ra = float(satellite.get('RA_OF_ASC_NODE', 0))
                if abs(sat_ra - target_ra) > 10:
                    matches = False
            except ValueError:
                matches = False
        
        # Argument of pericenter filter (approximate match within 10 degrees)
        if arg_of_pericenter and arg_of_pericenter.strip() != '0' and arg_of_pericenter.strip() != '':
            try:
                target_arg = float(arg_of_pericenter)
                sat_arg = float(satellite.get('ARG_OF_PERICENTER', 0))
                if abs(sat_arg - target_arg) > 10:
                    matches = False
            except ValueError:
                matches = False
        
        if matches:
            filtered_satellites.append(satellite)
    
    return filtered_satellites

def calculate_orbit_polar(eccentricity, mean_motion):
    """Calculate polar orbit coordinates from TLE parameters"""
    try:
        # Convert mean motion (revolutions per day) to period in seconds
        T = 86400 / mean_motion  # Period in seconds
        
        # Calculate semi-major axis using Kepler's third law
        # For Earth: GM = 3.986004418 × 10^14 m^3/s^2
        GM = 3.986004418e14
        a = (GM * T**2 / (4 * m.pi**2))**(1/3)
        
        # Calculate semi-latus rectum
        p = a * (1 - eccentricity**2)
        
        # Generate theta values
        theta = np.linspace(0, 2 * m.pi, 1000)
        
        # Calculate radius using polar equation of ellipse
        r = p / (1 + eccentricity * np.cos(theta))
        
        # Convert from meters to kilometers for better visualization
        r_km = r / 1000
        
        return theta, r_km
    except Exception as e:
        st.error(f"Error calculating orbit: {e}")
        return None, None

def plot_orbits(filtered_satellites):
    """Create polar plot of satellite orbits"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_satellites)))
    
    for i, satellite in enumerate(filtered_satellites):
        try:
            eccentricity = float(satellite.get('ECCENTRICITY', 0))
            mean_motion = float(satellite.get('MEAN_MOTION', 1))
            object_name = satellite.get('OBJECT_NAME', 'Unknown')
            
            theta, r_km = calculate_orbit_polar(eccentricity, mean_motion)
            
            if theta is not None and r_km is not None:
                ax.plot(theta, r_km, label=object_name, color=colors[i], linewidth=2)
        except Exception as e:
            st.warning(f"Could not plot orbit for {satellite.get('OBJECT_NAME', 'Unknown')}: {e}")
    
    # Add Earth for reference (radius ≈ 6371 km)
    earth_theta = np.linspace(0, 2 * m.pi, 100)
    earth_r = np.full_like(earth_theta, 6371)
    ax.fill(earth_theta, earth_r, color='blue', alpha=0.7, label='Earth')
    
    ax.set_ylim(0, None)
    ax.set_title('Satellite Orbits (Polar View)', pad=20, fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # Add radial labels
    ax.set_ylabel('Distance (km)', labelpad=30)
    
    return fig

key = st.secrets["key"]

genai.configure(api_key = key)
model = genai.GenerativeModel("gemini-1.5-flash")

def AI_insite_to_plots(figures):
    response = model.generate_content(f"Given the figure {figures} of the satellites orbitting give me some insite on the functional use of having satellites orbit at different eccentricity values.")
    print(response.text)

def search_info_about_sats(object_i):
    for sat_i in object_i:
        sat_n = sat_i['Name']
        response = model.generate_content(f"Given the name of the satellite is {sat_n} provide the user with some information of the function and history of this satellite. If the function or history of the satellite can explain the shape of the orbit given {orbit_i} please provide that to the user too.")
        print(response.text)
        
    
def main():
    
    # User input fields
    col1, col2 = st.columns(2)
    
    with col1:
        object_name = st.text_input("Enter Satellite name you want to search for", '0', 
                                   help="Partial name search (e.g., 'ISS', 'STARLINK')")
        eccentricity = st.text_input("Enter Eccentricity you want to search for", '0',
                                   help="Orbital eccentricity (0-1, e.g., '0.001')")
        inclination = st.text_input("Enter Inclination you want to search for", '0',
                                  help="Orbital inclination in degrees (e.g., '51.6')")
    
    with col2:
        ra_of_asc_node = st.text_input("Enter RA_OF_ASC_NODE you want to search for", '0',
                                     help="Right ascension of ascending node in degrees")
        arg_of_pericenter = st.text_input("Enter ARG_OF_PERICENTER you want to search for", '0',
                                        help="Argument of pericenter in degrees")
    
    # Search button
    if st.button("Search and Plot Orbits", type="primary"):
        with st.spinner("Fetching TLE data..."):
            tle_data = fetch_tle_data()
        
        if not tle_data:
            st.error("No TLE data available. Please try again later.")
            return
        
        # Filter satellites based on user criteria
        filtered_satellites = filter_satellites(
            tle_data, object_name, eccentricity, inclination, 
            ra_of_asc_node, arg_of_pericenter
        )
        
        if not filtered_satellites:
            st.warning("No satellites found matching your criteria. Try adjusting your search parameters.")
            return
        
        # Check if showing all satellites (no criteria provided)
        has_any_criteria = any([
            object_name and object_name.strip() != '0' and object_name.strip() != '',
            eccentricity and eccentricity.strip() != '0' and eccentricity.strip() != '',
            inclination and inclination.strip() != '0' and inclination.strip() != '',
            ra_of_asc_node and ra_of_asc_node.strip() != '0' and ra_of_asc_node.strip() != '',
            arg_of_pericenter and arg_of_pericenter.strip() != '0' and arg_of_pericenter.strip() != ''
        ])
        
        if has_any_criteria:
            st.success(f"Found {len(filtered_satellites)} satellite(s) matching your criteria:")
        else:
            st.info(f"No search criteria provided. Showing all {len(filtered_satellites)} satellites:")
            st.warning("⚠️ Plotting all satellites may take time and create a cluttered visualization. Consider using search criteria for better results.")
        
        # Display satellite information
        satellite_info = []
        for sat in filtered_satellites[:10]:  # Limit to first 10 for display
            satellite_info.append({
                'Name': sat.get('OBJECT_NAME', 'Unknown'),
                'Eccentricity': f"{float(sat.get('ECCENTRICITY', 0)):.6f}",
                'Inclination': f"{float(sat.get('INCLINATION', 0)):.2f}°",
                'Mean Motion': f"{float(sat.get('MEAN_MOTION', 0)):.8f} rev/day",
                'RA of Asc Node': f"{float(sat.get('RA_OF_ASC_NODE', 0)):.2f}°",
                'Arg of Pericenter': f"{float(sat.get('ARG_OF_PERICENTER', 0)):.2f}°"
            })
        
        st.dataframe(pd.DataFrame(satellite_info))
        search_info_about_sats(satellite_info)
        if len(filtered_satellites) > 10:
            st.info(f"Showing first 10 satellites. {len(filtered_satellites) - 10} more found.")
        
        # Plot orbits
        st.subheader("Orbital Trajectories")
        with st.spinner("Generating orbital plots..."):
            # For all satellites (no criteria), limit to first 20 for reasonable visualization
            # For filtered searches, limit to first 10
            if not has_any_criteria:
                plot_satellites = filtered_satellites[:20]
                if len(filtered_satellites) > 20:
                    st.info("Plotting first 20 satellites from all available satellites for optimal performance.")
            else:
                plot_satellites = filtered_satellites[:10]
                if len(filtered_satellites) > 10:
                    st.info("Plotting first 10 satellites for optimal performance.")
            
            fig = plot_orbits(plot_satellites)
            st.pyplot(fig)
            AI_insite_to_plots(fig)
        # Display orbital parameters
        st.subheader("Orbital Parameters")
        for i, sat in enumerate(plot_satellites):
            with st.expander(f"{sat.get('OBJECT_NAME', 'Unknown')} - Orbital Details"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Eccentricity", f"{float(sat.get('ECCENTRICITY', 0)):.6f}")
                    st.metric("Inclination", f"{float(sat.get('INCLINATION', 0)):.2f}°")
                
                with col2:
                    st.metric("Mean Motion", f"{float(sat.get('MEAN_MOTION', 0)):.8f} rev/day")
                    period_hours = 24 / float(sat.get('MEAN_MOTION', 1))
                    st.metric("Orbital Period", f"{period_hours:.2f} hours")
                
                with col3:
                    st.metric("RA of Asc Node", f"{float(sat.get('RA_OF_ASC_NODE', 0)):.2f}°")
                    st.metric("Arg of Pericenter", f"{float(sat.get('ARG_OF_PERICENTER', 0)):.2f}°")

    # Add some example searches
    st.sidebar.header("Example Searches")
    st.sidebar.write("**International Space Station:**")
    st.sidebar.write("Object Name: ISS")
    st.sidebar.write("")
    st.sidebar.write("**Starlink Satellites:**")
    st.sidebar.write("Object Name: STARLINK")
    st.sidebar.write("")
    st.sidebar.write("**Geostationary Satellites:**")
    st.sidebar.write("Inclination: 0")
    st.sidebar.write("Eccentricity: 0")
    st.sidebar.write("")
    st.sidebar.write("**Show All Satellites:**")
    st.sidebar.write("Leave all fields as '0' or empty")
    

if __name__ == "__main__":
    main()
