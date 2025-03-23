# Description: This script demonstrates how to compare the density data from the Swarm mission with the NRLMSISE-00 model.
# The script loads the measured data from the esa swarm repository and then computes the atmospheric density using the NRLMSISE-00 model 
# which then filters it for a specific date and time range (same as swam data) and then compares the measured density with the computed density.

import cdflib
from cdflib.epochs import CDFepoch
import numpy as np
import pandas as pd
import pyatmos
from pyatmos import nrlmsise00, download_sw_nrlmsise00, read_sw_nrlmsise00
import matplotlib.pyplot as plt

# Swarm CDF file path
cdf_file = r'F:\SDCS\Density study\Data\SW_OPER_DNSAACC_2__20140201T000000_20140201T235950_0201\SW_OPER_DNSAACC_2__20140201T000000_20140201T235950_0201.cdf'

# Loading CDF file
cdf = cdflib.CDF(cdf_file)

# Extracting data from the CDF file
time = cdf.varget('time')
density = cdf.varget('density')
altitude = cdf.varget('altitude')
latitude = cdf.varget('latitude')
longitude = cdf.varget('longitude')
lst = cdf.varget('local_solar_time')  # Local Solar Time

# Converting time from CDF_EPOCH to readable format
time = CDFepoch.to_datetime(time)

# Creating a DataFrame
swarm_df = pd.DataFrame({
    "nrl_time": time,
    "nrl_density": density,
    "nrl_altitude": altitude,
    "nrl_latitude": latitude,
    "nrl_longitude": longitude,
    "nrl_local_solar_time": lst,
})

# Filter data for the date 2014-02-01
swarm_df = swarm_df[swarm_df["nrl_time"].dt.date == pd.to_datetime("2014-02-01").date()]

# Filter data to only include the range from 00:00:10 to 00:00:40
swarm_df = swarm_df[(swarm_df["nrl_time"].dt.time >= pd.to_datetime("00:00:10").time()) &
                    (swarm_df["nrl_time"].dt.time <= pd.to_datetime("00:00:40").time())]

print(swarm_df.head())

# Download and read space weather data
swfile = download_sw_nrlmsise00()
swdata = read_sw_nrlmsise00(swfile)

# Computing atmospheric density using NRLMSISE-00 model
computed_densities = []

# Iterate over each row in the filtered DataFrame
for index, row in swarm_df.iterrows():
    # Convert time to string format required by nrlmsise00()
    time_str = row["nrl_time"].strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract the necessary inputs for NRLMSISE-00: latitude, longitude, altitude, and local solar time
    latitude = row["nrl_latitude"]
    longitude = row["nrl_longitude"]
    altitude = row["nrl_altitude"] / 1000  # Convert altitude to kilometers (assuming it's in meters in the dataset)
    local_solar_time = row["nrl_local_solar_time"]
    
    # Run NRLMSISE-00 model with the provided data
    result = nrlmsise00(time_str, (latitude, longitude, altitude), swdata)
    
    # Extract the density from the result and append it to the list
    computed_densities.append(result.rho)

# Add the computed density values to the DataFrame as a new column
swarm_df["nrlmsise_density"] = computed_densities

# Print the updated DataFrame with density included
print(swarm_df[["nrl_time", "nrl_density", "nrl_altitude", "nrl_latitude", "nrl_longitude", "nrl_local_solar_time", "nrlmsise_density"]])

# Plotting the density
plt.figure(figsize=(10, 5))
plt.plot(swarm_df["nrl_time"], swarm_df["nrl_density"], label="Swarm Density (Measured)", color="red")
plt.plot(swarm_df["nrl_time"], swarm_df["nrlmsise_density"], label="NRLMSISE-00 Density (Model)", color="blue")
plt.xlabel("Time (UTC)")
plt.ylabel("Density (kg/m³)")
plt.title("Comparison of Swarm Density Data vs. NRLMSISE-00 Model")
plt.legend()
plt.grid()
plt.show()
