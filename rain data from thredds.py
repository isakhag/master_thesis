# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:02:15 2023

@author: Isak9
"""
#%% libraries
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
from datetime import date, timedelta, datetime, timezone
import pyproj
import cartopy.crs as ccrs
import xarray as xr
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
from rasterio.transform import from_origin
from rasterio.transform import Affine
import os
from osgeo import gdal #gis library
import time, json, urllib
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
#%%RETURN PERIOD, spezialisation project
#rr er rainfall, sdfswx er nok mer standard deviation
#1dag
#file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/return_levels/version_22.09/M100rr1d_seNorge2018_v2209_GEV_1991-2020.nc'
#file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/return_levels/version_22.09/M10rr1d_seNorge2018_v2209_GEV_1991-2020.nc'
#file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/return_levels/version_22.09/M50rr1d_seNorge2018_v2209_GEV_1991-2020.nc'
#file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/return_levels/version_23.09/M100rr1d_seNorge2018_v2309_GEV_1991-2020.nc'
#3dager
#file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/return_levels/version_22.09/M10rr3d_seNorge2018_v2209_GEV_1991-2020.nc'
#file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/return_levels/version_22.09/M100rr3d_seNorge2018_v2209_GEV_1991-2020.nc'
#file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/return_levels/version_22.09/M50rr3d_seNorge2018_v2209_GEV_1991-2020.nc'


# f = netCDF4.Dataset(file_path, 'r')

# plt.rcParams.update({'font.size': 14})

# point_x = 203575
# point_y = 6855333

# x_data = f.variables['X'][:]
# y_data = f.variables['Y'][:]

# # Find the indices corresponding to the target coordinates
# x_indices = np.abs(x_data - point_x).argmin()
# y_indices = np.abs(y_data - point_y).argmin()
# # Check if the coordinates are within the range of the data
# if 0 <= x_indices < x_data.size and 0 <= y_indices < y_data.size:
#     # Extract the return period value at the specified coordinates
#     m100_data = f.variables['M100'][:]  # Change this to match the return period you want
#     return_period_value = m100_data[0, y_indices, x_indices]

#     print(f'Return period value at coordinates ({point_x}, {point_y}): {return_period_value}')
# else:
#     print('Coordinates are out of range.')

# # Close the NetCDF file
# f.close()

# try:
#     with netCDF4.Dataset(file_path, 'r') as ncfile:
#         # Get a list of variable names in the file
#         variable_names = list(ncfile.variables.keys())
            
#         for name in variable_names:
#                 variable = ncfile.variables[name]
#                 print(f"Variable: {name}")
#                 print(f"Shape: {variable.shape}")
#                 print(f"Dimensions: {variable.dimensions}")
#                 print("Attributes:")
#                 for attr_name in variable.ncattrs():
#                     attr_value = variable.getncattr(attr_name)
#                     print(f"  {attr_name}: {attr_value}")
#                 print("\n")

# except Exception as e:
#     print(f"Error opening the NetCDF file: {e}")
    
# try:
#     # Extract the 'M50', 'X', and 'Y' variables
#     m5_data = f.variables['M100'][:]
#     x_data = f.variables['X'][:]
#     y_data = f.variables['Y'][:]
    
#     point_x = 189847 
#     point_y = 6874417
#     label = 'Vågåmo' #Vågåmo coordinates

#     point_x1 = 191632 
#     point_y1 = 6873374
#     label1 = 'Point 2'

#     point_x2 = 184854 
#     point_y2 = 6852316
#     label2 = 'Point 3'

#     point_x3 = 203575
#     point_y3 =  6855333
#     label3 = 'Point 4'

#     point_x4 = 155291
#     point_y4 =  6872373
#     label4 = 'Point 1'
    
#     x_min = 135000  
#     x_max = 225910  
#     y_min = 6839596  
#     y_max = 6896745  

#     # Find the indices corresponding to your desired coordinates
#     x_min_idx = np.argmin(np.abs(x_data - x_min))
#     x_max_idx = np.argmin(np.abs(x_data - x_max))
#     y_min_idx = np.argmin(np.abs(y_data - y_min))
#     y_max_idx = np.argmin(np.abs(y_data - y_max))

#     # Slice the data within the specified region
#     data_slice = m5_data[0, y_min_idx:y_max_idx, x_min_idx:x_max_idx]
#     # x_min_idx = np.argmin(np.abs(x_data - x_min))
#     # x_max_idx = np.argmin(np.abs(x_data - x_max))
#     # y_min_idx = np.argmin(np.abs(y_data - y_min))
#     # y_max_idx = np.argmin(np.abs(y_data - y_max))

#     # Create a contour plot or heatmap
#     plt.figure(figsize=(10, 6))
#     #contour = plt.contourf(x_data[x_min_idx:x_max_idx], y_data[y_min_idx:y_max_idx], m5_data[0, y_min_idx:y_max_idx, x_min_idx:x_max_idx], cmap='viridis')
#     #contour = plt.contourf(x_data, y_data, m5_data[0,:,:], cmap='viridis')
#     levels = [0, 20, 40, 60, 80, 100, 120]
#     contour = plt.contourf(x_data[x_min_idx:x_max_idx], y_data[y_min_idx:y_max_idx], data_slice, cmap='viridis', levels=levels, fontsize=14)
    
#     plt.scatter([point_x, point_x1, point_x2, point_x3, point_x4], [point_y, point_y1, point_y2, point_y3, point_y4], c='red', marker='.', s=100)
#     plt.text(point_x-8000, point_y+3000, label, color='red', fontsize=14, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
#     plt.text(point_x2+2500, point_y2+2500, label2, color='red', fontsize=14, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
#     plt.text(point_x3+2500, point_y3+2500, label3, color='red', fontsize=14, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
#     plt.text(point_x4+2500, point_y4+2500, label4, color='red', fontsize=14, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
#     plt.text(point_x1+2500, point_y1+2500, label1, color='red', fontsize=14, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
#     # plt.scatter([point_x], [point_y], c='red', marker='.', s=150)
#     # plt.text(point_x+2500, point_y+2500, label, color='red', fontsize=14, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
#     # Add color bar
#     cbar = plt.colorbar(contour)
#     cbar.set_label('[mm]', fontsize=14)
    
#     square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
#     square = Polygon(square_coords)
#     x_square, y_square = square.exterior.xy
#     plt.plot(x_square, y_square, color='red', linewidth=2)

#     plt.xlabel('East [m]', fontsize=14)
#     plt.ylabel('North [m]', fontsize=14)
#     plt.title('        Contour plot of T = 100 years return period for rainfall', fontsize=14) #WGS84 coordinate system
#     plt.grid(True)

#     # Show the plot
#     plt.show()

# except Exception as e:
#     print(f"Error opening the NetCDF file: {e}")

# # Close the NetCDF file
# f.close()

#%%METEOROLOGISK SITUASJON, spezialisation project
# file_path = 'https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/06/met_analysis_1_0km_nordic_20230806T17Z.nc'

# f = netCDF4.Dataset(file_path, 'r')

# try:
#     with netCDF4.Dataset(file_path, 'r') as ncfile:
#         # Get a list of variable names in the file
#         variable_names = list(ncfile.variables.keys())
            
#         for name in variable_names:
#                 variable = ncfile.variables[name]
#                 print(f"Variable: {name}")
#                 print(f"Shape: {variable.shape}")
#                 print(f"Dimensions: {variable.dimensions}")
#                 print("Attributes:")
#                 for attr_name in variable.ncattrs():
#                     attr_value = variable.getncattr(attr_name)
#                     print(f"  {attr_name}: {attr_value}")
#                 print("\n")

# except Exception as e:
#     print(f"Error opening the NetCDF file: {e}")


#https://thredds.met.no/thredds/catalog/metpparchive/2023/08/catalog.html to find the other days

#filename = "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/06/met_analysis_1_0km_nordic_20230806T17Z.nc"
#filename = "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/07/met_analysis_1_0km_nordic_20230807T17Z.nc"
#filename = "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/08/met_analysis_1_0km_nordic_20230808T17Z.nc"
#filename = "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/09/met_analysis_1_0km_nordic_20230809T17Z.nc"
#filename = "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/10/met_analysis_1_0km_nordic_20230810T17Z.nc"

#filename = "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/07/met_analysis_1_0km_nordic_20230807T11Z.nc"
#filename = "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/08/met_analysis_1_0km_nordic_20230808T11Z.nc"
#filename = "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/09/met_analysis_1_0km_nordic_20230809T11Z.nc"


# List of file names
# file_names = [
#     "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/07/met_analysis_1_0km_nordic_20230807T11Z.nc",
#     "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/08/met_analysis_1_0km_nordic_20230808T11Z.nc",
#     "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/09/met_analysis_1_0km_nordic_20230809T11Z.nc",
# ]

# # Create subplots
# fig, axs = plt.subplots(nrows=len(file_names), subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(8, 6 * len(file_names)))

# # Initialize variables to store colorbar limits
# min_precipitation = float('inf')
# max_precipitation = float('-inf')
# min_pressure = float('inf')
# max_pressure = float('-inf')

# for i, file_name in enumerate(file_names):
#     ncfile = xr.open_dataset(file_name)
#     timestep = 0
#     air_pressure_data = ncfile.air_pressure_at_sea_level.isel(time=timestep)
#     precipitation_data = ncfile.precipitation_amount.isel(time=timestep)
#     latitudes = ncfile.latitude
#     longitudes = ncfile.longitude

#     # Update colorbar limits
#     min_precipitation = min(min_precipitation, precipitation_data.min().item())
#     max_precipitation = max(max_precipitation, precipitation_data.max().item())
#     min_pressure = min(min_pressure, air_pressure_data.min().item())
#     max_pressure = max(max_pressure, air_pressure_data.max().item())

#     # Format the date and time
#     date_str = str(air_pressure_data.time.values).split('T')[0]
#     time_str = str(air_pressure_data.time.values).split('T')[-1][:5]

#     # Plot the data on a map
#     ax = axs[i]
#     pcm2 = ax.pcolormesh(longitudes, latitudes, precipitation_data, cmap='Reds', shading='auto', vmin=0, vmax=30)
#     contour_levels = list(range(int(min_pressure), int(max_pressure), 500))
#     contour = ax.contour(longitudes, latitudes, air_pressure_data, levels=contour_levels, colors='blue', linewidths=0.5)
#     ax.clabel(contour, inline=True, fmt='%1.0f', colors='blue', fontsize=8)
#     cbar2 = plt.colorbar(pcm2, ax=ax, extend='max', shrink=0.8)
#     cbar2.set_label('Precipitation Amount [mm]', fontsize=12)
#     ax.set_title(f'Air Pressure and Precipitation on {date_str} at {time_str}', fontsize=14)
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=':')

# plt.tight_layout()
# plt.show()

# filename = "https://thredds.met.no/thredds/dodsC/metpparchive/2023/08/06/met_analysis_1_0km_nordic_20230806T17Z.nc"
# ncfile = xr.open_dataset(filename)

# timestep = 0
# air_pressure_data = ncfile.air_pressure_at_sea_level.isel(time=timestep)
# precipitation_data = ncfile.precipitation_amount.isel(time=timestep)


# date_str = str(air_pressure_data.time.values).split('T')[0]
# time_str = str(air_pressure_data.time.values).split('T')[-1][:5]

# print(ncfile.air_pressure_at_sea_level.shape)
# timestep = 0
# air_pressure_data = ncfile.air_pressure_at_sea_level.isel(time=timestep)
# precipitation_data = ncfile.precipitation_amount.isel(time=timestep)
# #ncfile.air_pressure_at_sea_level.isel(time=timestep).plot()
# latitudes = ncfile.latitude
# longitudes = ncfile.longitude

# # Plot the data on a map
# fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
# pcm2 = ax.pcolormesh(longitudes, latitudes, precipitation_data, cmap='Reds', shading='auto', vmin=0, vmax=30)
# contour_levels = list(range(int(air_pressure_data.min()), int(air_pressure_data.max()), 500))
# contour = ax.contour(longitudes, latitudes, air_pressure_data, levels=contour_levels, colors='blue', linewidths=0.5)
# ax.clabel(contour, inline=True, fmt='%1.0f', colors='blue', fontsize=8)
# cbar2 = plt.colorbar(pcm2, ax=ax, extend='max', shrink=0.6)
# cbar2.set_label('Precipitation Amount [mm]', fontsize=14)
# #cbar2.ax.tick_params(labelsize=14)
# ax.set_title(f'Air Pressure and Precipitation on {date_str} at {time_str}', fontsize=14)
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# plt.show()
 



#%% precipitation data seNorge 1km grid 

# file_paths = ['https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230807.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230808.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230809.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230810.nc',
# ]

# # Define your square coordinates within the UTM zone 33N system
# square_coords = [(140000, 6896745), (140000, 6844000), (220000, 6844000), (220000, 6896745)]

# # Load the first dataset to get the spatial coordinates
# ncfile = netCDF4.Dataset(file_paths[0])

# # Extract the X (longitude) and Y (latitude) values - in this context, they are easting and northing
# longitudes = ncfile.variables["X"][:]
# latitudes = ncfile.variables["Y"][:]

# # Find the indices that correspond to your square's boundaries
# x_min, y_min = min(coord[0] for coord in square_coords), min(coord[1] for coord in square_coords)
# x_max, y_max = max(coord[0] for coord in square_coords), max(coord[1] for coord in square_coords)
# x_min_idx = np.argmin(np.abs(longitudes - x_min))
# x_max_idx = np.argmin(np.abs(longitudes - x_max))
# y_min_idx = np.argmin(np.abs(latitudes - y_min))
# y_max_idx = np.argmin(np.abs(latitudes - y_max))

# # Correct the order if necessary
# if y_min_idx > y_max_idx:
#     y_min_idx, y_max_idx = y_max_idx, y_min_idx
#%% plotting the wanted file paths. 
# ncfile = netCDF4.Dataset(file_paths[0])

# # Assuming 'rr' is the variable name for precipitation
# precipitation = ncfile.variables['rr'][:]

# # Adjust for time dimension if present
# if precipitation.ndim > 2:
#     precipitation = precipitation[0, :, :]  # Taking the first time step

# # Extract the subset for the specific square region
# precip_subset = precipitation[y_min_idx:y_max_idx + 1, x_min_idx:x_max_idx + 1]

# # Adjust longitude and latitude arrays to have one extra element for pcolormesh
# # This extra element is needed for the right/top edges of the last cells
# if len(longitudes) > precip_subset.shape[1]:
#     longitudes_subset = longitudes[x_min_idx:x_max_idx + 2]  # One extra element
# else:
#     longitudes_subset = longitudes[x_min_idx:x_max_idx + 1]

# if len(latitudes) > precip_subset.shape[0]:
#     latitudes_subset = latitudes[y_min_idx:y_max_idx + 2]  # One extra element
# else:
#     latitudes_subset = latitudes[y_min_idx:y_max_idx + 1]

# # Plotting the data
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(longitudes_subset, latitudes_subset, precip_subset, shading='flat')
# plt.colorbar(label='Precipitation (mm)')
# plt.title('Precipitation on the first available date')
# plt.xlabel('Easting (m)')
# plt.ylabel('Northing (m)')
# plt.show()
#%% combining the file paths to get total precipitation during hans and writing to raster file
# Load and combine the datasets
# datasets = [xr.open_dataset(fp) for fp in file_paths]
# combined_dataset = xr.concat(datasets, dim='time')

# # Sum the precipitation data within the specified bounds
# total_precip = combined_dataset['rr'][:, y_min_idx:y_max_idx, x_min_idx:x_max_idx].sum(dim='time')

# # Plotting the data
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(longitudes[x_min_idx:x_max_idx], latitudes[y_min_idx:y_max_idx], total_precip, shading='auto')
# plt.colorbar(label='Total Precipitation (mm)')
# plt.title('Total Precipitation within Specified Square Over Period')
# plt.xlabel('Easting (m)')
# plt.ylabel('Northing (m)')
# plt.show()

# # Define the pixel size (resolution) of your raster cells
# pixel_size_x = 1000  # Adjust this based on the actual resolution of your data
# pixel_size_y = -1000  # This should be negative to move southward

# # Calculate the top left x and y coordinates
# # For the top_left_y, since the pixel size is negative, we add (not subtract) the height of the raster in terms of the coordinate system
# top_left_x = longitudes[x_min_idx]
# top_left_y = latitudes[y_min_idx]

# transform = rasterio.transform.from_origin(top_left_x, top_left_y, pixel_size_x, -pixel_size_y)

# # Save the data as a GeoTIFF
# with rasterio.open(
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/precipitation_amount_hans.tif',
#     'w',
#     driver='GTiff',
#     height=total_precip.shape[0],
#     width=total_precip.shape[1],
#     count=1,
#     dtype=total_precip.dtype,
#     crs="EPSG:25833",
#     transform=transform
# ) as new_dataset:
#     new_dataset.write(total_precip.values.astype(rasterio.float32), 1)

#%% precip data each day august 7 to august 10 and writing to raster file
# file_paths = [
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230722.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230723.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230724.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230725.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230726.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230727.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230728.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230729.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230730.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230731.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230801.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230802.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230803.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230804.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230805.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230806.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230807.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230808.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230809.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230810.nc',
# ]

# square_coords = [(140000, 6896745), (140000, 6844000), (220000, 6844000), (220000, 6896745)]

# # Define pixel size
# pixel_size_x = 1000  
# pixel_size_y = -1000  # This should be negative to move southward

# for idx, file_path in enumerate(file_paths):
#     ncfile = netCDF4.Dataset(file_path)
    
#     longitudes = ncfile.variables["X"][:]
#     latitudes = ncfile.variables["Y"][:]
    
#     x_min, y_min = min(coord[0] for coord in square_coords), min(coord[1] for coord in square_coords)
#     x_max, y_max = max(coord[0] for coord in square_coords), max(coord[1] for coord in square_coords)
    
#     x_min_idx = np.argmin(np.abs(longitudes - x_min))
#     x_max_idx = np.argmin(np.abs(longitudes - x_max))
#     y_min_idx = np.argmin(np.abs(latitudes - y_min))
#     y_max_idx = np.argmin(np.abs(latitudes - y_max))

#     # Ensure the indices are in the correct order for slicing the dataset
#     if y_min_idx > y_max_idx:
#         y_min_idx, y_max_idx = y_max_idx, y_min_idx

#     dataset = xr.open_dataset(file_path)
    
#     # Extract the precipitation data for the dataset
#     precip_data = dataset['rr'][:, y_min_idx:y_max_idx, x_min_idx:x_max_idx]

#     # Define the top-left corner
#     top_left_x = longitudes[x_min_idx]
#     top_left_y = latitudes[y_min_idx]

#     # Create the transform
#     transform = from_origin(top_left_x, top_left_y, pixel_size_x, -pixel_size_y)

#     # Define the output file name
#     output_file_name = f'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/precipitation_amount_{idx}.tif'

#     # Save the data as a GeoTIFF
#     with rasterio.open(
#         output_file_name,
#         'w',
#         driver='GTiff',
#         height=precip_data.shape[1],
#         width=precip_data.shape[2],
#         count=1,
#         dtype=str(precip_data.dtype),
#         crs="EPSG:25833",
#         transform=transform
#     ) as new_dataset:
#         new_dataset.write(precip_data.values[0].astype(rasterio.float32), 1)

#%% reading soil saturation data
time_ini = datetime(2023, 8, 7, 00) #year, month, day, hour
time_end = datetime(2023, 8, 10, 00) #year, month, day, hour
# xp = 189847 #Vågåmo
# yp = 6874417

def readData_xgeo(x, y, inidate, enddate): 
    '''
Reading time series of several climate parameters for the periode 1957-09-01 until today for a given (x,y - UTM33) position
 	Input: 	x,y 		easting and northing coordinate in UTM33
 			mode		('24h' or '3h') for reading 24 h or 3h data.
   
 	
  ** more parameters: http://gts.nve.no/api/GridTimeSeries/Themes/json

 	
 	Output:
 			t			numpy array of datetime values
 			ta			temperature, deg Celcius (1xN)
 			rr			precipitation, mm (1xN)
 			sd			snow depth, m (1xN)
 			fsw			fresh snow in water equivalent, mm (1xN)
 			swe			snow water equivalent, mm
 			gwb_gwtp30	ground water in % from the maximum in 30 years
 			gwb_gwt		groundwater (table depth?), mm
 			gwb_gwtdev  daily groundwater change, mm
 			eva 		evapotransporation, mm
 			gwb_sssdev soil water capacity, mm
 			gwb_sssrel, Vannmetning i jord, % (soil saturation)
             '''
    enddate1 = np.datetime64(enddate)
    inidate1 = np.datetime64(inidate)
    #BaseURL = ('http://h-web02.nve.no:8080/api/GridTimeSeries/' + str(x) + "/" + str(y) + "/" + inidate + "/" + enddate + "/")
    BaseURL = ('https://gts.nve.no/api/GridTimeSeries/' + str(x) + "/" + str(y) + "/" + inidate + "/" + enddate)
    t = np.arange(inidate1 + np.timedelta64(6, 'h'), enddate1+1+np.timedelta64(6, 'h'), timedelta(days=1)).astype(datetime)
    t3h = np.arange(inidate1, enddate1+np.timedelta64(3, 'h'), timedelta(hours=3)).astype(datetime)
    #print(BaseURL)
    def get(url, object_hook=None):
        with urllib.request.urlopen(url) as resource:  # 'with' is important to close the resource after use
            return json.load(resource, object_hook=object_hook)
    # data = get(BaseURL + "/tm" + ".json") # celcius
    # ta = np.array(data['Data'], dtype=float)
    # ta[ta == data['NoDataValue']] = np.NaN
    # altitude = np.array(data['Altitude'])
    
    # data = get(BaseURL + "/rr" + ".json") # mm
    # rr = np.array(data['Data'], dtype=float)
    # rr[rr == data['NoDataValue']] = np.NaN
    
    # data = get(BaseURL + "/rr3h" + ".json") # mm
    # rr3h = np.array(data['Data'], dtype=float)
    # rr3h[rr3h == data['NoDataValue']] = np.NaN

    # data = get(BaseURL + "fsw" + ".json") # mm
    # fsw = np.array(data['Data'])
    # #fsw[fsw >= 254] = np.NaN

    # data = get(BaseURL + "sd" + ".json") # cm
    # sd = np.array(data['Data'])
    # sd[sd == data['NoDataValue']] = np.NaN
    # sd = sd/100 # convert to meters snow depth

    # data = get(BaseURL + "swe" + ".json") #mm
    # swe = np.array(data['Data'])
    # swe[swe == data['NoDataValue']] = np.NaN

    # data = get(BaseURL + "/qsw" + ".json") #mm
    # qsw = np.array(data['Data'], dtype=float)
    # qsw[qsw == data['NoDataValue']] = np.NaN
     
    # data = get(BaseURL + "/gwb_gwtprgwb_gwtxyrx30yr" + ".json") # % (Grunnvann i % av maksimum)
    # gwtp30 = np.array(data['Data']).astype(float)
    # gwtp30[gwtp30 == data['NoDataValue']] = np.nan

    # data = get(BaseURL + "/gwb_gwt" + ".json") # % (Grunnvann i % av maksimum)
    # gwb_gwt = np.array(data['Data']).astype(float)
    # gwb_gwt[gwb_gwt == data['NoDataValue']] = np.nan
         
    # data = get(BaseURL + "/gwb_eva" + ".json") # fordampning
    # eva = np.array(data['Data']).astype(float)
    # eva[eva == data['NoDataValue']] = np.nan
     
    # data = get(BaseURL + "/gwb_sssdev" + ".json") # Jordas vannkapasitet
    # gwb_sssdev = np.array(data['Data']).astype(float)
    # gwb_sssdev[gwb_sssdev == data['NoDataValue']] = np.nan
     
    data = get(BaseURL + "/gwb_sssrel" + ".json") # Vannmetning i jord
    gwb_sssrel = np.array(data['Data']).astype(float)
    gwb_sssrel[gwb_sssrel == data['NoDataValue']] = np.nan
     
    # return t, t3h, ta, rr, rr3h, qsw, gwb_gwt, eva, gwb_sssdev, gwb_sssrel
    return gwb_sssrel

# Use function to read SeNorge data at a given location.  
inidate = time_ini.strftime("%Y-%m-%d")
enddate = time_end.strftime("%Y-%m-%d")
# t, t3h, ta, rr, rr3h, qsw, gwb_gwt, eva, gwb_sssdev, gwb_sssrel = readData_xgeo(xp, yp, inidate, enddate)
#gwb_sssrel = readData_xgeo(xp, yp, inidate, enddate)

# Define your square's corners
min_x, max_x = 134000, 220000  # Min and max easting
min_y, max_y = 6844000, 6895000  # Min and max northing
grid_size = 1000
saturation_data = []

for x in range(min_x, max_x, grid_size):  # Step by 1000 meters (1 km)
    for y in range(max_y, min_y, -grid_size):  # Step by 1000 meters (1 km), moving southward
        # Fetch the saturation data for the grid cell
        saturation_values = readData_xgeo(x, y, inidate, enddate)
        
        saturation_data.append({
            'x_min': x,
            'x_max': x + grid_size,
            'y_min': y - grid_size,
            'y_max': y,
            'saturation_value': saturation_values 
        })

df = pd.DataFrame(saturation_data)

# date_labels = ["2023-07-22", "2023-07-23", "2023-07-24", "2023-07-25", 
#                 "2023-07-26", "2023-07-27", "2023-07-28", "2023-07-29", 
#                 "2023-07-30", "2023-07-31", "2023-08-01", "2023-08-02", 
#                 "2023-08-03", "2023-08-04", "2023-08-05", "2023-08-06", 
#                 "2023-08-07", "2023-08-08", "2023-08-09", "2023-08-10"]
date_labels = ["2023-08-07", "2023-08-08", "2023-08-09", "2023-08-10"]
for i, date in enumerate(date_labels):
    df[date] = df['saturation_value'].apply(lambda x: x[i])
df = df.drop('saturation_value', axis=1)


#%% writing to raster
# date_columns = ["2023-08-07", "2023-08-08", "2023-08-09", "2023-08-10"]
# date_columns = ["2023-07-22", "2023-07-23", "2023-07-24", "2023-07-25", 
#                 "2023-07-26", "2023-07-27", "2023-07-28", "2023-07-29", 
#                 "2023-07-30", "2023-07-31", "2023-08-01", "2023-08-02", 
#                 "2023-08-03", "2023-08-04", "2023-08-05", "2023-08-06", 
#                 "2023-08-07", "2023-08-08", "2023-08-09", "2023-08-10"]

# cell_size = 1000
# width = int((max_x - min_x) / cell_size)
# height = int((max_y - min_y) / cell_size)
# transform = from_origin(west=min_x, north=max_y, xsize=cell_size, ysize=cell_size)

# for date_column in date_columns:
#     output_raster_file = f'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/saturation_raster_{date_column}.tif'

#     with rasterio.open(
#         output_raster_file,
#         'w',
#         driver='GTiff',
#         height=height,
#         width=width,
#         count=1,
#         dtype=df[date_column].dtype,
#         crs='EPSG:25833',  
#         transform=transform,
#     ) as dst:
#         data_array = np.full((height, width), np.nan)
        
        
#         for index, row in df.iterrows():
#             x_idx = int((row['x_min'] - min_x) / cell_size)
#             y_idx = height - int((row['y_max'] - min_y) / cell_size) - 1
#             data_array[y_idx, x_idx] = row[date_column]
        
#         dst.write(data_array, 1)


#%%Finding the average precipitation for each day over the total study area
# file_paths = [
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230807.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230808.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230809.nc',
#     'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230810.nc'
# ]

# # Area coordinates
# square_coords = [(140000, 6896745), (140000, 6844000), (220000, 6844000), (220000, 6896745)]
# x_min, y_min = min(coord[0] for coord in square_coords), min(coord[1] for coord in square_coords)
# x_max, y_max = max(coord[0] for coord in square_coords), max(coord[1] for coord in square_coords)

# # Process each file
# for file_path in file_paths:
#     date = file_path[-11:-3]
#     # Load the dataset
#     ncfile = netCDF4.Dataset(file_path)
    
#     # Extract the spatial coordinates
#     longitudes = ncfile.variables["X"][:]
#     latitudes = ncfile.variables["Y"][:]
    
#     # Find the indices for the square's boundaries
#     x_min_idx = np.argmin(np.abs(longitudes - x_min))
#     x_max_idx = np.argmin(np.abs(longitudes - x_max))
#     y_min_idx = np.argmin(np.abs(latitudes - y_min))
#     y_max_idx = np.argmin(np.abs(latitudes - y_max))
    
#     # Correct the order
#     if y_min_idx > y_max_idx:
#         y_min_idx, y_max_idx = y_max_idx, y_min_idx
    
#     # Extract the precipitation variable rr
#     precipitation = ncfile.variables['rr'][0, y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
    
#     # Calculate the average precipitation in the area
#     area_avg_precip = np.mean(precipitation)
#     print(f"Average precipitation for {date} is {area_avg_precip:.2f} mm/day")

#     # Close the dataset
#     ncfile.close()


#%%
    