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

#%%RETURN PERIOD
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

#%%METEOROLOGISK SITUASJON
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
 
#%% precipitation and soil moisture thredds data 1 km grid

# file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230806.nc'
# file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230807.nc'
file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230808.nc'
# file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230809.nc'
# file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230810.nc'

#file_path = 'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Archive/seNorge2018_2023.nc'
square_coords = [(135438, 6906745), (135438, 6840000), (220000, 6840000), (220000, 6906745)]

ncfile   = netCDF4.Dataset(file_path)
for variable in ncfile.variables:
    print(variable)

latitudes = ncfile.variables["Y"][:]
longitudes = ncfile.variables["X"][:]

x_min, y_min = min(coord[0] for coord in square_coords), min(coord[1] for coord in square_coords)
x_max, y_max = max(coord[0] for coord in square_coords), max(coord[1] for coord in square_coords)
x_min_idx = np.argmin(np.abs(longitudes - x_min))
x_max_idx = np.argmin(np.abs(longitudes - x_max))
y_min_idx = np.argmin(np.abs(latitudes - y_min))
y_max_idx = np.argmin(np.abs(latitudes - y_max))

# Correct the order
if y_min_idx > y_max_idx:
    y_min_idx, y_max_idx = y_max_idx, y_min_idx

file_paths = [
    'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230808.nc',
    'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230809.nc',
    'https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/seNorge2018_20230810.nc',
]

datasets = [xr.open_dataset(fp) for fp in file_paths]
combined_dataset = xr.concat(datasets, dim='time')
total_precip = combined_dataset['rr'][:, y_min_idx:y_max_idx, x_min_idx:x_max_idx].sum(dim='time')

plt.figure(figsize=(10, 6))
plt.pcolormesh(longitudes[x_min_idx:x_max_idx], latitudes[y_min_idx:y_max_idx], total_precip, shading='auto')
plt.colorbar(label='Total Precipitation (mm)')
plt.title('Total Precipitation within Specified Square Over Period')
plt.xlabel('Longitude (X)')
plt.ylabel('Latitude (Y)')
plt.show()

pixel_size=1000

top_left_x = longitudes[x_min_idx]
top_left_y = latitudes[y_max_idx]
transform = rasterio.transform.from_origin(top_left_x, top_left_y, pixel_size, pixel_size)

with rasterio.open(
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/precipitation_amount_hans.tif',
    'w',
    driver='GTiff',
    height=total_precip.shape[0],
    width=total_precip.shape[1],
    count=1,
    dtype=total_precip.dtype,
    crs="EPSG:25833",
    transform=transform
) as new_dataset:
    new_dataset.write(total_precip.values.astype(rasterio.float32), 1)


# try:
#     # Extract the 'M50', 'X', and 'Y' variables
#     m5_data = f.variables['rr'][:]
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
#     plt.title('Rainfall', fontsize=14) #WGS84 coordinate system
#     plt.grid(True)
#     plt.show()

# except Exception as e:
#     print(f"Error opening the NetCDF file: {e}")
# f.close()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    