# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:19:39 2024

@author: Isak9
"""

import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import from_origin

# # Original raster path
# raster_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/sat_aug_2_raster.tif'
# # Modified raster path
# modified_raster_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/modified_sat_aug_2_raster.tif'

# # Define your value range and the replacement value
# lower_bound = 10**10
# upper_bound = 10**18
# replacement_value = 72

# # Function to modify raster data
# def modify_raster_data(raster_path, modified_raster_path, lower_bound, upper_bound, replacement_value):
#     with rasterio.open(raster_path) as src:
#         data = src.read(1)
#         mask = (data >= lower_bound) & (data <= upper_bound)
#         data[mask] = replacement_value
#         out_meta = src.meta
        
#         with rasterio.open(modified_raster_path, 'w', **out_meta) as dest:
#             dest.write(data, 1)

# # Modify the raster
# modify_raster_data(raster_path, modified_raster_path, lower_bound, upper_bound, replacement_value)

# # Function to inspect modified raster data
# def inspect_raster_data(modified_raster_path):
#     with rasterio.open(modified_raster_path) as src:
#         data = src.read(1)
#         mask = (data >= lower_bound) & (data <= upper_bound)
#         count_of_values_in_range = np.sum(mask)
#         unique_values_in_range = np.unique(data[mask])

#         print(f"Number of values between {lower_bound} and {upper_bound} in the modified file: {count_of_values_in_range}")
#         print(f"Unique values between {lower_bound} and {upper_bound} in the modified file: {unique_values_in_range}")

# # Inspect the modified raster
# inspect_raster_data(modified_raster_path)

#%% changing the nan data values in soil sat rasters to specified values
# Define your file paths
# raster_path = [
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/saturation_raster_2023-08-07.tif',
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/saturation_raster_2023-08-08.tif',
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/saturation_raster_2023-08-09.tif',
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/saturation_raster_2023-08-10.tif'
# ]

# modified_raster_path = [
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/modified_sat_raster_2023-08-07.tif',
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/modified_sat_raster_2023-08-08.tif',
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/modified_sat_raster_2023-08-09.tif',
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/modified_sat_raster_2023-08-10.tif'
# ]
# # Define the replacement values for each file
# replacement_values = [65, 90, 106, 106]

# # Function to modify raster data
# def modify_raster_data(raster_paths, modified_paths, replacement_vals):
#     for raster_file, modified_file, replacement_val in zip(raster_paths, modified_paths, replacement_vals):
#         with rasterio.open(raster_file) as src:
#             data = src.read(1)
#             mask = np.isnan(data)
#             data[mask] = replacement_val
#             out_meta = src.meta
            
#             with rasterio.open(modified_file, 'w', **out_meta) as dest:
#                 dest.write(data, 1)

# # Modify the raster files
# modify_raster_data(raster_path, modified_raster_path, replacement_values)

# # Adjusted function to inspect raster data
# def inspect_raster_data(modified_raster_path, replacement_value):
#     with rasterio.open(modified_raster_path) as src:
#         data = src.read(1)
#         # Create a mask for the specific replacement value
#         mask = (data == replacement_value)
#         count_of_replacement_values = np.sum(mask)
#         print(f"Number of '{replacement_value}' values in the modified file: {count_of_replacement_values}")

# # Define paths and replacement values
# modified_raster_paths = [
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/modified_sat_raster_2023-08-07.tif',
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/modified_sat_raster_2023-08-08.tif',
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/modified_sat_raster_2023-08-09.tif',
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/modified_sat_raster_2023-08-10.tif'
# ]
# replacement_values = [65, 90, 106, 106]

# # Loop through each modified raster to apply the inspection
# for path, replacement in zip(modified_raster_paths, replacement_values):
#     inspect_raster_data(path, replacement)


#%% creating excel files to open them in arcgis and showing the release points on August 8 and 9

# df_1 = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/dataset_ratios/ratio_copy.xlsx')

# landslide_data = df_1[df_1['landslide'] == 1]
# # Select the columns 'x', 'y', and 'Day'
# selected_columns = landslide_data[['x', 'y', 'Day']]
# august_8 = selected_columns[(selected_columns['Day'] == 8)]
# august_9 = selected_columns[(selected_columns['Day'] == 9)]

# august_8.to_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/landslide initiation points/august_8.xlsx', index=False)
# august_9.to_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/landslide initiation points/august_9.xlsx', index=False)

#%%











































