# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:28:56 2024

@author: Isak9
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import box
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from rasterio.enums import Resampling
#%% cohesion values with tensile strength based on equation
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NVE_NGIHVL.shp"
# input_raster = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRBMU.tif"
# input_raster2 = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTRESLAG.tif"

# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)
# pts = gpd.read_file(input_shapefile)  
# pts = pts[pts.geometry.within(square_polygon)]

# shapefile = gpd.read_file(input_shapefile)
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)

# with rasterio.open(input_raster) as src1:
#     raster1, raster1_transform = mask(src1, [square_polygon], crop=True)
#     raster1 = raster1[0]

# with rasterio.open(input_raster2) as src2:
#     raster2, raster2_transform = mask(src2, [square_polygon], crop=True)
#     raster2 = raster2[0] 
    
# raster1 = raster1.astype(float)
# raster2 = raster2.astype(float)    
# raster1[raster1 == -9999] = np.nan
# raster2[raster2 == -9999] = np.nan

# cohesion_gran = 1.2 * 11.55 * (raster1[raster2 == 1] / 451)
# cohesion_furu = 1.2 * 9.6 * (raster1[raster2 == 2] / 539)
# cohesion_lauv = 1.2 * 15.1 * (raster1[raster2 == 3] / 671)

# cohesion_values = np.zeros_like(raster2, dtype=float)
# cohesion_values[raster2 == 1] = cohesion_gran
# cohesion_values[raster2 == 2] = cohesion_furu
# cohesion_values[raster2 == 3] = cohesion_lauv

#%% sending the cohesion_values to a raster file
# output_raster_path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/cohesion/cohesion.tif"

# affine = raster1_transform
# crs = src1.crs  

# with rasterio.open(
#     output_raster_path, 'w',
#     driver='GTiff',
#     height=cohesion_values.shape[0],
#     width=cohesion_values.shape[1],
#     count=1,
#     dtype=cohesion_values.dtype,
#     crs=crs,
#     transform=affine,
#     nodata=np.nan  
# ) as dst:
#     dst.write(cohesion_values, 1)
    
# with rasterio.open(output_raster_path) as src:
#     crs = src.crs
#     print(f"CRS: {crs}")
    
#     fig, ax = plt.subplots(figsize=(10, 10))
    
#     show(src, ax=ax, title='Cohesion Map', cmap='viridis')  

#     ax.set_xlabel('Longitude' if crs.is_geographic else 'Easting')
#     ax.set_ylabel('Latitude' if crs.is_geographic else 'Northing')
#     ax.set_title('Root Cohesion Values in CRS')

# plt.show()

#%% kohesjonsverdier i ulike landslide points basert på nye kohesjonsverdier!
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NVE_NGIHVL.shp"
# input_raster = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/cohesion/cohesion_10/cohesion_10.tif"

# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)
# pts = gpd.read_file(input_shapefile)  
# pts = pts[pts.geometry.within(square_polygon)]

# with rasterio.open(input_raster) as src1:

#     coords = [(geom.x, geom.y) for geom in pts.geometry]
#     pts['cohesion'] = [x[0] for x in rasterio.sample.sample_gen(src1, coords)]

#%% determining area of the different forests. 
# Define the coordinates of the square polygon
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)

# # Read and mask the shapefile points within the square polygon
# shapefile = gpd.read_file(input_shapefile)
# pts = shapefile[shapefile.geometry.within(square_polygon)]

# # Function to mask and read raster data
# def mask_raster(input_raster, polygon):
#     with rasterio.open(input_raster) as src:
#         out_image, out_transform = mask(src, [polygon], crop=True)
#         out_meta = src.meta
#     return out_image[0], out_meta, out_transform

# # Mask raster2 to the square polygon
# raster2, raster2_meta, raster2_transform = mask_raster(input_raster2, square_polygon)

# # Get the unique values (bands) in raster2
# unique_values = np.unique(raster2)

# # Calculate the pixel area (assuming square pixels)
# pixel_size_x = raster2_transform[0]
# pixel_size_y = -raster2_transform[4]  # Typically negative, so take absolute value
# pixel_area = pixel_size_x * pixel_size_y

# # Calculate the total area for each unique value (band)
# area_dict = {}
# for value in unique_values:
#     area_dict[value] = np.sum(raster2 == value) * pixel_area / 1000000

# # Print the results
# for band, area in area_dict.items():
#     print(f"Band {band}: {area:.2f} square km")

#%% Areas of different forests parameters like height, diameter and basal area
# # Define file paths
# diam_path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRDIAMMIDDEL_GE8.tif"
# hoyde_path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRMHOYDE.tif"
# basal_path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRGRFLATE.tif"
# tree_type_path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTRESLAG.tif"

# # Define the coordinates of the square polygon
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)

# # Function to read and mask raster data
# def read_and_mask_raster(raster_path, polygon):
#     with rasterio.open(raster_path) as src:
#         out_image, out_transform = mask(src, [polygon], crop=True)
#         out_meta = src.meta
#     return out_image[0], out_meta, out_transform

# # Read and mask the rasters
# raster_diam, _, transform_diam = read_and_mask_raster(diam_path, square_polygon)
# raster_hoyde, _, transform_hoyde = read_and_mask_raster(hoyde_path, square_polygon)
# raster_basal, _, transform_basal = read_and_mask_raster(basal_path, square_polygon)
# raster_tree_type, _, transform_tree_type = read_and_mask_raster(tree_type_path, square_polygon)

# # Convert tree height from dm to meters
# raster_hoyde = raster_hoyde / 10.0

# # Function to calculate area of bins by tree type
# def calculate_area_of_bins_by_type(raster, raster_type, transform, bins, valid_types):
#     pixel_size_x = transform[0]
#     pixel_size_y = -transform[4]  # Typically negative, so take absolute value
#     pixel_area = pixel_size_x * pixel_size_y

#     areas_by_type = {tree_type: np.zeros(len(bins) - 1) for tree_type in valid_types}

#     for tree_type in valid_types:
#         mask = (raster_type == tree_type)
#         hist, bin_edges = np.histogram(raster[mask], bins=bins)
#         areas_by_type[tree_type] = hist * pixel_area / 1_000_000  # Convert to square kilometers

#     return areas_by_type, bin_edges

# # Define valid tree types and bins for diameters, heights, and basal areas
# valid_tree_types = {1: "Spruce", 2: "Pine", 3: "Deciduous"}
# tree_type_colors = {1: "blue", 2: "green", 3: "orange"}
# diam_bins = [0, 10, 20, 30, 40]
# hoyde_bins = [0, 5, 10, 15, 20, 25, 30]
# basal_bins = [0, 10, 20, 30, 40, 50, 60]

# # Calculate areas for each raster by tree type
# diam_areas_by_type, diam_edges = calculate_area_of_bins_by_type(raster_diam, raster_tree_type, transform_diam, diam_bins, valid_tree_types.keys())
# hoyde_areas_by_type, hoyde_edges = calculate_area_of_bins_by_type(raster_hoyde, raster_tree_type, transform_hoyde, hoyde_bins, valid_tree_types.keys())
# basal_areas_by_type, basal_edges = calculate_area_of_bins_by_type(raster_basal, raster_tree_type, transform_basal, basal_bins, valid_tree_types.keys())

# # Function to plot the areas by tree type
# def plot_area_distribution_by_type(bin_edges, areas_by_type, title, xlabel, colors):
#     bin_centers = bin_edges[1:]
#     width = 1 #/ (len(areas_by_type))  # Adjust width for multiple bars
#     fig, ax = plt.subplots(figsize=(10, 6))

#     for i, (tree_type, areas) in enumerate(areas_by_type.items()):
#         ax.bar(bin_centers + (i - 1) * width, areas, width=width, label=valid_tree_types[tree_type], color=colors[tree_type], edgecolor='black')

#     ax.set_xticks(bin_centers)
#     ax.set_xticklabels([str(int(x)) for x in bin_centers], fontsize = 12)
#     ax.set_title(title, fontsize = 14)
#     ax.set_xlabel(xlabel, fontsize = 12)
#     ax.set_ylabel('Area (sq km)', fontsize = 12)
#     ax.legend(fontsize = 12)
#     plt.grid(alpha=0.5)
#     plt.show()

# # Plot area distributions
# plot_area_distribution_by_type(diam_edges, diam_areas_by_type, 'Tree Diameter Distribution', 'Diameter (cm)', tree_type_colors)
# plot_area_distribution_by_type(hoyde_edges, hoyde_areas_by_type, 'Tree Height Distribution', 'Height (m)', tree_type_colors)
# plot_area_distribution_by_type(basal_edges, basal_areas_by_type, 'Basal Area Distribution', 'Basal Area (sq m/ha)', tree_type_colors)

#%%




























