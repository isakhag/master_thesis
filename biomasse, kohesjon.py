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

#%% kohesjonskart!!
input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NVE_NGIHVL.shp"
input_raster = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRBMU.tif"
input_raster2 = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTRESLAG.tif"

square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
square_polygon = Polygon(square_coords)
pts = gpd.read_file(input_shapefile)  
pts = pts[pts.geometry.within(square_polygon)]

shapefile = gpd.read_file(input_shapefile)
square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
square_polygon = Polygon(square_coords)

with rasterio.open(input_raster) as src1:
    raster1, raster1_transform = mask(src1, [square_polygon], crop=True)
    raster1 = raster1[0]

with rasterio.open(input_raster2) as src2:
    raster2, raster2_transform = mask(src2, [square_polygon], crop=True)
    raster2 = raster2[0] 
    
raster1 = raster1.astype(float)
raster2 = raster2.astype(float)    
raster1[raster1 == -9999] = np.nan
raster2[raster2 == -9999] = np.nan

cohesion_gran = 1.2 * 90 * (raster1[raster2 == 1] / 451)
cohesion_furu = 1.2 * 104 * (raster1[raster2 == 2] / 539)
cohesion_lauv = 1.2 * 111.6 * (raster1[raster2 == 3] / 671)

cohesion_values = np.zeros_like(raster2, dtype=float)
cohesion_values[raster2 == 1] = cohesion_gran
cohesion_values[raster2 == 2] = cohesion_furu
cohesion_values[raster2 == 3] = cohesion_lauv

#%% sending the cohesion_values to a raster file
# output_raster_path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ml/cohesion_raster.tif"

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
#%%plotting  cohesion map
# x_min, y_min = min(square_polygon.exterior.xy[0]), min(square_polygon.exterior.xy[1])
# x_max, y_max = max(square_polygon.exterior.xy[0]), max(square_polygon.exterior.xy[1])

# raster1_extent = [x_min, x_max, y_min, y_max]

# bins = [0,0.000001, 2.5, 5, 10, 15, 20, 25, np.inf]  # np.inf to cover all higher values

# hist, edges = np.histogram(cohesion_values, bins=bins)
# pixel_area = 16 * 16
# areas_km2 = (hist * pixel_area) / 1e6
# total_area_km2 = np.sum(areas_km2)

# # Output the areas for each bin
# for i in range(len(bins)-1):
#     if i == 0:
#         print(f"Area where cohesion is exactly 0: {areas_km2[i]} square kilometers")
#     else:
#         print(f"Area for bin {bins[i]} to {bins[i+1]}: {areas_km2[i]} square kilometers")

# # Print the total area
# print(f"Total area: {total_area_km2} square kilometers")

# plt.figure(figsize=(12, 10))  
# image = plt.imshow(cohesion_values, cmap='viridis', extent=raster1_extent)
# pts.plot(ax=plt.gca(), color='red', markersize=5)

# plt.scatter(154200, 6884400, color='red', marker='o', s=150)

# plt.text(154200 + 2000, 6884400-500, 'Landslide Points', color='red', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# point_x = 189847
# point_y = 6874417
# plt.scatter(point_x, point_y, color='red', marker='*', s=200)  # Plot the point with a star marker
# plt.annotate('Vågåmo', (point_x+2000, point_y+1500), color='red', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# cbar = plt.colorbar(image, label='Cohesion [kPa]', shrink=0.5)
# cbar.ax.tick_params(labelsize=12)  

# plt.title('Cohesion Map', fontsize=12)
# plt.xlabel('East [m]', fontsize=12)
# plt.ylabel('North [m]', fontsize=12)

# plt.xticks(fontsize=12) 
# plt.yticks(fontsize=12)  
# plt.tight_layout()
# plt.show()

#%% kohesjonsverdier i ulike landslide points
# excel_file_path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/mergeddf_master.xlsx"
# df = pd.read_excel(excel_file_path)
# cohesion = df[['Treetype', 'sr16_34_SRRBMU']].copy()

# def calculate_cohesion(row):
#     if row['Treetype'] == 1:
#         return 1.2 * 90 * (row['sr16_34_SRRBMU'] / 451)
#     elif row['Treetype'] == 2:
#         return 1.2 * 104 * (row['sr16_34_SRRBMU'] / 539)
#     elif row['Treetype'] == 3:
#         return 1.2 * 111.6 * (row['sr16_34_SRRBMU'] / 671)
#     else:
#         return -9999
# cohesion['calculated_cohesion'] = cohesion.apply(calculate_cohesion, axis=1)

# conditions = [
#     cohesion['calculated_cohesion'] == -9999,  # Special case for 'no trees'
#     (cohesion['calculated_cohesion'] >= 0) & (cohesion['calculated_cohesion'] <= 2.5),
#     (cohesion['calculated_cohesion'] > 2.5) & (cohesion['calculated_cohesion'] <= 5),
#     (cohesion['calculated_cohesion'] > 5) & (cohesion['calculated_cohesion'] <= 10),
#     (cohesion['calculated_cohesion'] > 10) & (cohesion['calculated_cohesion'] <= 15),
#     (cohesion['calculated_cohesion'] > 15) & (cohesion['calculated_cohesion'] <= 20),
#     (cohesion['calculated_cohesion'] > 20) & (cohesion['calculated_cohesion'] <= 25),

# ]
# values = ['no trees', '0-2,5','2,5-5', '5-10', '10-15', '15-20', '20-25']
# cohesion['cohesion_category'] = np.select(conditions, values, default='>30')
# category_counts = cohesion['cohesion_category'].value_counts().sort_index()
# print(category_counts)
 
# bin_labels = ['No trees', '0-2.5','2.5-5', '5-10', '10-15', '15-20']
# num_landslides = [63, 41, 46, 65, 14, 1] 
# slope_frequencies = [0.35,2.6, 2, 3, 2.6, 1.1 ]

# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Root Cohesion [kPa]', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue', labelsize = 14)

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red', labelsize = 14)

# ax1.legend(loc='upper left', fontsize=12)
# ax2.legend(loc='upper right', fontsize=12)

# plt.title('Landslides and Landslide Frequencies of Root Cohesion', fontsize=16)
# ax1.set_xticklabels(bin_labels, fontsize=14) 
# plt.show()


#%% getting information about the different raster files! creating histograms and getting a good overview
# of the release points and the forest details

# # Define input file paths
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NGI_HVL.shp"
# input_rasters = [
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRBMU.tif",
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTRESLAG.tif",
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRDIAMMIDDEL_GE8.tif",
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRMHOYDE.tif",
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRGRFLATE.tif",
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTREANTALL.tif",
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTREANTALL_GE16.tif", 
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRKRONEDEK.tif"
# ]

# # List to store the data
# data = []

# for input_raster in input_rasters:
#     with rasterio.open(input_raster) as src:
#         assert src.transform, "Raster dataset must have a transformation."
#         # Read shapefile
#         pts = gpd.read_file(input_shapefile)

#         # Define square polygon
#         square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
#         square_polygon = Polygon(square_coords)
#         square = gpd.GeoDataFrame(geometry=[square_polygon], crs='EPSG:32633') 
#         # Filter points within square
#         pts_within_square = pts[pts.geometry.within(square.geometry.iloc[0])]

#         # Sample raster values for each point
#         points_values = list(src.sample(zip(pts_within_square.geometry.x, pts_within_square.geometry.y), indexes=1))

#         # Extract column name from raster filename
#         column_name = input_raster.split('/')[-1].split('.')[0]

#         # Extract values for each point and store in a list
#         for point, value_array in zip(pts_within_square.geometry, points_values):
#             for value in value_array:
#                 if not value.item() is None:
#                     data.append({'Point': point, column_name: value.item()})

# # Create DataFrame from the collected data
# df = pd.DataFrame(data)
# output_excel_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/biomasse/biomasse.xlsx'
# df.to_excel(output_excel_path, index=False)

# from shapely.geometry import Polygon, mapping

# # Read the shapefile containing the points
# points_gdf = gpd.read_file(input_shapefile)

# # Plot the points
# ax = points_gdf.plot(marker='o', color='red', markersize=5, figsize=(10, 10))

# # Read the first raster
# first_raster = input_rasters[0]

# # Open the raster file
# with rasterio.open(first_raster) as src:
#     # Mask the raster data within the square polygon
#     masked_data, masked_transform = mask(src, [mapping(square_polygon)], crop=True)

#     # Plot the masked raster data
#     show(masked_data, transform=masked_transform, ax=ax)

# # Add title and labels
# plt.title('Raster Data and Points Inside Square')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')

# # Show the plot
# plt.show()

# forest_values = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/values_forest_parameters.xlsx'
# df = pd.read_excel(forest_values)

# columns_of_interest = ['sr16_34_SRRBMU', 'sr16_34_SRRDIAMMIDDEL_GE8', 'sr16_34_SRRMHOYDE', 
#                        'sr16_34_SRRGRFLATE', 'sr16_34_SRRTREANTALL', 'sr16_34_SRRTREANTALL_GE16', 
#                        'sr16_34_SRRKRONEDEK']

# bins = {
#     'sr16_34_SRRBMU': [-10000, -1, 0, 10, 20, 30, 40, 50, 60, 70, 80],  # tonnes/hektar
#     'sr16_34_SRRDIAMMIDDEL_GE8': [-10000, -1, 0, 10, 20, 30, 40, 50],  # cm
#     'sr16_34_SRRMHOYDE': [-10000, -1, 0, 50, 100, 150, 200, 250, 300],  # Dm
#     'sr16_34_SRRGRFLATE': [-10000, -1, 0, 10, 20, 30, 40, 50, 60, 70],  # m2/hektar
#     'sr16_34_SRRTREANTALL': [-10000, -1, 0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200],  # tree/hektar
#     'sr16_34_SRRTREANTALL_GE16': [-10000, -1, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900,1000],  # tree/hektar
#     'sr16_34_SRRKRONEDEK': [-10000, -1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # %
# }

# counts_data = []

# for column in columns_of_interest:
#     bin_edges = bins[column]
#     bin_labels = [f'{bin_edges[i]}-{bin_edges[i+1]}' for i in range(len(bin_edges) - 1)]
#     bins_count = pd.cut(df[column], bins=bin_edges).value_counts().sort_index()
#     for bin_range, count in bins_count.items():
#         counts_data.append({'Column': column, 'Bin Edges': bin_range, 'Count': count})

# counts_df = pd.DataFrame(counts_data)
# output_excel_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/counts.xlsx'
# counts_df.to_excel(output_excel_path)
#%% plotting the soil data

# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NVE_NGIHVL.shp"
# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/losmasse/losmasse_flate.shp'
# input_raster2 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/Losmasse/LosmasseGrense_20231125.shp'

# pts = gpd.read_file(input_shapefile)
# soil = gpd.read_file(input_raster)
# grense = gpd.read_file(input_raster2)

# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)
# pts = pts[pts.geometry.within(square_polygon)]

# soil_colors = {
#     'Randmorene/randmorenesone': (0, 255, 51),
#     'Skredmateriale, sammenhengende dekke': (255,97,97),  
#     'Elve- og bekkeavsetning (Fluvial avsetning)':(255,237,97), 
#     'Torv og myr':(196,148,126), 
#     'Ryggformet breelvavsetning (Esker)': (255,189,0),
#     'Bart fjell':(255,240,240), 
#     'Morenemateriale, sammenhengende dekke, stedvis med stor mektighet':(135,255,97),
#     'Breelvavsetning (Glasifluvial avsetning)':(255,171,0), 
#     'Morenemateriale, usammenhengende eller tynt dekke over berggrunnen':(191,255,135), 
#     'Forvitringsmateriale, usammenhengende eller tynt dekke over berggrunnen':(239,197,250), 
#     'Skredmateriale, usammenhengende eller tynt dekke':(255,97,97), 
#     'Forvitringsmateriale':(224,181,235), 
#     'Tynt dekke av organisk materiale over berggrunn':(232,219,196),
#     'Forvitringsmateriale, stein- og blokkrikt (blokkhav)':(212,159,212), 
#     'Fyllmasse (antropogent materiale)':(174,174,174),
#     'Bresjø- eller brekammeravsetning (Glasilakustrin avsetning)':(255,247,135),
#     'Forvitringsmateriale, ikke inndelt etter mektighet': (232,194,255),
#     'Avsmeltningsmorene (Ablasjonsmorene)': (107,199,130),
#     'Jordskred- og steinsprangavsetning, stedvis med stor mektighet':(224,158,176),
#     'Steinsprangavsetning, stedvis med stor mektighet':(255,153,153),
#     'Innsjøavsetning (Lakustrin avsetning)':(255,255,190 ),
#     'Steinsprangavsetning, usammenhengende eller tynt dekke':(255,153,153),
#     'Ikke angitt':(255,255,255),
#     'Vindavsetning (Eolisk avsetning)':(212,217,66),
#     'Fjellskredavsetning, stedvis med stor mektighet':(255,181,181),
#     'Jord- og flomskredavsetning, usammenhengende eller tynt dekke':(255,77,77),
#     'Jord- og flomskredavsetning':(255,77,77),
#     'Flomavsetning':(255, 214,41)
# }

# def rgb_to_rgba(rgb_tuple, alpha):
#     r, g, b = [x / 255.0 for x in rgb_tuple]
#     return (r, g, b, alpha)

# def get_soil_color(row):
#     return soil_colors.get(row['jorda_navn'], (1, 1, 1))

# soil['color'] = soil.apply(get_soil_color, axis=1)
# soil['color'] = soil['color'].apply(lambda x: rgb_to_rgba(x, alpha=0.7))

# fig, ax = plt.subplots(figsize=(10, 10))
# soil.plot(ax=ax, color=soil['color'], legend=True)
# pts.plot(ax=ax, color='blue', markersize=10, label='Landslide Points')
# point_x = 189847
# point_y = 6874417
# ax.scatter(point_x, point_y, color='blue', marker='*', s=200)  # Plot the point with a star marker
# ax.annotate('Vågåmo', (point_x, point_y), textcoords="offset points", xytext=(10,10), ha='center')
# gpd.GeoSeries([square_polygon]).plot(ax=ax, color='none', edgecolor='black', linewidth=2)
# grense.plot(ax=ax, color='blue', linewidth=0.3, alpha = 0.3)
# ax.set_xlabel('East [m]', fontsize=12)
# ax.set_ylabel('North [m]', fontsize=12)
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.set_title('Soil Type and Landslide points', fontsize=12)
# plt.legend(fontsize=12)
# plt.show() 

# soil_colors = {
#     15 : (0, 255, 51),
#     81: (255,97,97),  
#     50:(255,237,97), 
#     90:(196,148,126), 
#     22: (255,189,0),
#     130:(255,240,240), 
#     11:(135,255,97),
#     20:(255,171,0), 
#     12:(191,255,135), 
#     72:(239,197,250), 
#     82 :(255,97,97), 
#     71:(224,181,235), 
#     100 :(232,219,196),
#     73:(212,159,212), 
#     120:(174,174,174),
#     36 :(255,247,135),
#     70: (232,194,255),
#     14 : (107,199,130),
#     315:(224,158,176),
#     307:(255,153,153),
#     35 :(255,255,190 ),
#     308 :(255,153,153),
#    # 'Ikke angitt':(255,255,255),
#     60 :(212,217,66),
#     305 :(255,181,181),
#     302 :(255,77,77),
#     301 :(255,77,77),
#     56 :(255, 214,41)
# }
#%%



