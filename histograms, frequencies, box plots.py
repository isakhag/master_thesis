# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:20:49 2024

@author: Isak9
"""
#%%libraries
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

#%%Biomass histogram from biomasse script! values are from excel sheets from biomasse script
# bin_labels = ['No trees', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70']
# num_landslides = [70,31,25,48,37,11,7,1] #fra total_points_aspect excel file, basert på koden ovenfor.
# slope_frequencies = [0.4, 2.3, 1.2, 2.9, 3.9, 2.5, 3.1, 1]

# # Create figure and axes
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Bar plot for number of landslides
# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Mean Underground Biomass [t/ha]', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue', labelsize = 14)

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red', labelsize = 14)

# ax1.legend(loc='upper left', bbox_to_anchor=(0.15, 1), fontsize=12)

# ax2.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0.7, 1))
# # Increase font size of tick labels on both axes
# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=14)


# plt.title('Landslides and Landslide Frequencies of Underground Biomass', fontsize=16)
# ax1.set_xticklabels(bin_labels, fontsize=14) 
# plt.show()
#%%Tree diameter, values are from excel sheets
# bin_labels = ['No trees', '0-10', '10-20', '20-30', '30-40']
# num_landslides = [69, 4, 108, 48, 1] #fra total_points_aspect excel file, basert på koden ovenfor.
# slope_frequencies = [0.4, 2.4, 2.8, 1.7, 2.8]

# # # Create figure and axes
# fig, ax1 = plt.subplots(figsize=(10, 6))

# #Bar plot for number of landslides
# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Mean Tree Diameter [cm]', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue', labelsize = 14)

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red', labelsize = 14)

# ax1.legend(loc='upper left', fontsize=14)
# ax2.legend(loc='upper left', fontsize=14, bbox_to_anchor=(0.6, 1))

# # Increase font size of tick labels on both axes
# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=14)

# plt.title('Landslides and Landslide Frequencies of Tree Diameter', fontsize=16)
# ax1.set_xticklabels(bin_labels, fontsize=14) 
# plt.show()

#%% tree height
# bin_labels = ['No trees', '0-50', '50-100', '100-150', '150-200', '200-250']
# num_landslides = [69, 9, 38, 64, 44, 6] #fra total_points_aspect excel file, basert på koden ovenfor.
# slope_frequencies = [0.4, 2.8, 1.9, 2.2, 3.3, 3.1]

# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Mean Tree Height [dm]', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue', labelsize=14)

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red', labelsize=14)

# ax1.legend(loc='upper left', bbox_to_anchor=(0.2, 1), fontsize=12)

# ax2.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0.52, 1))
# # Increase font size of tick labels on both axes
# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=14)

# plt.title('Landslides and Landslide Frequencies of Tree Height', fontsize=16)
# plt.show()

#%%Grunnflate
# bin_labels = ['No trees', '0-10', '10-20', '20-30', '30-40', '40-50']
# num_landslides = [73, 27, 44, 61, 18, 7] #fra total_points_aspect excel file, basert på koden ovenfor.
# slope_frequencies = [0.5, 1.9, 1.9, 3.5, 1.9, 2]

# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Mean Basal Area [m2/ha]', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue')

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red')

# ax1.legend(loc='upper left', bbox_to_anchor=(0.2, 1), fontsize=12)

# ax2.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0.6, 1))

# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=14)

# plt.title('Landslides and Landslide Frequencies of Basal Area', fontsize=16)
# plt.show()

#%%Treantall
# bin_labels = ['No trees', '0-400', '400-800', '800-1200', '1200-1600', '1600-2000', '2000-2400', '2400-2800']
# num_landslides = [69, 13, 30, 26, 30, 34, 21, 7] #fra total_points_aspect excel file, basert på koden ovenfor.
# slope_frequencies = [0.4, 1.4, 1.9, 1.8, 2.2, 3.5, 4.6, 7.4]

# # Create figure and axes
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Bar plot for number of landslides
# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Mean amount of trees [trees/ha]', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue', labelsize=14)

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red', labelsize=14)

# ax1.legend(loc='upper left', bbox_to_anchor=(0.2, 1), fontsize=12)

# ax2.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0.5, 1))
# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=14)
# ax1.tick_params(axis='x', rotation=45)

# plt.title('Landslides and Landslide Frequencies at Amount of Trees with Diameter > 5cm at Breast Height', fontsize=16)
# plt.tight_layout()
# plt.show()

#%%treantall med diameter i brysthøyde (1.3m) > 16cm
# bin_labels = ['No trees', '0-100', '100-200', '200-300', '300-400', '400-500', '500-600', '600-700', '700-800', '800-900']
# num_landslides = [66, 25, 21, 29, 35, 19, 14, 10, 6, 5] #fra total_points_aspect excel file, basert på koden ovenfor.
# slope_frequencies = [0.4, 1.7, 1.6, 2.4, 3.7, 2.6, 2.7, 2.9, 2.9, 6]

# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Mean amount of trees [trees/ha]', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue', labelsize=14)

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red', labelsize=14)

# ax1.legend(loc='upper left', bbox_to_anchor=(0.2, 1), fontsize=12)

# ax2.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0.5, 1))
# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=14)
# ax1.tick_params(axis='x', rotation=45)


# plt.title('Landslides and Landslide Frequencies at Amount of Trees With Diameter > 16cm at Breast Height', fontsize=16)
# ax1.set_xticklabels(bin_labels, fontsize=14) 
# plt.tight_layout()
# plt.show()

#%% kronedekning

# bin_labels = ['No trees', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
# num_landslides = [70, 8, 4, 3, 8, 2, 8, 10, 9, 24, 84] #fra total_points_aspect excel file, basert på koden ovenfor.
# slope_frequencies = [0.4, 1.8, 2, 1.1, 2.9, 0.6, 1.7, 2, 1.2, 2.6, 3.2]

# # Create figure and axes
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Bar plot for number of landslides
# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Leaves coverage [%]', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue')

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red')

# ax1.legend(loc='upper left', bbox_to_anchor=(0.2, 1), fontsize=12)

# ax2.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0.5, 1))
# # Increase font size of tick labels on both axes
# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=14)
# # Rotate x-axis tick labels by 45 degrees
# ax1.tick_params(axis='x', rotation=45)


# plt.title('Landslides and Landslide Frequencies of Leaves Coverage', fontsize=16)
# plt.tight_layout()
# plt.show()

#%% soil type, getting info from excel sheet from last year!

# bin_labels = ['Till, thick', 'Till, thin', 'Bedrock', 'WM', 'Glacifluvial', 'Fluvial', 'MMD']
# num_landslides = [113, 54, 38, 18, 2, 2, 3] 
# slope_frequencies = [1.2, 0.7, 0.9, 5.3, 0.3, 0.4, 1.5]

# # Create figure and axes
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Bar plot for number of landslides
# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Soil', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue')

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red')

# ax1.legend(loc='upper left', bbox_to_anchor=(0.15, 1), fontsize=12)

# ax2.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0.6, 1))
# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=14)
# ax1.tick_params(axis='x', rotation=45)


# plt.title('Landslides and Landslide Frequencies of Different Soil', fontsize=16)
# plt.tight_layout()
# plt.show()

#%% flow accumulation stacked bar histogram plot
# excel_file_path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/mergeddf_master.xlsx"
# df = pd.read_excel(excel_file_path)

# bins = [0, 0.0001, 0.001, 0.01, 0.1, 1]
# labels = ['0-0.0001', '0.0001-0.001', '0.001-0.01', '0.01-0.1', '0.1-1']
# df['Flow accumulation area bin'] = pd.cut(df['Flow accumulation area km2'], bins=bins, labels=labels, include_lowest=True)
# flow_df = df.groupby(['Flow accumulation area bin', 'Skredtype']).size().unstack(fill_value=0)

# fig, ax1 = plt.subplots(figsize=(10, 6))
# flow_df.plot(kind='bar', stacked=True, ax=ax1)

# ax2 = ax1.twinx()
# frequencies = [0.65, 0.9, 1.2, 1.7, 2.3]
# ax2.plot(labels, frequencies, color='brown', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='brown', fontsize=16)
# ax2.tick_params('y', colors='brown')

# plt.title('Flow Accumulation Area and Landslide', fontsize=16)
# ax1.set_xlabel('Flow Accumulation Area (Km\u00B2)', fontsize=16)
# ax1.set_ylabel('Number of Landslides', fontsize=16)
# ax1.tick_params(axis='x', rotation=45, labelsize=14)
# ax1.tick_params(axis='y', labelsize=14)
# ax1.legend(title='Landslide types', loc='upper left', fontsize=12, title_fontsize=16)
# ax2.legend(loc='upper left', bbox_to_anchor=(0.7, 1), fontsize=12)
# plt.tight_layout()
# plt.grid(axis='y')
# plt.show()

#%% Histogram of number of landslides based on aspect and frequency 

# bin_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
# num_landslides = [56, 24, 20, 26, 37, 43, 11, 13] #fra total_points_aspect excel file, basert på koden ovenfor.
# slope_frequencies = [1.6, 0.72, 0.85, 1.03, 1.08, 1.4, 0.5, 0.54]

# # Create figure and axes
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Bar plot for number of landslides
# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Slope aspect', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue', labelsize = 14)

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red', labelsize = 14)

# ax1.legend(loc='upper center', fontsize=12)
# ax2.legend(loc='upper right', fontsize=12)

# plt.title('Landslides and Aspect Frequencies', fontsize=16)
# ax1.set_xticklabels(bin_labels, fontsize=14) 
# plt.show()

#%% Plotting histogrammer tretype og frekvens!

# bin_labels = ['No trees', 'Spruce', 'Pine', 'Decidous']
# num_landslides = [63, 17, 35, 115] 
# slope_frequencies = [0.39, 1.45, 1.2, 4.34]

# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Tree type', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue', labelsize = 14)

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red', labelsize = 14)

# ax1.legend(loc='upper left', fontsize=14)
# ax2.legend(loc='upper center', fontsize=14)

# plt.title('Landslides and Landslide Frequencies of Tree Types', fontsize=16)
# ax1.set_xticklabels(bin_labels, fontsize=14) 
# plt.show()

#%% slope angle and landslides histogram

# bin_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60']
# num_landslides = [2, 20, 84, 94, 26, 4]
# slope_frequencies = [0.02, 0.25, 2.34, 5.94, 5.6, 3.15]

# # Create figure and axes
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Bar plot for number of landslides
# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('Slope angle [°]', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue', labelsize = 14)

# ax2 = ax1.twinx()
# ax2.plot(bin_labels, slope_frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red', labelsize = 14)

# ax1.legend(loc='upper left', fontsize=12)
# ax2.legend(loc='upper right', fontsize=12)

# plt.title('Landslides and Slope Frequencies', fontsize=16)
# ax1.set_xticklabels(bin_labels, fontsize=14) 
# plt.show()

#%% box plot forest landslide and open area landslide and its inclination
# df = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/mergeddf_master.xlsx')
# forest_df = df[['Slope angle', 'Treetype']]

# forest_df['Treetype'] = forest_df['Treetype'].apply(lambda x: 'Forest' if x in [1, 2, 3] else 'No Forest')

# plt.figure(figsize=(10, 6))

# plt.boxplot([
#     forest_df[forest_df['Treetype'] == 'Forest']['Slope angle'],
#     forest_df[forest_df['Treetype'] == 'No Forest']['Slope angle']
# ], labels=['Forest', 'No Forest'], boxprops=dict(color='blue'))

# plt.title('Box plot of Slope Angle Range in Forest/No Forest', fontsize = 14)
# plt.ylabel('Slope angle [°]', fontsize = 14)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.tight_layout()
# plt.grid()
# plt.show()








