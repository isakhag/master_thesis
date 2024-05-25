# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:30:05 2023

@author: Isak9
"""


## Import packages
import numpy as np
import pandas as pd
#import rasterio
#from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from shapely.geometry import box
#import seaborn as sns
import rasterio
from rasterio.mask import mask
from rasterio.plot import show

#%%input data

# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/points2.shp"
# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/Slope_dtm10_6_1.tif'
# output_csv = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/output.csv"


# print("reading shp of inventory points")
# pts = gpd.read_file(input_shapefile) ## Read .shp of poinys
# npnts = len(pts)
# coords = [(x,y) for x, y in zip(pts.geometry.x, pts.geometry.y)] #coordinates of the points

# ## read rasters

# with rasterio.open(input_raster) as src_dataset:
#     print("Reading data: {}".format(input_raster))
#     kwds= src_dataset.profile # other info about raster. 
#     features_in = src_dataset.read(1, masked = True).astype(np.int64).filled(0) # Array with raster values
#     bbox = src_dataset.bounds # Bounding box (extent of the raster file. Isak you might need to filter points outside of here)

#     pts['slope'] = [x for x in src_dataset.sample(coords)] ## Assigns the sampled values to the 'slope' column of the pts DataFrame.
#     pts['slope'] = pts.apply(lambda x: x[['slope']][0], axis=1) ## Extract the first element from each value in the 'slope' column and overwrite the 'slope' column with these extracted values.
# pts.to_csv((output_csv), index=False)

# sampled_values = [x for x in src_dataset.sample(coords)]
# print("Sampled Values:", sampled_values)

# for coord, value in zip(coords, sampled_values):
#     print(f"Coordinate: {coord}, Sampled Value: {value}")

# print("Raster Statistics:")
# print("Min Slope:", np.min(features_in))
# print("Max Slope:", np.max(features_in))

# single_coord = [(140500, 6886700)]  # Replace with a valid coordinate
# single_value = src_dataset.sample(single_coord)
# print("Sampled Value at Single Coordinate:", single_value)
#%%POINTS SWECO(hvl) INSIDE STUDY AREA
# input data
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/points2.shp"
# input_raster1 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/Slope_dtm10_6_1.tif'
# input_raster2 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/Slope_dtm10_5.tif'
# input_raster3 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/Slope_dtm10_4.tif'
# output_csv = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/output.csv"

# Bounding box coordinates
#square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]

#print("reading shp of inventory points")
#pts = gpd.read_file(input_shapefile)  # Read .shp of points

# Create a polygon representing the bounding box
#bounding_box = Polygon(square_coords)

# # Filter points within the bounding box
# pts = pts[pts.geometry.within(bounding_box)]

# npnts = len(pts)
# coords = [(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]  # coordinates of the points

# # read rasters and assign slope values for each raster
# def assign_slope(pts, input_raster, slope_col_name):
#     with rasterio.open(input_raster) as src_dataset:
#         print("Reading data: {}".format(input_raster))

#         pts[slope_col_name] = [x for x in src_dataset.sample(coords)]
#         pts[slope_col_name] = pts.apply(lambda x: x[[slope_col_name]][0], axis=1)

# # Assign slope values for each raster
# assign_slope(pts, input_raster1, 'slope1')
# assign_slope(pts, input_raster2, 'slope2')
# assign_slope(pts, input_raster3, 'slope3')

# # Save to CSV
# # pts.to_csv(output_csv, index=False)

# bin_edges = np.arange(0, 90, 10)  # You can adjust the bin edges as needed

# # Define a function to count values in each bin
# def count_values_in_bins(slope_values):
#     return pd.cut(slope_values, bins=bin_edges).value_counts().sort_index()

# # Apply the function to each slope column
# count_df = pd.DataFrame({
#     'slope1': count_values_in_bins(pts['slope1']),
#     'slope2': count_values_in_bins(pts['slope2']),
#     'slope3': count_values_in_bins(pts['slope3'])
# })

# count_df['Total landslides'] = count_df.sum(axis=1)

# count_df.index.name = 'Slope angle (degrees)'

# # Keep only the 'total' column and the 'slope angle' column
# count_df = count_df[['Total landslides']]

# # Save the count DataFrame to CSV
# #count_df.to_csv("C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/slope_counts.csv", index=True)

# lower_limit = 10

# # Filter points within the specified slope angle range
# filtered_points = pts[(pts['slope1'] <= lower_limit)]

# # Display the filtered points
# print(filtered_points)
#%%POINTS NGI SLOPE
# input data
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/Ottadalen-NGI/Release point Ottadalen.shp"
# input_raster1 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/slope/Slope_dtm10_6_1.tif'
# input_raster2 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/slope/Slope_dtm10_5.tif'
# input_raster3 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/slope/Slope_dtm10_4.tif'

# output_csv_ngi = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/slope_counts_ngi.csv"

# # Bounding box coordinates
# square_coords = [(140438, 6886745), (140438, 6850000), (218910, 6850000), (218910, 6886745)]

# print("reading shp of inventory points")
# pts_ngi = gpd.read_file(input_shapefile)  # Read .shp of points

# # Create a polygon representing the bounding box
# bounding_box = Polygon(square_coords)

# # Filter points within the bounding box
# pts_ngi = pts_ngi[pts_ngi.geometry.within(bounding_box)]

# npnts = len(pts_ngi)
# coords = [(x, y) for x, y in zip(pts_ngi.geometry.x, pts_ngi.geometry.y)]  # coordinates of the points

# # read rasters and assign slope values for each raster
# def assign_slope(pts, input_raster, slope_col_name):
#     with rasterio.open(input_raster) as src_dataset:
#         print("Reading data: {}".format(input_raster))

#         pts[slope_col_name] = [x[0] for x in src_dataset.sample(coords)]
#         pts[slope_col_name] = pts.apply(lambda x: x[[slope_col_name]][0], axis=1)

# # Assign slope values for each raster
# assign_slope(pts_ngi, input_raster1, 'slope1')
# assign_slope(pts_ngi, input_raster2, 'slope2')
# assign_slope(pts_ngi, input_raster3, 'slope3')

# # Save the GeoDataFrame to CSV
# #pts_ngi.to_csv(output_csv_ngi, index=False)
# # Define bin edges
# bin_edges = np.arange(0, 90, 10)  # You can adjust the bin edges as needed

# # Define a function to count values in each bin
# def count_values_in_bins(slope_values):
#     return pd.cut(slope_values, bins=bin_edges).value_counts().sort_index()

# # Apply the function to each slope column
# count_df_ngi = pd.DataFrame({
#     'slope1': count_values_in_bins(pts_ngi['slope1']),
#     'slope2': count_values_in_bins(pts_ngi['slope2']),
#     'slope3': count_values_in_bins(pts_ngi['slope3'])
# })

# # Sum the counts across columns to get the total landslides
# count_df_ngi['Total landslides'] = count_df_ngi.sum(axis=1)

# # Rename the index name to 'Slope angle (degrees)'
# count_df_ngi.index.name = 'Slope angle (degrees)'

# # Keep only the 'Total landslides' column and the 'Slope angle (degrees)' column
# count_df_ngi = count_df_ngi[['Total landslides']]

# # Save the count DataFrame to CSV
# count_df_ngi.to_csv("C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/slope/slope_counts_ngi.csv", index=True)

#%%RASTER SOIL TYPE NGI

#input data
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Ottadalen-NGI/Release point Ottadalen.shp"
# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Losmasse/LosmasseFlate_20231125.shp'

# output = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Losmasse/soil_type_ngi.csv"

# # Read shapefile data
# shapefile_data = gpd.read_file(input_shapefile)

# # Create a GeoDataFrame with Point geometries from the shapefile_data
# geometry = [Point(xy) for xy in zip(shapefile_data.geometry.x, shapefile_data.geometry.y)]
# points_gdf = gpd.GeoDataFrame(shapefile_data, geometry=geometry, crs=shapefile_data.crs)

# # Read raster data
# raster_data = gpd.read_file(input_raster)

# # Spatial join the two GeoDataFrames based on their geometries
# joined_data = gpd.sjoin(points_gdf, raster_data, how="left", op="within")

# # Extract coordinates and 'jordart' column
# result_data = joined_data[['geometry', 'jorda_navn']]

# # Count occurrences of each 'jordart' and create a new DataFrame
# jordart_counts = result_data['jorda_navn'].value_counts().reset_index()
# jordart_counts.columns = ['jorda_navn', 'jordart_count']

# # Save the result to a CSV file
# jordart_counts.to_csv(output, index=False)

#%% SOIL TYPE SWECO
#Input files
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/points2.shp"
# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Losmasse/LosmasseFlate_20231125.shp'

# # Output CSV file
# output = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Losmasse/soil_type_sweco.csv"

# # Bounding box coordinates
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# bounding_box = Polygon(square_coords)

# # Read shapefile data
# shapefile_data = gpd.read_file(input_shapefile)

# # Filter points within the bounding box
# shapefile_data = shapefile_data[shapefile_data.geometry.within(bounding_box)]

# # Create a GeoDataFrame with Point geometries from the shapefile_data
# geometry = [Point(xy) for xy in zip(shapefile_data.geometry.x, shapefile_data.geometry.y)]
# points_gdf = gpd.GeoDataFrame(shapefile_data, geometry=geometry, crs=shapefile_data.crs)

# # Read raster data
# raster_data = gpd.read_file(input_raster)

# # Spatial join the two GeoDataFrames based on their geometries

# joined_data = gpd.sjoin(points_gdf, raster_data, how="inner", op="within")

# # Extract coordinates and 'jordart' column
# result_data = joined_data[['geometry', 'jorda_navn']]

# # Count occurrences of each 'jordart' and create a new DataFrame
# jordart_counts = result_data['jorda_navn'].value_counts().reset_index()
# jordart_counts.columns = ['jorda_navn', 'jorda_navn_count']

# # Save the result to a CSV file
# jordart_counts.to_csv(output, index=False)

#%% VEGETATION NGI POINTS

# Input file paths
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Ottadalen-NGI/Release point Ottadalen.shp"
# input_dovre = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/vegetation1.shp'
# input_lom = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/lom/lom.shp'
# input_sel = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/sel/sel.shp'
# input_skjåk = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/skjåk/skjåk.shp'
# input_vågå = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/vågå/vågå.shp'

# # Output CSV file
# output_csv = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/vegetation_ngi.csv"

# # Read Ottadalen points
# ottadalen_points = gpd.read_file(input_shapefile)

# lom = gpd.read_file(input_lom)
# vågå = gpd.read_file(input_vågå)
# dovre = gpd.read_file(input_dovre)
# skjåk = gpd.read_file(input_skjåk)
# sel = gpd.read_file(input_sel)

# # Extract coordinates from the ottadalen shapefile
# geometry = [Point(xy) for xy in zip(ottadalen_points.geometry.x, ottadalen_points.geometry.y)]
# ottadalen_points = gpd.GeoDataFrame(ottadalen_points, geometry=geometry, crs=ottadalen_points.crs)

# # Spatial join
# joined_data_lom = gpd.sjoin(ottadalen_points, lom[['geometry', 'artype']], how="inner", op="within")
# joined_data_vågå = gpd.sjoin(ottadalen_points, vågå[['geometry', 'artype']], how="inner", op="within")
# joined_data_dovre = gpd.sjoin(ottadalen_points, dovre[['geometry', 'artype']], how="inner", op="within")
# joined_data_skjåk = gpd.sjoin(ottadalen_points, skjåk[['geometry', 'artype']], how="inner", op="within")
# joined_data_sel = gpd.sjoin(ottadalen_points, sel[['geometry', 'artype']], how="inner", op="within")

# joined_data = pd.concat([joined_data_lom, joined_data_vågå, joined_data_dovre, joined_data_skjåk, joined_data_sel])

# # Count occurrences of each 'artype' and create a new DataFrame
# artype_counts = joined_data['artype'].value_counts().reset_index()
# artype_counts.columns = ['artype', 'artype_count']

# # Create a DataFrame with unique 'artype' values
# unique_artypes = pd.DataFrame({'artype': joined_data['artype'].unique()})

# # Merge the unique_artypes DataFrame with the counts DataFrame
# unique_artypes = pd.merge(unique_artypes, artype_counts, on='artype', how='left')
# unique_artypes.to_csv(output_csv, index=False)

#%% SWECO POINTS vegetation ar50

# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/points2.shp"
# input_dovre = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/vegetation1.shp'
# input_lom = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/lom/lom.shp'
# input_sel = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/sel/sel.shp'
# input_skjåk = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/skjåk/skjåk.shp'
# input_vågå = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/vågå/vågå.shp'

# # Output CSV file
# output_csv = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/vegetation_sweco.csv"

# lom = gpd.read_file(input_lom)
# vågå = gpd.read_file(input_vågå)
# dovre = gpd.read_file(input_dovre)
# skjåk = gpd.read_file(input_skjåk)
# sel = gpd.read_file(input_sel)

# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]

# print("Reading shp of inventory points")
# pts = gpd.read_file(input_shapefile)  # Read .shp of points

# # Create a Polygon representing the bounding box
# bounding_box = Polygon(square_coords)

# # Filter points within the bounding box
# sweco_points = pts[pts.geometry.within(bounding_box)]

# npnts = len(pts)
# coords = [(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]

# joined_data_lom = gpd.sjoin(sweco_points[sweco_points.geometry.within(bounding_box)], lom[['geometry', 'artype']], how="inner", op="within")
# joined_data_vågå = gpd.sjoin(sweco_points[sweco_points.geometry.within(bounding_box)], vågå[['geometry', 'artype']], how="inner", op="within")
# joined_data_dovre = gpd.sjoin(sweco_points[sweco_points.geometry.within(bounding_box)], dovre[['geometry', 'artype']], how="inner", op="within")
# joined_data_skjåk = gpd.sjoin(sweco_points[sweco_points.geometry.within(bounding_box)], skjåk[['geometry', 'artype']], how="inner", op="within")
# joined_data_sel = gpd.sjoin(sweco_points[sweco_points.geometry.within(bounding_box)], sel[['geometry', 'artype']], how="inner", op="within")

# joined_data = pd.concat([joined_data_lom, joined_data_vågå, joined_data_dovre, joined_data_skjåk, joined_data_sel])

# # Count occurrences of each 'artype' and create a new DataFrame
# artype_counts = joined_data['artype'].value_counts().reset_index()
# artype_counts.columns = ['artype', 'artype_count']

# # Create a DataFrame with unique 'artype' values
# unique_artypes = pd.DataFrame({'artype': joined_data['artype'].unique()})

# # Merge the unique_artypes DataFrame with the counts DataFrame
# unique_artypes = pd.merge(unique_artypes, artype_counts, on='artype', how='left')
# unique_artypes.to_csv(output_csv, index=False)

#%% AREA OF THE DIFFERENT SOIL TYPES
# from matplotlib.patches import Patch
# soiltype = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Losmasse/LosmasseFlate_20231125.shp'

# # Read the shapefiles for soil types and the square
# soil_types = gpd.read_file(soiltype)
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]

# # Create a Polygon geometry for the square
# square_polygon = Polygon(square_coords)

# # Convert the square into a GeoDataFrame
# square = gpd.GeoDataFrame(geometry=[square_polygon], crs='EPSG:25833')

# # Reproject the soil types to EPSG:25833
# soil_types = soil_types.to_crs('EPSG:25833')

# # Perform a spatial join to find soil types within the square
# soil_within_square = gpd.overlay(soil_types, square, how="intersection")
# # Calculate the area of each intersection
# soil_within_square['intersection_area'] = soil_within_square['geometry'].area
# soil_within_square['area_km2'] = soil_within_square['intersection_area'] / 1e6
# # Sum the area for each soil type

# total_area_by_soil_type = soil_within_square.groupby('jorda_navn')['area_km2'].sum().reset_index()
# total_area_by_soil_type['cumulative_area_km2'] = total_area_by_soil_type['area_km2'].cumsum()
# # Print the result
# print(total_area_by_soil_type)

# merged_data_soil = soil_within_square.merge(total_area_by_soil_type, on='jorda_navn')

# # Plot the map without a colorbar
# fig, ax = plt.subplots(figsize=(10, 8))
# merged_data_soil.plot(column='area_km2_y', ax=ax, legend=False)
# ax.set_title('Total Area by Soil Type')

# plt.show()

#%% AREA OF VEGETATION TYPES
# Define the input shapefiles
# input_dovre = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/vegetation1.shp'
# input_lom = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/lom/lom.shp'
# input_sel = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/sel/sel.shp'
# input_skjåk = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/skjåk/skjåk.shp'
# input_vågå = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/vågå/vågå.shp'
# input_nord_fron = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/vegetation/ar50/nord_fron/nord_fron.shp'

# # Read the shapefiles
# dovre = gpd.read_file(input_dovre)
# lom = gpd.read_file(input_lom)
# sel = gpd.read_file(input_sel)
# skjåk = gpd.read_file(input_skjåk)
# vågå = gpd.read_file(input_vågå)
# nord_fron = gpd.read_file(input_nord_fron)

# # Your square coordinates
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]

# # Create a GeoDataFrame for the square
# square_polygon = Polygon(square_coords)
# square = gpd.GeoDataFrame(geometry=[square_polygon], crs='EPSG:25833')
# # Reproject the shapefiles to EPSG:25833
# dovre = dovre.to_crs('EPSG:25833')
# lom = lom.to_crs('EPSG:25833')
# sel = sel.to_crs('EPSG:25833')
# skjåk = skjåk.to_crs('EPSG:25833')
# vågå = vågå.to_crs('EPSG:25833')
# nord_fron = nord_fron.to_crs('EPSG:25833')

# # Perform spatial joins
# # joined_data_dovre = gpd.sjoin(dovre, square, how="inner", op="intersects")
# # joined_data_lom = gpd.sjoin(lom, square, how="inner", op="intersects")
# # joined_data_sel = gpd.sjoin(sel, square, how="inner", op="intersects")
# # joined_data_skjåk = gpd.sjoin(skjåk, square, how="inner", op="intersects")
# # joined_data_vågå = gpd.sjoin(vågå, square, how="inner", op="intersects")

# joined_data_dovre = gpd.overlay(dovre, square, how="intersection")
# joined_data_lom = gpd.overlay(lom, square, how="intersection")
# joined_data_sel = gpd.overlay(sel, square, how="intersection")
# joined_data_skjåk = gpd.overlay(skjåk, square, how="intersection")
# joined_data_vågå = gpd.overlay(vågå, square, how="intersection")
# joined_data_nord_fron = gpd.overlay(nord_fron, square, how="intersection")

# # Calculate the area of each intersection
# for df in [joined_data_dovre, joined_data_lom, joined_data_sel, joined_data_skjåk, joined_data_vågå, joined_data_nord_fron]:
#     df['intersection_area'] = df['geometry'].area
#     df['area_km2'] = df['intersection_area'] / 1e6

# # Concatenate the DataFrames
# all_joined_data = pd.concat([joined_data_dovre, joined_data_lom, joined_data_sel, joined_data_skjåk, joined_data_vågå, joined_data_nord_fron])

# # Sum the area for each artype
# total_area_by_artype = all_joined_data.groupby('artype')['area_km2'].sum().reset_index()
# #total_area_by_artype['cumulative_area_km2'] = total_area_by_artype['area_km2'].cumsum()

# # Print the result
# print(total_area_by_artype)

# # ... (your previous code to create total_area_by_artype)

# # Merge the total_area_by_artype data with the geometry of the original GeoDataFrame
# merged_data = all_joined_data.merge(total_area_by_artype, on='artype')

# # Plot the map with colors based on artype
# fig, ax = plt.subplots(figsize=(10, 8))
# merged_data.plot(column='artype', cmap='viridis', legend=True, ax=ax, legend_kwds={'label': "Artype"})
# ax.set_title('Total Area by Artype')

# plt.show()


#%% SLOPE ASPECTS
#input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/points2.shp"
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Ottadalen-NGI/Release point Ottadalen.shp"
# input_aspect1 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Aspect/Aspect_Slope1.tif'
# input_aspect2 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Aspect/Aspect_Slope2.tif'
# input_aspect3 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Aspect/Aspect_Slope3.tif'
# output_csv = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Aspect/output_ngi.csv"

# # Bounding box coordinates
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]

# print("reading shp of inventory points")
# pts = gpd.read_file(input_shapefile)  # Read .shp of points

# # Create a polygon representing the bounding box
# bounding_box = Polygon(square_coords)

# # Filter points within the bounding box
# pts = pts[pts.geometry.within(bounding_box)]

# npnts = len(pts)
# coords = [(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]  # coordinates of the points

# aspect_df = pd.DataFrame()

# # Assign aspect values for each raster
# def assign_aspect(pts, input_aspect, aspect_col_name):
#     with rasterio.open(input_aspect) as src_dataset:
#         print("Reading data: {}".format(input_aspect))

#         pts[aspect_col_name] = [x for x in src_dataset.sample(coords)]
#         pts[aspect_col_name] = pts.apply(lambda x: x[[aspect_col_name]][0], axis=1)

# # Assign aspect values for each raster
# assign_aspect(pts, input_aspect1, 'aspect1')
# assign_aspect(pts, input_aspect2, 'aspect2')
# assign_aspect(pts, input_aspect3, 'aspect3')

# # Concatenate aspect columns into a single DataFrame
# aspect_df = pd.concat([pts['aspect1'], pts['aspect2'], pts['aspect3']], axis=1)
# #aspect_df.to_csv(output_csv, index=False)

# # Define bin edges for aspect values
# aspect_bin_edges = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]

# # Define aspect labels
# aspect_labels = ['0-22.5', '22.5-67.5', '67.5-112.5', '112.5-157.5', '157.5-202.5', '202.5-247.5', '247.5-292.5', '292.5-337.5', '337.5-360']

# # Function to count values in each bin for aspect
# def count_values_in_aspect_bins(aspect_values):
#     aspect_values_wrapped = aspect_values % 360  # Wrap aspect values to [0, 360)
#     return pd.cut(aspect_values_wrapped, bins=aspect_bin_edges, labels=aspect_labels).value_counts().sort_index()

# # Apply the function to each aspect column
# aspect_count_df = pd.DataFrame({
#     'aspect1': count_values_in_aspect_bins(aspect_df['aspect1']),
#     'aspect2': count_values_in_aspect_bins(aspect_df['aspect2']),
#     'aspect3': count_values_in_aspect_bins(aspect_df['aspect3'])
# })

# # Add a column for the total count
# aspect_count_df['Total'] = aspect_count_df.sum(axis=1)

# # Save the aspect count DataFrame to CSV
# aspect_count_df.to_csv(output_csv, index=True)
#%% Point manipulation
#input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/slope/points2.shp"
# input_shapefile2 = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Ottadalen-NGI/Polygons_Ottadalen_Hans.shp"

# output_csv = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/skredtype/validation_counts.csv"

# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]

# print("Reading shp of inventory points")
# #pts = gpd.read_file(input_shapefile)  # Read .shp of points
# polygons = gpd.read_file(input_shapefile2)
# # Create a Polygon representing the bounding box
# bounding_box = Polygon(square_coords)

# Filter points within the bounding box
#sweco_points = pts[pts.geometry.within(bounding_box)]

# Plot the points
#ax = sweco_points.plot(marker='o', color='blue', markersize=5, label='Points')

# Plot the bounding box
# x, y = zip(*square_coords)
# bounding_box_polygon = Polygon(square_coords)
# bounding_box_x, bounding_box_y = bounding_box_polygon.exterior.xy
# ax.plot(bounding_box_x, bounding_box_y, color='red', linewidth=2, label='Bounding Box')

# # Customize the plot
# ax.set_title('Points within Bounding Box')
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# ax.legend()

# # Show the plot
# plt.show()

#df = pd.concat([sweco_points['Validation']], axis=1)

# skredtyper = ['DFw', 'DS', 'DFd', '144', '140', '142', '145', '140']

# #Count occurrences of each unique value in the 'Validation' column
# validation_counts = df['Validation'].value_counts()

# #Display the counts
# print(validation_counts)

#validation_counts.to_csv(output_csv, index=False)
#%% Total area of forest 
#input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/Ottadalen-NGI/Release point Ottadalen.shp"
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/slope/points2.shp"
# input_dovre = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/dovre/dovre.shp'
# input_lom = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/lom/lom.shp'
# input_sel = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/sel/sel.shp'
# input_skjåk = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/skjåk/skjåk.shp'
# input_vågå = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/vågå/vågå.shp'
# input_nordfron = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/nord_fron/nord fron.shp'
# #Output CSV file
# #output_csv = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/biomasse.csv"

# input_dovre_sr = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/dovre/dovre.shp'
# input_lom_sr = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/lom/lom.shp'
# input_sel_sr = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/sel/sel.shp'
# input_skjåk_sr = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/skjåk/skjåk.shp'
# input_vågå_sr = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/vågå/vågå.shp'
# input_nordfron_sr = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/SR16/nord_fron/nord fron.shp'

# lom_sr = gpd.read_file(input_lom_sr)
# vågå_sr = gpd.read_file(input_vågå_sr)
# dovre_sr = gpd.read_file(input_dovre_sr)
# skjåk_sr = gpd.read_file(input_skjåk_sr)
# sel_sr = gpd.read_file(input_sel_sr)
# nordfron_sr = gpd.read_file(input_nordfron_sr)

# lom = gpd.read_file(input_lom)
# vågå = gpd.read_file(input_vågå)
# dovre = gpd.read_file(input_dovre)
# skjåk = gpd.read_file(input_skjåk)
# sel = gpd.read_file(input_sel)
# nordfron = gpd.read_file(input_nordfron)

# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)
# square = gpd.GeoDataFrame(geometry=[square_polygon], crs='EPSG:25833')

# print("Reading shp of inventory points")
# pts = gpd.read_file(input_shapefile)  # Read .shp of points

# # Create a Polygon representing the bounding box
# bounding_box = Polygon(square_coords)

# # Filter points within the bounding box
# sweco_points = pts[pts.geometry.within(bounding_box)]

# npnts = len(pts)
# coords = [(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]

# joined_data_lom = gpd.sjoin(sweco_points, lom_sr[['geometry', 'srtreslags']], how="inner", op="within")
# joined_data_vågå = gpd.sjoin(sweco_points, vågå_sr[['geometry', 'srtreslags']], how="inner", op="within")
# joined_data_dovre = gpd.sjoin(sweco_points, dovre_sr[['geometry', 'srtreslags']], how="inner", op="within")
# joined_data_skjåk = gpd.sjoin(sweco_points, skjåk_sr[['geometry', 'srtreslags']], how="inner", op="within")
# joined_data_sel = gpd.sjoin(sweco_points, sel_sr[['geometry', 'srtreslags']], how="inner", op="within")
# joined_data_nordfron = gpd.sjoin(sweco_points, nordfron_sr[['geometry', 'srtreslags']], how="inner", op="within")

# joined_data_dovre = gpd.overlay(dovre, square, how="intersection")
# joined_data_lom = gpd.overlay(lom, square, how="intersection")
# joined_data_sel = gpd.overlay(sel, square, how="intersection")
# joined_data_skjåk = gpd.overlay(skjåk, square, how="intersection")
# joined_data_vågå = gpd.overlay(vågå, square, how="intersection")
# joined_data_nordfron = gpd.overlay(nordfron, square, how="intersection")

# joined_data_sr = pd.concat([joined_data_lom, joined_data_vågå, joined_data_dovre, joined_data_skjåk, joined_data_sel, joined_data_nordfron])

# #sr_df = joined_data_sr[['ORIG_FID', 'srtreslags']]
# sr_df = joined_data_sr[['srtreslags']]
# # Count occurrences of each 'artype' and create a new DataFrame
# treslag_counts = joined_data_sr['srtreslags'].value_counts().reset_index()
# # artype_counts.columns = ['artype', 'artype_count']

# # Calculate the area of each intersection
# for df in [joined_data_dovre, joined_data_lom, joined_data_sel, joined_data_skjåk, joined_data_vågå, joined_data_nordfron]:
#     df['intersection_area'] = df['geometry'].area
#     df['area_km2'] = df['intersection_area'] / 1e6
# all_joined_data = pd.concat([joined_data_dovre, joined_data_lom, joined_data_sel, joined_data_skjåk, joined_data_vågå, joined_data_nordfron])
# # # Create a DataFrame with unique 'artype' values
# # unique_artypes = pd.DataFrame({'artype': joined_data['artype'].unique()})
# total_area_forest = all_joined_data.groupby('srtreslags')['area_km2'].sum().reset_index()
# # # Merge the unique_artypes DataFrame with the counts DataFrame
# # unique_artypes = pd.merge(unique_artypes, artype_counts, on='artype', how='left')
# #treslag_counts.to_csv(output_csv, index=False)

#%% Mergeddf excel file creation

# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/slope/points2.shp"
# input_raster1 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/slope/Slope_dtm10_6_1.tif'
# input_raster2 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/slope/Slope_dtm10_5.tif'
# input_raster3 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/slope/Slope_dtm10_4.tif'
# output_csv = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/mergeddf_attempt.csv"
# input_soil = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/Losmasse/LosmasseFlate_20231125.shp'
# input_dovre = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/ar50/vegetation1.shp'
# input_lom = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/ar50/lom/lom.shp'
# input_sel = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/ar50/sel/sel.shp'
# input_skjåk = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/ar50/skjåk/skjåk.shp'
# input_vågå = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/vegetation/ar50/vågå/vågå.shp'

# lom = gpd.read_file(input_lom)
# vågå = gpd.read_file(input_vågå)
# dovre = gpd.read_file(input_dovre)
# skjåk = gpd.read_file(input_skjåk)
# sel = gpd.read_file(input_sel)

# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/Aspect/total_aspect.tif'
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/slope/points2.shp"

# pts = gpd.read_file(input_shapefile)

# with rasterio.open(input_raster) as src:
#     # Ensure the raster dataset has a transformation
#     assert src.transform, "Raster dataset must have a transformation."

#     # Read points from the shapefile
#     pts = gpd.read_file(input_shapefile)

#     # Filter points within the bounding box
#     square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
#     square_polygon = Polygon(square_coords)
#     square = gpd.GeoDataFrame(geometry=[square_polygon], crs=src.crs)

#     pts_within_square = pts[pts.geometry.within(square.geometry.iloc[0])]

#     # Sample the aspect values at the point locations within the square
#     points_values = list(src.sample(zip(pts_within_square.geometry.x, pts_within_square.geometry.y)))

#     # Filter out masked values and print the results
#     for point, value in zip(pts_within_square.geometry, points_values):
#         if not value.item() is None:
#             print(f"Point: {point}, Aspect Value: {value.item()}")
# data = [{'Point': point, 'Aspect Value': value.item()} for point, value in zip(pts_within_square.geometry, points_values) if value is not None]

# # Create the DataFrame
# aspect_df = pd.DataFrame(data)

# soil = gpd.read_file(input_soil)

# joined_data_lom = gpd.sjoin(pts[pts.geometry.within(bounding_box)], lom[['geometry', 'artype']], how="inner", op="within")
# joined_data_vågå = gpd.sjoin(pts[pts.geometry.within(bounding_box)], vågå[['geometry', 'artype']], how="inner", op="within")
# joined_data_dovre = gpd.sjoin(pts[pts.geometry.within(bounding_box)], dovre[['geometry', 'artype']], how="inner", op="within")
# joined_data_skjåk = gpd.sjoin(pts[pts.geometry.within(bounding_box)], skjåk[['geometry', 'artype']], how="inner", op="within")
# joined_data_sel = gpd.sjoin(pts[pts.geometry.within(bounding_box)], sel[['geometry', 'artype']], how="inner", op="within")

# joined_data_ar = pd.concat([joined_data_lom, joined_data_vågå, joined_data_dovre, joined_data_skjåk, joined_data_sel])
# #joined_data_ar_done = joined_data_ar[['ORIG_FID', 'artype']]
# joined_data_ar_done = joined_data_ar[['artype']]


# def assign_slope(pts, input_raster, slope_col_name):
#     with rasterio.open(input_raster) as src_dataset:
#         print("Reading data: {}".format(input_raster))

#         pts[slope_col_name] = [x for x in src_dataset.sample(coords)]
#         pts[slope_col_name] = pts.apply(lambda x: x[[slope_col_name]][0], axis=1)

# # Assign slope values for each raster
# assign_slope(pts, input_raster1, 'slope1')
# assign_slope(pts, input_raster2, 'slope2')
# assign_slope(pts, input_raster3, 'slope3')

# # slope_df = pd.DataFrame({
# #     'ORIG_FID': pts['ORIG_FID'],  
# #     'slope1': pts['slope1'],
# #     'slope2': pts['slope2'],
# #     'slope3': pts['slope3']
# # })
# slope_df = pd.DataFrame({ 
#     'slope1': pts['slope1'],
#     'slope2': pts['slope2'],
#     'slope3': pts['slope3']
# })

# slope_df['Slope angle'] = slope_df[['slope1', 'slope2', 'slope3']].sum(axis=1)

# slope_df.index.name = 'Slope angle (degrees)'

# # Keep only the 'total' column and the 'slope angle' column
# slope_df = slope_df[['Slope angle']]

# joined_soil = gpd.sjoin(pts, soil, how="inner", op="within")
# #result_soil = joined_soil[['ORIG_FID', 'jorda_navn']]
# result_soil = joined_soil[['jorda_navn']]

# merged_df = pd.concat([pts['Skredtype'], pts['ORIG_FID'], slope_df, result_soil, joined_data_ar_done, aspect_df, sr_df], axis=1)
# merged_df.to_csv(output_csv, index=False)

#%% slope angle and type of landslide
# path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/resultater/mergeddf.xlsx"
# # #path2 = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/mergeddf_ngi.csv"
# hsv = pd.read_excel(path)
# # input_shapefile2 = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/Ottadalen-NGI/Polygons_Ottadalen_Hans.shp"
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/Ottadalen-NGI/Release point Ottadalen.shp"
# #polygons = gpd.read_file(input_shapefile2)
# pts = gpd.read_file(input_shapefile)
# # Remove brackets and convert to float
# # hsv['Slope angle'] = hsv['Slope angle'].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)

# # Define bin edges
# bin_edges = np.arange(0, 90, 10)

# # Define a function to assign bin edges for each Skredtype
# def assign_bins(row):
#     for skredtype in ['DFw', 'DFd', 'DS', 'Debris avalanche', 'Landslide, unspecified']:
#         if row['Skredtype'] == skredtype:
#             return pd.cut([row['Slope angle']], bins=bin_edges, include_lowest=True)[0]  # Access the first element
#     return np.nan

# # Apply the function to create a new column 'Bin'
# hsv['Bin'] = hsv.apply(assign_bins, axis=1)

# # Group by 'Skredtype' and 'Bin' and count occurrences
# result_df = hsv.groupby(['Skredtype', 'Bin']).size().unstack(fill_value=0)

# ax = result_df.T.plot(kind='bar', stacked=True, figsize=(10, 6))

# plt.title('Slope Angle and Landslide', fontsize=16)
# plt.xlabel('Slope angle [°]', fontsize=16)
# plt.ylabel('Number of Landslides', fontsize=16)
# #plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(title='Landslide types', loc='upper right', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize=16)

# # Customize x-axis labels
# bin_labels = [f'{int(b.left)}-{int(b.right)}'.rjust(10) for b in result_df.columns]
# ax.set_xticks(np.arange(len(bin_labels)))
# ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=14)

# plt.tight_layout()
# plt.grid(axis='y')
# plt.show()



#%% aspect and type of landslide

# aspect_bin_edges = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]

# aspect_labels = ['0-22.5', '22.5-67.5', '67.5-112.5', '112.5-157.5', '157.5-202.5', '202.5-247.5', '247.5-292.5', '292.5-337.5', '337.5-360']
# # Define a function to assign bin edges for each Skredtype based on 'Aspect'
# def assign_aspect_bins(row):
#     for skredtype in ['DFw', 'DFd', 'DS', 'Debris avalanche', 'Landslide, unspecified']:
#         if row['Skredtype'] == skredtype:
#             return pd.cut([row['Aspect']], bins=aspect_bin_edges, labels=aspect_labels, include_lowest=True)[0]
#     return np.nan

# # Apply the function to create a new column 'Aspect_Bin'
# hsv['Aspect_Bin'] = hsv.apply(assign_aspect_bins, axis=1)

# # Group by 'Skredtype' and 'Aspect_Bin' and count occurrences
# result_df_aspect = hsv.groupby(['Skredtype', 'Aspect_Bin']).size().unstack(fill_value=0)

# # If needed, you can reset the index to make the result_df_aspect more usable
# result_df_aspect.reset_index(inplace=True)

# #Assuming your DataFrame is named 'result_df_aspect'
# result_df_aspect['0-22.5'] += result_df_aspect['337.5-360']
# result_df_aspect = result_df_aspect.drop(columns='337.5-360')
# result_df_aspect = result_df_aspect.rename(columns={'0-22.5': '337.5-22.5'})

# result_df_aspect.set_index('Skredtype', inplace=True)

# #Assuming 'Aspect_Bin' is the correct column name
# aspect_order = ['337.5-22.5', '22.5-67.5', '67.5-112.5', '112.5-157.5', '157.5-202.5', '202.5-247.5', '247.5-292.5', '292.5-337.5']
# result_df_aspect = result_df_aspect[aspect_order]

# label_mapping = {
#     '337.5-22.5': 'N',
#     '22.5-67.5': 'NE',
#     '67.5-112.5': 'E',
#     '112.5-157.5': 'SE',
#     '157.5-202.5': 'S',
#     '202.5-247.5': 'SW',
#     '247.5-292.5': 'W',
#     '292.5-337.5': 'NW'
# }

# # Rename the columns using the label mapping
# result_df_aspect.columns = [label_mapping[col] for col in result_df_aspect.columns]

# result_df_aspect.T.plot(kind='bar', stacked=True, figsize=(10, 6))
# plt.title('Aspect and Landslide', fontsize = 16)
# plt.xlabel('Slope Aspect', fontsize = 16)
# plt.ylabel('Number of Landslides', fontsize=16)
# plt.xticks(rotation=45, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(title='Landslide types', loc='upper right', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize=16)
# plt.tight_layout()
# plt.grid(axis='y')
# plt.show()

#%% soil type and skredtype
#Group by 'jorda_navn' and 'Skredtype' and count occurrences
# path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/resultater/mergeddf.xlsx"
# # #path2 = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/Fordypningsprosjekt/mergeddf_ngi.csv"
# hsv = pd.read_excel(path)
# result_df_jorda = hsv.groupby(['jorda_navn', 'Skredtype']).size().unstack(fill_value=0)

# # If needed, you can reset the index to make the result_df_jorda more usable
# result_df_jorda.reset_index(inplace=True)

# # Assuming your DataFrame is named 'result_df_jorda'
# result_df_jorda.set_index('jorda_navn', inplace=True)

# # Transpose the DataFrame
# result_df_jorda_transposed = result_df_jorda.T

# # Define shortened labels
# shortened_labels = ['Bedrock', 'Glacifluvial', 'Fluvial', 'WM', 'Till, thick', 'Till, thin', 'MMD']

# # Plot the bar chart with shortened x-labels
# ax = result_df_jorda_transposed.T.plot(kind='bar', stacked=True, figsize=(10, 6))
# plt.title('Soil type and Landslide', fontsize = 16)
# plt.xlabel('Soil type', fontsize=16)
# plt.ylabel('Number of Landslides', fontsize=16)
# plt.xticks(rotation=45, fontsize=14)
# plt.yticks(fontsize=14)

# # Manually set shortened x-labels
# ax.set_xticklabels(shortened_labels)

# plt.yticks(fontsize=12)
# plt.legend(title='Landslide types', loc='upper right', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize = 16)

# # Adjust layout to prevent overlapping
# plt.tight_layout()
# plt.grid(axis='y')
# # Alternatively, if saving to a file, you can use the following:
# # plt.savefig("output.png", bbox_inches='tight')
# plt.show()


#Vegetation and skredtype
# result_df_artype_skredtype = hsv.groupby(['artype', 'Skredtype']).size().unstack(fill_value=0)

# # If needed, you can reset the index to make the result_df_artype_skredtype more usable
# result_df_artype_skredtype.reset_index(inplace=True)

# result_df_artype_skredtype.set_index('artype', inplace=True)

# # Transpose the DataFrame
# result_df_artype_skredtype_transposed = result_df_artype_skredtype.T

# # Specify custom labels for x-axis
# custom_labels = ['Urbanized', 'Agriculture', 'Forest', 'Thin veg. cover']

# result_df_artype_skredtype_transposed.T.plot(kind='bar', stacked=True, figsize=(10, 6))
# plt.title('Area type and Landslide', fontsize = 16)
# plt.xlabel('Area type', fontsize=16)
# plt.ylabel('Number of Landslides', fontsize=16)
# plt.xticks(rotation=45, fontsize=14)
# plt.yticks(fontsize=14)

# # Set custom labels for x-axis
# plt.xticks(range(len(custom_labels)), custom_labels)

# plt.yticks(fontsize=12)
# plt.legend(title='Landslide types', loc='upper right', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize=16)
# plt.tight_layout()
# plt.grid(axis='y')
# plt.show()

#%% SR16 og skredtype

# # Assuming your DataFrame is named 'hsv'
# result_df_srtreslags_skredtype = hsv.groupby(['srtreslags', 'Skredtype']).size().unstack(fill_value=0)

# # If needed, you can reset the index to make the result_df_srtreslags_skredtype more usable
# result_df_srtreslags_skredtype.reset_index(inplace=True)

# # Assuming your DataFrame is named 'result_df_srtreslags_skredtype'
# result_df_srtreslags_skredtype.set_index('srtreslags', inplace=True)

# # Transpose the DataFrame
# result_df_srtreslags_skredtype_transposed = result_df_srtreslags_skredtype.T

# # Specify custom labels for x-axis
# custom_labels = ['Spruce', 'Pine', 'Mix P & S', 'Mix of all', 'Decidous']

# result_df_srtreslags_skredtype_transposed.T.plot(kind='bar', stacked=True, figsize=(10, 6))
# plt.title('Forest type and Landslide', fontsize=16)
# plt.xlabel('Forest type', fontsize=16)
# plt.ylabel('Number of Landslides', fontsize=16)
# plt.xticks(rotation=45, fontsize=14)
# plt.yticks(fontsize=14)

# # Set custom labels for x-axis
# plt.xticks(range(len(custom_labels)), custom_labels)

# plt.yticks(fontsize=12)
# plt.legend(title='Landslide types', loc='upper left', bbox_to_anchor=(0, 1), fontsize=12, title_fontsize=16)
# plt.tight_layout()
# plt.grid(axis='y')
# plt.show()

#%% area slope DEM
# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/høydedata/dtm10/data/total_dem.tif'
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)
# square = gpd.GeoDataFrame(geometry=[square_polygon], crs='EPSG:25833')

# bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# area_by_group = {f"{bin_start}-{bin_end}": 0 for bin_start, bin_end in zip(bin_edges[:-1], bin_edges[1:])}


# with rasterio.open(input_raster) as src:
#     masked_data, _ = mask(src, square.geometry, crop=True)

#     flat_data = masked_data[0].flatten()

#     hist, _ = np.histogram(flat_data, bins=bin_edges)

#     for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
#         area_by_group[f"{bin_start}-{bin_end}"] += (hist[i] * src.res[0] * src.res[1])/1e6

# for group, area in area_by_group.items():
#     print(f"Slope Group: {group}, Area: {area} square km")
# total_area = sum(area_by_group.values()) #see red note book for calculations of frequency!


#%% area aspect DEM
# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/Aspect/total_aspect.tif'

# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)
# square = gpd.GeoDataFrame(geometry=[square_polygon], crs='EPSG:25833')

# aspect_groups = {'N1': (337.5, 360), 'N2': (0, 22.5), 'NE': (22.5, 67.5), 'E': (67.5, 112.5),
#                 'SE': (112.5, 157.5), 'S': (157.5, 202.5), 'SW': (202.5, 247.5), 'W': (247.5, 292.5),
#                 'NW': (292.5, 337.5)}
# area_by_label = {label: 0.0 for label in aspect_groups.keys()}

# with rasterio.open(input_raster) as src:
#     # Mask the raster based on the bounding box geometry
#     masked_data, _ = mask(src, square.geometry, crop=True)

#     # Flatten the 2D array to 1D for histogram calculation
#     flat_data = masked_data[0].flatten()

#     hist, _ = np.histogram(flat_data, bins=np.arange(0, 361, 45))  # Adjust bins for aspect

#     # Iterate over label groups and accumulate area
#     for label, angle_range in aspect_groups.items():
#         lower_bound, upper_bound = angle_range
#         group_mask = np.logical_and(flat_data >= lower_bound, flat_data < upper_bound)
#         area_by_label[label] += np.sum(group_mask) * src.res[0] * src.res[1] / 1e6  # Convert to square km

# # Print the total area for each aspect group
# for label, area in area_by_label.items():
#     print(f"Aspect Group: {label}, Area: {area} square km")
# total_area = sum(area_by_label.values()) #see red note book for calculations of frequency!



#%%NGI and hvl points and aspect based on the new aspect tif file

# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/Aspect/total_aspect.tif'
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NGI_HVL.shp"

# with rasterio.open(input_raster) as src:
#     # Ensure the raster dataset has a transformation
#     assert src.transform, "Raster dataset must have a transformation."

#     # Read points from the shapefile
#     pts = gpd.read_file(input_shapefile)

#     # Filter points within the bounding box
#     square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
#     square_polygon = Polygon(square_coords)
#     square = gpd.GeoDataFrame(geometry=[square_polygon], crs=src.crs)

#     pts_within_square = pts[pts.geometry.within(square.geometry.iloc[0])]

#     # Sample the aspect values at the point locations within the square
#     points_values = list(src.sample(zip(pts_within_square.geometry.x, pts_within_square.geometry.y)))

#     # Filter out masked values and print the results
#     for point, value in zip(pts_within_square.geometry, points_values):
#         if not value.item() is None:
#             print(f"Point: {point}, Aspect Value: {value.item()}")
            
# data = [{'Point': point, 
#           'Aspect Value': value.item(), 
#           'x_utm': point.x, 
#           'y_utm': point.y} 
#         for i, (point, value) in enumerate(zip(pts_within_square.geometry, points_values)) 
#         if value is not None]
# aspect_df = pd.DataFrame(data)
# excel_output_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/Aspect/aspect_points.xlsx'
# aspect_df.to_excel(excel_output_path, index_label='Index')

# aspect_groups = {'N1': (337.5, 360), 'N2': (0, 22.5), 'NE': (22.5, 67.5), 'E': (67.5, 112.5),
#                   'SE': (112.5, 157.5), 'S': (157.5, 202.5), 'SW': (202.5, 247.5), 'W': (247.5, 292.5),
#                   'NW': (292.5, 337.5)}

# # Create a DataFrame to store counts
# counts_df = pd.DataFrame(index=aspect_groups.keys(), columns=['Count'])

# # Initialize counts to zero
# counts_df['Count'] = 0

# # Iterate through aspect groups and count occurrences
# for label, (lower_bound, upper_bound) in aspect_groups.items():
#     group_mask = [(val >= lower_bound) and (val < upper_bound) for val in points_values]
#     counts_df.at[label, 'Count'] = sum(group_mask)

# excel_output_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/Aspect/ngi_points_aspect.xlsx'
# counts_df.to_excel(excel_output_path, index_label='Aspect Group')

#%% Area tree type and points within different types

# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NGI_HVL.shp"
# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTRESLAG.tif'
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)
# square = gpd.GeoDataFrame(geometry=[square_polygon], crs='EPSG:25833')
# pts = gpd.read_file(input_shapefile)

# with rasterio.open(input_raster) as src:
#     # Ensure the raster dataset has a transformation
#     assert src.transform, "Raster dataset must have a transformation."

#     # Read points from the shapefile
#     pts = gpd.read_file(input_shapefile)

#     # Filter points within the bounding box
#     square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
#     square_polygon = Polygon(square_coords)
#     square = gpd.GeoDataFrame(geometry=[square_polygon], crs=src.crs)

#     pts_within_square = pts[pts.geometry.within(square.geometry.iloc[0])]

#     # Sample the aspect values at the point locations within the square
#     points_values = list(src.sample(zip(pts_within_square.geometry.x, pts_within_square.geometry.y)))

#     # Filter out masked values and print the results
#     for point, value in zip(pts_within_square.geometry, points_values):
#         if not value.item() is None:
#             print(f"Point: {point}, Treetype: {value.item()}")

# # Create a list of dictionaries containing the data
# data = [{'Point': point, 
#           'Treetype': value.item(), 
#           'ORIG_FID': pts_within_square.iloc[i]['ORIG_FID'],
#           'x_utm': point.x, 
#           'y_utm': point.y} 
#         for i, (point, value) in enumerate(zip(pts_within_square.geometry, points_values)) 
#         if value is not None]

# # # Create the DataFrame
# treetype_df = pd.DataFrame(data)
# excel_output_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/tretype/tretype_points.xlsx'
# treetype_df.to_excel(excel_output_path, index_label='Index')

# counts_df = pd.DataFrame(index=['Treetype -9999', 'Treetype 1', 'Treetype 2', 'Treetype 3'], columns=['Count'])

# # Initialize counts to zero
# counts_df['Count'] = 0

# # Iterate through treetypes and count occurrences
# for treetype in [-9999, 1, 2, 3]:
#     treetype_mask = [val.item() == treetype for val in points_values]
#     counts_df.at[f'Treetype {treetype}', 'Count'] = sum(treetype_mask)
    
# index_mapping = {
#     'Treetype -9999': 'No trees',
#     'Treetype 1': 'Spruce',
#     'Treetype 2': 'Pine',
#     'Treetype 3': 'Deciduous'
# }

# # Rename the index using the mapping
# counts_df = counts_df.rename(index=index_mapping)
# # Display the counts DataFrame
# print(counts_df)

# excel_file_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/tretype/tretype_count_hvl.xlsx'

# # Export the DataFrame to Excel
# counts_df.to_excel(excel_file_path)

# #%% Area tree type NGI 
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/1. semester/Fordypningsprosjekt/Ottadalen-NGI/Release point Ottadalen.shp"
# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTRESLAG.tif'
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)
# square = gpd.GeoDataFrame(geometry=[square_polygon], crs='EPSG:25833')
# pts = gpd.read_file(input_shapefile)

# with rasterio.open(input_raster) as src:
#     # Mask the raster based on the bounding box geometry
#     masked_data, _ = mask(src, square.geometry, crop=True)

#     # Flatten the 2D array to 1D for counting occurrences
#     flat_data = masked_data[0].flatten()

#     # Count occurrences of each treetype value
#     counts = {treetype: np.sum(flat_data == treetype) for treetype in [-9999, 1, 2, 3]}

#     # Accumulate area for each treetype
#     area_by_group = {str(treetype): (counts[treetype] * src.res[0] * src.res[1]) / 1e6 for treetype in counts}

# # Print the total area for each treetype group
# for group, area in area_by_group.items():
#     print(f"Treetype: {group}, Area: {area} square km")

# total_area = sum(area_by_group.values())
# print(f"Total Area: {total_area} square km")

# #%% points within the different treetypes NGI
# with rasterio.open(input_raster) as src:
#     # Ensure the raster dataset has a transformation
#     assert src.transform, "Raster dataset must have a transformation."

#     # Read points from the shapefile
#     pts = gpd.read_file(input_shapefile)

#     # Filter points within the bounding box
#     square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
#     square_polygon = Polygon(square_coords)
#     square = gpd.GeoDataFrame(geometry=[square_polygon], crs=src.crs)

#     pts_within_square = pts[pts.geometry.within(square.geometry.iloc[0])]

#     # Sample the aspect values at the point locations within the square
#     points_values = list(src.sample(zip(pts_within_square.geometry.x, pts_within_square.geometry.y)))

#     # Filter out masked values and print the results
#     for point, value in zip(pts_within_square.geometry, points_values):
#         if not value.item() is None:
#             print(f"Point: {point}, Treetype: {value.item()}")
            
# data = [{'Point': point, 
#           'Treetype': value.item(), 
#           'x_utm': point.x, 
#           'y_utm': point.y} 
#         for i, (point, value) in enumerate(zip(pts_within_square.geometry, points_values)) 
#         if value is not None]
# aspect_df = pd.DataFrame(data)
# excel_output_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/tretype/tretype_ngi_points.xlsx'
# aspect_df.to_excel(excel_output_path, index_label='Index')

# counts_df = pd.DataFrame(index=['Treetype -9999', 'Treetype 1', 'Treetype 2', 'Treetype 3'], columns=['Count'])

# # Initialize counts to zero
# counts_df['Count'] = 0

# # Iterate through treetypes and count occurrences
# for treetype in [-9999, 1, 2, 3]:
#     treetype_mask = [val.item() == treetype for val in points_values]
#     counts_df.at[f'Treetype {treetype}', 'Count'] = sum(treetype_mask)
    
# index_mapping = {
#     'Treetype -9999': 'No trees',
#     'Treetype 1': 'Spruce',
#     'Treetype 2': 'Pine',
#     'Treetype 3': 'Deciduous'
# }

# # Rename the index using the mapping
# counts_df = counts_df.rename(index=index_mapping)
# # Display the counts DataFrame
# print(counts_df)

#excel_file_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/tretype/tretype_count_ngi.xlsx'

# Export the DataFrame to Excel
#counts_df.to_excel(excel_file_path)



#%% flow accumulation
# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/flow_accu/flowacc_tot.tif'
# input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NVE_NGIHVL.shp"

# with rasterio.open(input_raster) as src:

#     assert src.transform, "Raster dataset must have a transformation."
#     pts = gpd.read_file(input_shapefile)
#     square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
#     square_polygon = Polygon(square_coords)
#     square = gpd.GeoDataFrame(geometry=[square_polygon], crs=src.crs)

#     pts_within_square = pts[pts.geometry.within(square.geometry.iloc[0])]
#     points_values = list(src.sample(zip(pts_within_square.geometry.x, pts_within_square.geometry.y)))

#     for point, value in zip(pts_within_square.geometry, points_values):
#         if not value.item() is None:
#             print(f"Point: {point}, Value: {value.item()}")
# data = [{'Point': point, 
#           'Flow accumulation area km2': value.item()*100*10**(-6), 
#           'x_utm': point.x, 
#           'y_utm': point.y} 
#         for i, (point, value) in enumerate(zip(pts_within_square.geometry, points_values)) 
#         if value is not None]
# flow_accu_df = pd.DataFrame(data)
# excel_output_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/flow_accu/flowacc_points_area.xlsx'
# flow_accu_df.to_excel(excel_output_path, index_label='Index')

# bin_edges = [0, 0.0001, 0.001, 0.01, 0.1, 1, 1*10**12]

# area_by_group = {f"{bin_start}-{bin_end}": 0 for bin_start, bin_end in zip(bin_edges[:-1], bin_edges[1:])}

# with rasterio.open(input_raster) as src:
#     masked_data, _ = mask(src, square.geometry, crop=True)
#     flat_data = masked_data[0].flatten()
#     flat_data = flat_data *100 *10**(-6)
#     hist, _ = np.histogram(flat_data, bins=bin_edges)

#     for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
#         area_by_group[f"{bin_start}-{bin_end}"] += (hist[i] * src.res[0] * src.res[1])/1e6

# for group, area in area_by_group.items():
#     print(f"Flow accumulation Group: {group}, Area: {area} square km")
# total_area = sum(area_by_group.values()) 

# area_df = pd.DataFrame(area_by_group.items(), columns=['Flow Accumulation Group', 'Area (square km)'])

# merged_df = pd.concat([flow_accu_df, area_df.set_index('Flow Accumulation Group')], axis=1)

# bins = [0, 0.0001, 0.001, 0.01, 0.1, 1]
# flow_accu_df['m2'] = pd.cut(flow_accu_df['Flow accumulation area km2'], bins=bins, include_lowest=True)
# counts_df = flow_accu_df['m2'].value_counts().sort_index()

# merged_df = pd.concat([merged_df, counts_df], axis=1)

# merged_df.to_excel(excel_output_path)

# bin_labels = ['0-0.0001', '0.0001-0.001', '0.001-0.01', '0.01-0.1', '0.1-1']
# num_landslides = [47, 66, 79, 30, 8] 
# frequencies = [0.65, 0.9, 1.2, 1.7, 2.3]

# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.bar(bin_labels, num_landslides, color='blue', alpha=0.7, label='Number of Landslides')
# ax1.set_xlabel('km\u00B2', fontsize=16)
# ax1.set_ylabel('Number of Landslides', color='blue', fontsize=16)
# ax1.tick_params('y', colors='blue', labelsize = 14)
# ax2 = ax1.twinx()
# ax2.plot(bin_labels,frequencies, color='red', marker='o', label='Frequency')
# ax2.set_ylabel('Frequency', color='red', fontsize=16)
# ax2.tick_params('y', colors='red', labelsize = 14)
# ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
# ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.93), fontsize=12)
# plt.title('Landslides and Landslide Frequencies of Flow Accumulation', fontsize=16)
# ax1.set_xticklabels(bin_labels, fontsize=14) 
# plt.show()
#%%area forest parameters

# input_raster = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRBMU.tif'
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)
# square = gpd.GeoDataFrame(geometry=[square_polygon], crs='EPSG:25833')

# bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 190, 10**9]

# area_by_group = {f"{bin_start}-{bin_end}": 0 for bin_start, bin_end in zip(bin_edges[:-1], bin_edges[1:])}


# with rasterio.open(input_raster) as src:
#     masked_data, _ = mask(src, square.geometry, crop=True)

#     flat_data = masked_data[0].flatten()

#     hist, _ = np.histogram(flat_data, bins=bin_edges)

#     for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
#         area_by_group[f"{bin_start}-{bin_end}"] += (hist[i] * src.res[0] * src.res[1])/1e6


# for group, area in area_by_group.items():
#     print(f"Biomass [t/ha] Group: {group}, Area: {area} square km")
# total_area = sum(area_by_group.values())

# df_area = pd.DataFrame(list(area_by_group.items()), columns=['Bin', 'Area'])

# # Save the DataFrame to an Excel file
# output_excel_path = 'area_by_bin.xlsx'
# df_area.to_excel(output_excel_path, index=False)
# C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/values_forest_parameters.xlsx

# import os
# import pandas as pd
# import rasterio
# from rasterio.mask import mask
# from shapely.geometry import Polygon
# import geopandas as gpd
# import numpy as np

# # Function to calculate area by group for a given raster
# def calculate_area_by_group(input_raster, square_geometry, bin_edges):
#     area_by_group = {f"{bin_start}-{bin_end}": 0 for bin_start, bin_end in zip(bin_edges[:-1], bin_edges[1:])}
    
#     with rasterio.open(input_raster) as src:
#         masked_data, _ = mask(src, [square_geometry], crop=True)
#         flat_data = masked_data[0].flatten()
#         hist, _ = np.histogram(flat_data, bins=bin_edges)
        
#         for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
#             area_by_group[f"{bin_start}-{bin_end}"] += (hist[i] * src.res[0] * src.res[1]) / 1e6
    
#     return area_by_group

# # Define input raster files and their corresponding bin edges
# input_rasters = {
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRBMU.tif": [-9999, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 190, 1000],
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRDIAMMIDDEL_GE8.tif': [-9999, 0, 10, 20, 30, 40,1000, 10000],  # cm
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRMHOYDE.tif': [-9999, 0, 50, 100, 150, 200, 250, 1000, 10000, 20000],  # Dm
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRGRFLATE.tif': [-9999, 0, 10, 20, 30, 40, 50, 60, 1000, 10000],  # m2/hektar
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTREANTALL.tif': [-9999, 0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 10000, 20000, 100000],  # tree/hektar
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTREANTALL_GE16.tif': [-9999, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 10000, 20000, 100000],  # tree/hektar
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRKRONEDEK.tif': [-9999, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000, 10000]  # %
# }

# # Define square polygon geometry
# square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
# square_polygon = Polygon(square_coords)
# square_geometry = gpd.GeoDataFrame(geometry=[square_polygon], crs='EPSG:25833').geometry.iloc[0]

# # Create a dictionary to store area by group for each raster
# area_by_group_dict = {}

# # Loop over each raster and calculate area by group
# for input_raster, bin_edges in input_rasters.items():
#     # Calculate area by group
#     area_by_group = calculate_area_by_group(input_raster, square_geometry, bin_edges)
    
#     # Extract raster name from file path
#     raster_name = os.path.splitext(os.path.basename(input_raster))[0]
    
#     # Store area by group in the dictionary
#     area_by_group_dict[raster_name] = area_by_group

# # Convert dictionary to DataFrame
# df = pd.DataFrame(area_by_group_dict)

# # Write DataFrame to Excel file with each raster's data on a separate sheet
# output_excel_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/area.xlsx'
# with pd.ExcelWriter(output_excel_path) as writer:
#     for raster_name, data in area_by_group_dict.items():
#         df_raster = pd.DataFrame(list(data.items()), columns=['Group', 'Area (square km)'])
#         df_raster.to_excel(writer, sheet_name=raster_name, index=False)

