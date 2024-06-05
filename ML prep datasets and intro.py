# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:24:15 2024

@author: Isak9
"""
#%%
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
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
from rasterio.windows import from_bounds
#from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.linear_model import LogisticRegression
#%%Creating excel sheets/parquet files for the entire study area to use
square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
square_polygon = Polygon(square_coords)
gdf_square = gpd.GeoDataFrame([1], geometry=[square_polygon], crs="EPSG:25833") 

raster_files = {
    "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/cohesion/cohesion_10/cohesion_10.tif": "root cohesion",   
    "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/tretype/tretype.tif": "tree type",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/flow_accu/flowacc_tot.tif': "flow accumulation",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/høydedata/dtm10/data/total_dem.tif': "slope angle",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/Aspect/total_aspect.tif': "slope aspect",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/soil_raster/soil_raster.tif': "soil",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/antecedent rainfall/precip_aug_2_raster.tif': "precipitation august 2",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/aug_7/precipitation_amount_hans_aug7_resampled_10.tif': 'precipitation august 7',
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/aug_8/precipitation_amount_hans_august8_resampled_10.tif': 'precipitation august 8',
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/aug_9/precipitation_amount_hans_august9_resampled_10.tif': 'precipitation august 9',
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/aug_10/precipitation_amount_hans_august10_resampled_10.tif': "precipitation august 10",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/modified_sat_aug_2_raster.tif': 'soil saturation august 2',
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/aug_7/saturation_raster_2023-08-07_resampled_10.tif': 'soil saturation august 7',
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/aug_8/saturation_raster_2023-08-08_resampled_10.tif': 'soil saturation august 8',
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/aug_9/saturation_raster_2023-08-09_resampled_10.tif': 'soil saturation august 9',
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/precip and saturation data/aug_10/saturation_raster_2023-08-10_resampled_10.tif': 'soil saturation august 10',
}

polygon_bounds = square_polygon.bounds
def read_raster_data(raster_path, polygon_bounds):
    with rasterio.open(raster_path) as src:
        window = from_bounds(*polygon_bounds, transform=src.transform)
        data = src.read(1, window=window, masked=True)
        return np.ma.filled(data, fill_value=0).flatten()

raster_data = {column_name: read_raster_data(path, polygon_bounds) for path, column_name in raster_files.items()}

some_raster_path = next(iter(raster_files.keys()))

with rasterio.open(some_raster_path) as src:
    window = from_bounds(*polygon_bounds, transform=src.transform)
    rows, cols = src.read(1, window=window, masked=True).shape
    row_indices, col_indices = np.indices((rows, cols))
    xs, ys = rasterio.transform.xy(src.transform, row_indices.flatten(), col_indices.flatten(), offset='center')

raster_data['x'] = xs
raster_data['y'] = ys

df_pixels = pd.DataFrame(raster_data)

df_pixels.to_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/study_area.parquet', index=True)

#%% Read in the dataframe created above
# df_study_area = pd.read_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/study_area_corrected_with_rain.parquet', engine='fastparquet')
# landslides = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ml_landslide.xlsx')
# # df = gpd.read_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/study_area.parquet', engine='fastparquet')

# landslides['flow accumulation'] = landslides['flow accumulation'] * 10**4
# #landslides['slope angle'] = landslides['slope angle'].round(0)
# # landslides['cohesion'] = landslides['cohesion'].round(0)
# # landslides['cohesion'] = landslides['cohesion'].replace(-9999, 0)
# landslides['tree type'] = landslides['tree type'].replace(-9999, 0)
# #nan values due to missing data in the xgeo map, NVE API, replacing the nan values with the most natural values based on the map and ArcGIS pro. 
# df_study_area['soil saturation august 7'] = df_study_area['soil saturation august 7'].replace(np.nan, 65)
# df_study_area['soil saturation august 8'] = df_study_area['soil saturation august 8'].replace(np.nan, 90)
# df_study_area['soil saturation august 9'] = df_study_area['soil saturation august 9'].replace(np.nan, 106)
# df_study_area['soil saturation august 10'] = df_study_area['soil saturation august 10'].replace(np.nan, 106)

# soil_type_to_code = {
#     'Bart fjell': 2,
#     'Morenemateriale, sammenhengende dekke, stedvis med stor mektighet': 3,
#     'Morenemateriale, usammenhengende eller tynt dekke over berggrunnen': 1,
#     'Torv og myr': 4,
#     'Ryggformet breelvavsetning (Esker)': 5,
#     'Skredmateriale, sammenhengende dekke': 6,
#     'Elve- og bekkeavsetning (Fluvial avsetning)': 7,
#     'Breelvavsetning (Glasifluvial avsetning)': 8,
#     'Randmorene/randmorenesone': 9,
#     'Bresjø- eller brekammeravsetning (Glasilakustrin avsetning)': 10,
#     'Forvitringsmateriale, usammenhengende eller tynt dekke over berggrunnen': 11,
#     'Skredmateriale, usammenhengende eller tynt dekke': 12,
#     'Forvitringsmateriale, stein- og blokkrikt (blokkhav)': 13,
#     'Tynt dekke av organisk materiale over berggrunn': 16,
#     'Fyllmasse (antropogent materiale)': 18,
# }

# landslides['soil'] = landslides['soil'].map(soil_type_to_code)

# pixel_size = 10
# half_pixel = pixel_size / 2

# df_study_area['x_min'] = df_study_area['x'] - half_pixel
# df_study_area['x_max'] = df_study_area['x'] + half_pixel
# df_study_area['y_min'] = df_study_area['y'] - half_pixel
# df_study_area['y_max'] = df_study_area['y'] + half_pixel
# df_study_area['landslide'] = 0
# df_study_area['landslide_count'] = 0

# for index, landslide in landslides.iterrows():
#     matches = df_study_area[
#         (landslide['x'] >= df_study_area['x_min']) & 
#         (landslide['x'] <= df_study_area['x_max']) & 
#         (landslide['y'] >= df_study_area['y_min']) & 
#         (landslide['y'] <= df_study_area['y_max'])
#     ]
    
#     for match_index in matches.index:
#         df_study_area.loc[match_index, 'landslide'] = 1
#         df_study_area.loc[match_index, 'landslide_count'] += 1
        
# landslide_distribution = df_study_area['landslide_count'].value_counts().sort_index()
# print(landslide_distribution)
# total_landslides = df_study_area['landslide'].sum()
# print(f"Total landslide occurrences: {total_landslides}")
# #it is only 227 landslides, but that is because we are only counting amount of pixels
# # where landslides happen. the code above provides info that 3 pixels have two landslides.
# majority = df_study_area[df_study_area['landslide'] == 0]
# minority = df_study_area[df_study_area['landslide'] == 1]

# num_minority_samples = len(minority)
# num_majority_samples_needed = 3 * num_minority_samples
# majority_needed = majority.sample(n=num_majority_samples_needed, random_state=42)
# df_balanced = pd.concat([majority_needed, minority], ignore_index=True)



#%%







