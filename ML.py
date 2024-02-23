# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:24:15 2024

@author: Isak9
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from IPython.display import HTML

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from shapely.geometry import box
#import seaborn as sns
#import tifffile
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import box
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from rasterio.windows import from_bounds

#%%# Load the Iris dataset

# iris = load_iris()
# X = iris.data
# y = iris.target
# names = iris.feature_names

# df = pd.DataFrame(data=X, columns=names)
# df['target'] = y

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# # Train the Random Forest model on the training data
# rf_classifier.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = rf_classifier.predict(X_test)

# # Evaluate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

#%%Creating excel sheets/parquet files for the entire study area to use in SMOTE
square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
square_polygon = Polygon(square_coords)
gdf_square = gpd.GeoDataFrame([1], geometry=[square_polygon], crs="EPSG:25833") 

# raster_files = {
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/cohesion/cohesion_10/cohesion_10.tif": "cohesion",   
#     "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/tretype/tretype.tif": "tree type",
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/flow_accu/flowacc_tot.tif': "flow accumulation",
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/høydedata/dtm10/data/total_dem.tif': "slope angle",
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/Aspect/total_aspect.tif': "slope aspect",
#     'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/soil_raster/soil_raster.tif': "soil"
# }

# polygon_bounds = square_polygon.bounds

# def read_raster_data(raster_path, polygon_bounds):
#     with rasterio.open(raster_path) as src:
#         window = from_bounds(*polygon_bounds, transform=src.transform)
#         data = src.read(1, window=window, masked=True)
#         return np.ma.filled(data, fill_value=0).flatten()

# raster_data = {column_name: read_raster_data(path, polygon_bounds) for path, column_name in raster_files.items()}
# some_raster_path = next(iter(raster_files.keys()))

# with rasterio.open(some_raster_path) as src:
#     window = from_bounds(*polygon_bounds, transform=src.transform)
#     rows, cols = src.read(1, window=window, masked=True).shape
#     row_indices, col_indices = np.indices((rows, cols))
#     xs, ys = rasterio.transform.xy(src.transform, row_indices.flatten(), col_indices.flatten(), offset='center')

# raster_data['x'] = xs
# raster_data['y'] = ys

# df_pixels = pd.DataFrame(raster_data)

# df_pixels.to_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/study_area.parquet', index=True)

#%% Read in the dataframe created above
df_study_area = pd.read_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/study_area.parquet', engine='fastparquet')
landslides = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ml_landslide.xlsx')

landslides['flow accumulation'] = landslides['flow accumulation'] * 10**4
landslides['slope angle'] = landslides['slope angle'].round(0)
landslides['cohesion'] = landslides['cohesion'].round(0)
landslides['cohesion'] = landslides['cohesion'].replace(-9999, 0)
landslides['tree type'] = landslides['tree type'].replace(-9999, 0)

soil_type_to_code = {
    'Bart fjell': 2,
    'Morenemateriale, sammenhengende dekke, stedvis med stor mektighet': 3,
    'Morenemateriale, usammenhengende eller tynt dekke over berggrunnen': 1,
    'Torv og myr': 4,
    'Ryggformet breelvavsetning (Esker)': 5,
    'Skredmateriale, sammenhengende dekke': 6,
    'Elve- og bekkeavsetning (Fluvial avsetning)': 7,
    'Breelvavsetning (Glasifluvial avsetning)': 8,
    'Randmorene/randmorenesone': 9,
    'Bresjø- eller brekammeravsetning (Glasilakustrin avsetning)': 10,
    'Forvitringsmateriale, usammenhengende eller tynt dekke over berggrunnen': 11,
    'Skredmateriale, usammenhengende eller tynt dekke': 12,
    'Forvitringsmateriale, stein- og blokkrikt (blokkhav)': 13,
    'Tynt dekke av organisk materiale over berggrunn': 16,
    'Fyllmasse (antropogent materiale)': 18,
}

landslides['soil'] = landslides['soil'].map(soil_type_to_code)

pixel_size = 10
half_pixel = pixel_size / 2

df_study_area['x_min'] = df_study_area['x'] - half_pixel
df_study_area['x_max'] = df_study_area['x'] + half_pixel
df_study_area['y_min'] = df_study_area['y'] - half_pixel
df_study_area['y_max'] = df_study_area['y'] + half_pixel

df_study_area['landslide'] = 0

for index, landslide in landslides.iterrows():
    matches = df_study_area[
        (landslide['x'] >= df_study_area['x_min']) & 
        (landslide['x'] <= df_study_area['x_max']) & 
        (landslide['y'] >= df_study_area['y_min']) & 
        (landslide['y'] <= df_study_area['y_max'])
    ]
    
    for match_index in matches.index:
        df_study_area.loc[match_index, 'landslide'] = 1


total_landslides = df_study_area['landslide'].sum()
print(f"Total landslide occurrences: {total_landslides}")
#it is only 227 landslides, but that is because we are only counting amount of pixels
# where landslides happen, and not the amount of landslides. 
df_filtered = df_study_area[df_study_area['slope angle'] > 0]
majority = df_filtered[df_filtered['landslide'] == 0]
minority = df_filtered[df_filtered['landslide'] == 1]

# Downsample the majority class
sample_fraction = 0.1
majority_downsampled = majority.sample(frac=sample_fraction, random_state=42)

# Combine downsampled majority with the minority class
df_balanced = pd.concat([majority_downsampled, minority], ignore_index=True)

X = df_balanced.drop(['x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1) #not parameters that decide landslides
y = df_balanced['landslide'] 

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# majority_class_size = (y_train == 0).sum()
# desired_minority_class_size = majority_class_size // 3

# Calculate the ratio for SMOTE
# This will be the number of minority samples after resampling divided by the number of majority samples
# sampling_strategy = {1: desired_minority_class_size}

# Initialize SMOTE with the calculated sampling_strategy
#smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize and train the classifier
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train_res, y_train_res)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
