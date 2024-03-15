# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:24:15 2024

@author: Isak9
"""

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

#%%Creating excel sheets/parquet files for the entire study area to use
square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
square_polygon = Polygon(square_coords)
gdf_square = gpd.GeoDataFrame([1], geometry=[square_polygon], crs="EPSG:25833") 

raster_files = {
    "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/cohesion/cohesion_10/cohesion_10.tif": "cohesion",   
    "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/tretype/tretype.tif": "tree type",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/flow_accu/flowacc_tot.tif': "flow accumulation",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/høydedata/dtm10/data/total_dem.tif': "slope angle",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/Aspect/total_aspect.tif': "slope aspect",
    'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/soil_raster/soil_raster.tif': "soil"
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

geometry = [Point(xy) for xy in zip(df_pixels['x'], df_pixels['y'])]

# Create a GeoDataFrame from df_pixels
gdf_pixels = gpd.GeoDataFrame(df_pixels, geometry=geometry, crs="EPSG:25833")

# Define the path for the output shapefile
output_shapefile = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/df_SA.shp'

# Save the GeoDataFrame to a shapefile
gdf_pixels.to_file(output_shapefile)

# df_pixels.to_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/study_area.parquet', index=True)

#%% Read in the dataframe created above
df_study_area = pd.read_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/study_area.parquet', engine='fastparquet')
landslides = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ml_landslide.xlsx')
df = gpd.read_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/study_area.parquet', engine='fastparquet')

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
df_study_area['landslide_count'] = 0


for index, landslide in landslides.iterrows():
    matches = df_study_area[
        (landslide['x'] >= df_study_area['x_min']) & 
        (landslide['x'] <= df_study_area['x_max']) & 
        (landslide['y'] >= df_study_area['y_min']) & 
        (landslide['y'] <= df_study_area['y_max'])
    ]
    
    for match_index in matches.index:
        df_study_area.loc[match_index, 'landslide'] = 1
        df_study_area.loc[match_index, 'landslide_count'] += 1
        
landslide_distribution = df_study_area['landslide_count'].value_counts().sort_index()
print(landslide_distribution)
total_landslides = df_study_area['landslide'].sum()
print(f"Total landslide occurrences: {total_landslides}")
#it is only 227 landslides, but that is because we are only counting amount of pixels
# where landslides happen. the code above provides info that 3 pixels have two landslides.
majority = df_study_area[df_study_area['landslide'] == 0]
minority = df_study_area[df_study_area['landslide'] == 1]

# Downsample the majority class
sample_fraction = 0.0001
majority_downsampled = majority.sample(frac=sample_fraction, random_state=42)
df_balanced = pd.concat([majority_downsampled, minority], ignore_index=True)

#scatter plot
plt.figure()
landslide_points = df_balanced[df_balanced['landslide'] == 1]
non_landslide_points = df_balanced[df_balanced['landslide'] == 0]
plt.scatter(non_landslide_points['x'], non_landslide_points['y'], color='blue', label='No Landslide')
plt.scatter(landslide_points['x'], landslide_points['y'], color='red', label='Landslide')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Landslide Occurrences')
plt.legend()
plt.show()

X = df_balanced.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1) #not parameters that decide landslides
# X = df_balanced.drop(['slope aspect', 'flow accumulation', 'tree type', 'soil', 'cohesion','x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
y = df_balanced['landslide'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

from imblearn.over_sampling import SMOTE
from collections import Counter

counter = Counter(y_train)
print('Before', counter)
smt = SMOTE()
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
counter = Counter(y_train_res)
print('Before', counter)
# Initialize SMOTE with the specified sampling strategy
smote = SMOTE(sampling_strategy=1, random_state=42)

#smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

clf = RandomForestClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
#CHECK FOR OVERFITTING
y_train_pred = clf.predict(X_train)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
y_pred_proba_lr = logreg.predict_proba(X_test)[:, 1]
y_train_pred_lr = logreg.predict(X_train)

#CHECK FOR OVERFITTING, SEE HOW THE MODEL DOES ON TRAINED DATA COMPARED TO TEST DATA
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
# For regression models, you can use mean_squared_error:
train_mse = mean_squared_error(y_train, y_train_pred_lr)
# Calculate metrics for validation/test data
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred_lr)

train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
train_precision_lr = precision_score(y_train, y_train_pred_lr)
train_recall_lr = recall_score(y_train, y_train_pred_lr)
train_f1_lr = f1_score(y_train, y_train_pred_lr)

test_accuracy_lr = accuracy_score(y_test, y_pred_lr)
test_precision_lr = precision_score(y_test, y_pred_lr)
test_recall_lr = recall_score(y_test, y_pred_lr)
test_f1_lr = f1_score(y_test, y_pred_lr)

#Scatter plot SMOTE
# plt.figure()
# plt.scatter(X_train[y_train == 0]['x'], X_train[y_train == 0]['y'], color='blue', label='No Landslide (Original)')
# #plt.scatter(X_train[y_train == 1]['x'], X_train[y_train == 1]['y'], color='red', label='Landslide (Original)')
# plt.scatter(X_train_res[y_train_res == 1]['x'], X_train_res[y_train_res == 1]['y'], color='orange', label='Landslide (SMOTE)')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('SMOTE Visualization')
# plt.legend()
# plt.show()

# clf = RandomForestClassifier(random_state=42)
# clf.fit(X_train_res, y_train_res)

# y_pred = clf.predict(X_test)

#TSS
# conf_matrix = confusion_matrix(y_test, y_pred)
# TP = conf_matrix[1, 1]
# TN = conf_matrix[0, 0]
# FP = conf_matrix[0, 1]
# FN = conf_matrix[1, 0]
# sensitivity = TP / (TP + FN)
# specificity = TN / (TN + FP)
# TSS = sensitivity + specificity - 1
# print("True Skill Statistics (TSS):", TSS)

# print(classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision", precision_score(y_test, y_pred))

#y_pred_proba = clf.predict_proba(X_test)[:, 1]

# from sklearn.metrics import roc_curve, roc_auc_score
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# auc = roc_auc_score(y_test, y_pred_proba)

# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve of all parameters')
# plt.legend(loc='lower right')
# plt.grid(True)

# plt.text(0.6, 0.2, f'AUC = {auc:.2f}', fontsize=12, ha='center')

# plt.show()

#LOGISTIC REGRESSION, HOW CAN I MAKE THE SUSCEPTIBILITY MAP BASED ON THIS. 
#ADD AR50 AND MAYBE FOREST PARAMETERS. 

# importances = clf.feature_importances_
# feature_names = X_train.columns

# # Sort feature importances in descending order
# indices = np.argsort(importances)[::-1]

# plt.figure(figsize=(10, 6))
# plt.title("Feature Importances")
# plt.bar(range(X_train.shape[1]), importances[indices], color="b", align="center")
# plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.xlabel("Features")
# plt.ylabel("Importance")
# plt.show()







