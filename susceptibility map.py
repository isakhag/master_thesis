# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:05:45 2024

@author: Isak9
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
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
from sklearn.preprocessing import StandardScaler
import os
import dask_geopandas as ddg

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
majority = df_study_area[df_study_area['landslide'] == 0]
minority = df_study_area[df_study_area['landslide'] == 1]

num_minority_samples = len(minority)
num_majority_samples_needed = 3 * num_minority_samples
majority_needed = majority.sample(n=num_majority_samples_needed, random_state=42)
df_balanced = pd.concat([majority_needed, minority], ignore_index=True)

X = df_balanced.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1) #not parameters that decide landslides
# X = df_balanced.drop(['slope aspect', 'flow accumulation', 'tree type', 'soil', 'cohesion','x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
y = df_balanced['landslide'] 
features_list = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#SMOTE
# from imblearn.over_sampling import SMOTE
# from collections import Counter
# counter = Counter(y_train)
# print('Before:', counter)
# Define the desired ratio for the majority class (non-events) to the minority class (events)
# Here, we want approximately 3 times as many non-events as events
# sampling_strategy = {0: int(3 * counter[1])}
# smt = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
# X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
# counter = Counter(y_train_res)
# print('After:', counter)


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

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
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))

from sklearn.metrics import roc_curve, roc_auc_score
#ROC and AUC curve, let the top of the ROC graph be 1, dont need to see above 1.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of all parameters')
plt.legend(loc='lower right')
plt.grid(True)

plt.text(0.6, 0.2, f'AUC = {auc:.2f}', fontsize=12, ha='center')

plt.show()

# feature_imp = pd.Series(clf.feature_importances_, index=features_list).sort_values(ascending=False)
# print(feature_imp)
# feature_imp.plot.bar()

X_SA = df_study_area.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
prediction_SA = clf.predict(X_SA)
prediction_prob=clf.predict_proba(X_SA)
df_study_area['LSM']= prediction_prob[:,1]
df_filtered = df_study_area[df_study_area['LSM'] >= 0.25]

#PLOTTING THE DF_FILTERED
# colors = []
# for prob in df_filtered['LSM']:
#     if 0.25 <= prob < 0.5:
#         colors.append('green')
#     elif 0.5 <= prob <= 0.75:
#         colors.append('yellow')
#     elif 0.75 <= prob <= 1:
#         colors.append('red')         
# plt.figure(figsize=(10, 8))
# plt.scatter(df_filtered['x'], df_filtered['y'], c=colors, marker='.')

# plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='0.25-0.5 PoL'),
#                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='0.5-0.75 PoL'),
#                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='0.75-1 PoL')])

# plt.title('Landslide Susceptibility Map')
# plt.xlabel('East [m]')
# plt.ylabel('North [m]')
# plt.show()

#SENDING THE PROBABILITY OF LANDSLIDE TO TIF-FILE FOR ARCGIS
unique_x = df_study_area['x'].unique()
unique_y = df_study_area['y'].unique()
unique_x.sort()
unique_y.sort()
num_rows = len(unique_y)
num_cols = len(unique_x)
raster_array = np.zeros((num_rows, num_cols), dtype=np.float32)
min_y = unique_y.min()
max_y = unique_y.max()

for index, row in df_study_area.iterrows():
    x_index = np.where(unique_x == row['x'])[0][0]
    y_index = num_rows - 1 - np.where(unique_y == row['y'])[0][0]
    raster_array[y_index, x_index] = row['LSM']

from rasterio.transform import from_origin
pixel_width = 10  
pixel_height = 10 
transform = from_origin(unique_x.min(), unique_y.max(), pixel_width, pixel_height)
crs = "EPSG:25833"  
dtype = raster_array.dtype
count = 1  
driver = 'GTiff'  

output_tif_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/susceptibility_map.tif'

with rasterio.open(output_tif_path, 'w', driver=driver, width=num_cols, height=num_rows, count=count, dtype=dtype, crs=crs, transform=transform) as dst:
    dst.write(raster_array, 1)  

    