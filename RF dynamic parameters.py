# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:37:38 2024

@author: Isak9
"""
#%%libraries 
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
from sklearn.preprocessing import StandardScaler
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skopt.space import Integer, Categorical, Real
from skopt import BayesSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, auc
from rasterio.transform import from_origin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#%% creating subsets of total study area to train model.
#df_study_area = pd.read_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/study_area.parquet', engine='fastparquet')
# landslides = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ml_landslide.xlsx')
# read = df_study_area.iloc[:1000]

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
# df_study_area.loc[df_study_area['soil saturation august 10'] > 100, 'soil saturation august 10'] = 72 #a part of the data was missing and got really high values

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

# for index, landslide in landslides.iterrows():
#     matches = df_study_area[
#         (landslide['x'] >= df_study_area['x_min']) & 
#         (landslide['x'] <= df_study_area['x_max']) & 
#         (landslide['y'] >= df_study_area['y_min']) & 
#         (landslide['y'] <= df_study_area['y_max'])
#     ]
    
#     for match_index in matches.index:
#         df_study_area.loc[match_index, 'landslide'] = 1

# total_landslides = df_study_area['landslide'].sum()
# print(f"Total landslide occurrences: {total_landslides}")
# #it is only 227 landslides, but that is because we are only counting amount of pixels where landslides happen, and not the amount of landslides. 
# majority = df_study_area[df_study_area['landslide'] == 0]
# minority = df_study_area[df_study_area['landslide'] == 1]

# num_minority_samples = len(minority)
# num_majority_samples_needed = 10 * num_minority_samples
# majority_needed = majority.sample(n=num_majority_samples_needed, random_state=42)
# df = pd.concat([majority_needed, minority], ignore_index=True)

# df.to_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/dataset_ratios/1_10_ratio.xlsx', index=False)
#%% probability and predictions from total dataset, not looking into days seperately. 
# df_1 = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/dataset_ratios/13_percent_ratio_updated.xlsx')

# X = df_1.drop(['x', 'y','x_min','y_min','x_max','y_max' ,'landslide', 'Day'], axis=1)
# y = df_1['landslide']
# #RF
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) #data frames with train and test data and dates
# clf = RandomForestClassifier(max_depth = 5, random_state=42) #initiate model
# clf.fit(X_train, y_train) #fit model

# y_pred = clf.predict(X_test) # rembember to drop day column
# y_pred_proba = clf.predict_proba(X_test)[:, 1]
# #CHECK FOR OVERFITTING
# y_train_pred = clf.predict(X_train)
# print("Confusion Matrix RF:\n", confusion_matrix(y_test, y_pred))
# features_list = X.columns
# plt.figure()
# feature_imp = pd.Series(clf.feature_importances_, index=features_list).sort_values(ascending=False)
# feature_imp.plot.bar()
# plt.xticks(rotation=45) 
# plt.tight_layout() 
# plt.show()  

# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_precision = precision_score(y_train, y_train_pred)
# train_recall = recall_score(y_train, y_train_pred)
# train_f1 = f1_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_pred)
# test_precision = precision_score(y_test, y_pred)
# test_recall = recall_score(y_test, y_pred)
# test_f1 = f1_score(y_test, y_pred)

# print("Training Data Metrics RF:")
# print(f"Accuracy: {train_accuracy}")
# print(f"Precision: {train_precision}")
# print(f"Recall: {train_recall}")
# print(f"F1 Score: {train_f1}")
# print("\nValidation/Test Data Metrics RF:")
# print(f"Accuracy: {test_accuracy}")
# print(f"Precision: {test_precision}")
# print(f"Recall: {test_recall}")
# print(f"F1 Score: {test_f1}")

# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# auc = roc_auc_score(y_test, y_pred_proba)
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='green', lw=2, label=f'RF ROC curve (AUC = {auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1])
# plt.xlabel('False Positive Rate', fontsize = 12)
# plt.ylabel('True Positive Rate', fontsize = 12)
# plt.title('ROC Curve', fontsize = 12)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.legend(loc='lower right', fontsize = 14)
# plt.grid(True)
# plt.show()

# #K-fold
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scores = []

# # Loop through each fold
# for train_index, test_index in skf.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#     # Train the model
#     clf.fit(X_train, y_train)
#     # Predict on test set
#     y_pred = clf.predict(X_test)

#     # Evaluate the model
#     scores.append({
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred),
#         'recall': recall_score(y_test, y_pred),
#         'f1': f1_score(y_test, y_pred)
#     })
    
# scores_df = pd.DataFrame(scores)
# average_scores = scores_df.mean()
# print('Average scores after 5 K-fold:', average_scores)

# def create_landslide_susceptibility_map(df, classifier, output_tif_path):
#     df = df.rename(columns={
#         'precipitation august 2': 'precipitation',
#         'soil saturation august 2': 'soil saturation'
#     })
#     X = df.drop([
#         'soil saturation august 10', 'soil saturation august 9', 'soil saturation august 8',
#         'soil saturation august 7', 'precipitation august 10', 'precipitation august 7',
#         'precipitation august 9', 'precipitation august 8', 'x', 'y'
#         # 'x_min', 'y_min', 'x_max', 'y_max', 'landslide'
#     ], axis=1)
    
#     # Predict landslide susceptibility
#     prediction_prob = classifier.predict_proba(X)
#     df['LSM'] = prediction_prob[:, 1]

#     # Create unique coordinates and sort them
#     unique_x = np.unique(df['x'])
#     unique_y = np.unique(df['y'])
#     unique_x.sort()
#     unique_y.sort()

#     # Create a raster array
#     num_rows = len(unique_y)
#     num_cols = len(unique_x)
#     raster_array = np.zeros((num_rows, num_cols), dtype=np.float32)
    
#     # Fill the raster array with LSM values
#     for index, row in df.iterrows():
#         x_index = np.where(unique_x == row['x'])[0][0]
#         y_index = num_rows - 1 - np.where(unique_y == row['y'])[0][0]
#         raster_array[y_index, x_index] = row['LSM']

#     # Define raster metadata
#     pixel_width = 10
#     pixel_height = 10
#     transform = from_origin(unique_x.min(), unique_y.max(), pixel_width, pixel_height)
#     crs = "EPSG:25833"
#     dtype = raster_array.dtype
#     count = 1
#     driver = 'GTiff'

#     with rasterio.open(output_tif_path, 'w', driver=driver, width=num_cols, height=num_rows, 
#                         count=count, dtype=dtype, crs=crs, transform=transform) as dst:
#         dst.write(raster_array, 1)

# df_study_area = df_study_area
# clf = clf
# output_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/susceptibility_map_RF_rain_august_2.tif'
# create_landslide_susceptibility_map(df_study_area, clf, output_path)


#df_filtered = df_study_area[df_study_area['LSM'] >= 0.25]

# PLOTTING THE DF_FILTERED
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

# X_SA = df_study_area.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
# X_SA = df_study_area.drop(['soil saturation august 10', 'soil saturation august 9','soil saturation august 8', 
#                         'soil saturation august 7', 'precipitation august 10', 'precipitation august 7',
#                         'precipitation august 9', 'precipitation august 8', 'x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
# prediction_SA = clf.predict(X_SA)
# prediction_prob=clf.predict_proba(X_SA)
# df_study_area['LSM']= prediction_prob[:,1]

# unique_x = df_study_area['x'].unique()
# unique_y = df_study_area['y'].unique()
# unique_x.sort()
# unique_y.sort()
# num_rows = len(unique_y)
# num_cols = len(unique_x)
# raster_array = np.zeros((num_rows, num_cols), dtype=np.float32)
# min_y = unique_y.min()
# max_y = unique_y.max()

# for index, row in df_study_area.iterrows():
#     x_index = np.where(unique_x == row['x'])[0][0]
#     y_index = num_rows - 1 - np.where(unique_y == row['y'])[0][0]
#     raster_array[y_index, x_index] = row['LSM']


# pixel_width = 10  
# pixel_height = 10 
# transform = from_origin(unique_x.min(), unique_y.max(), pixel_width, pixel_height)
# crs = "EPSG:25833"  
# dtype = raster_array.dtype
# count = 1  
# driver = 'GTiff'  

# output_tif_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/susceptibility_map_LR.tif'

# with rasterio.open(output_tif_path, 'w', driver=driver, width=num_cols, height=num_rows, count=count, dtype=dtype, crs=crs, transform=transform) as dst:
#     dst.write(raster_array, 1)  

#%% predictions and probabilities for each day. also generating susceptibility map. 
#df_1 = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/dataset_ratios/13_percent_ratio_updated.xlsx')
df_1 = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/dataset_ratios/1_5_ratio.xlsx')
X = df_1.drop(['x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
y = df_1[['landslide', 'Day']]
#RF
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) #data frames with train and test data and dates
clf = RandomForestClassifier(max_depth = 5, random_state=42) #initiate model
X_train1 = X_train.drop(['Day'], axis=1)
y_train1 = y_train.drop(['Day'], axis=1)
clf.fit(X_train1, y_train1["landslide"].to_numpy(dtype=int))

features_list = X_train1.columns
plt.figure(figsize=(10, 8))
feature_imp = pd.Series(clf.feature_importances_, index=features_list).sort_values(ascending=False)
feature_imp.plot(kind='bar')
plt.title('Feature Importances')
plt.xticks(rotation=45) 
plt.ylabel('Importance')
plt.tight_layout() 
plt.show()

def predict_for_day(clf, X_test, y_test, day):
    # Filter test data for the specified day
    X_test_day = X_test[X_test['Day'] == day]
    y_test_day = y_test[y_test['Day'] == day]
    
    # Drop the 'Day' column
    X_test_day = X_test_day.drop(['Day'], axis=1)
    y_test_day = y_test_day.drop(['Day'], axis=1)
    
    # Predict
    y_pred_day = clf.predict(X_test_day)
    y_pred_proba_day = clf.predict_proba(X_test_day)[:, 1]
    
    return y_pred_day, y_pred_proba_day, y_test_day

#Predictions and probabilities
for day in [2, 8, 9, 10]:  # Specify the days for prediction
    y_pred_day, y_pred_proba_day, y_test_day = predict_for_day(clf, X_test, y_test, day)
    print(f"Predictions for Day {day}: {y_pred_day}")
    print(f"Probabilities for Day {day}: {y_pred_proba_day}")
    
def plot_roc_curve(y_test, y_scores, day_label):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{day_label} ROC curve (AUC = {auc_score:.2f})')

# Start plotting
plt.figure(figsize=(10, 8))

for day in [8, 9]:  # Days for which to plot ROC curves
    y_pred_day, y_pred_proba_day, y_test_day = predict_for_day(clf, X_test, y_test, day)
    print(f"Predictions for Day {day}: {y_pred_day}")
    print(f"Probabilities for Day {day}: {y_pred_proba_day}")
    print(f"Confusion Matrix for Day {day}:")

    # Plot ROC Curve
    plot_roc_curve(y_test_day, y_pred_proba_day, f'August {day}')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for August 8 and August 9')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()    


def create_landslide_susceptibility_map(df, classifier, output_tif_path):
    df = df.rename(columns={
        'precipitation august 8': 'precipitation',
        'soil saturation august 8': 'soil saturation'
    })
    X = df.drop([
        'soil saturation august 10', 'soil saturation august 9', 'soil saturation august 2',
        'soil saturation august 7', 'precipitation august 10', 'precipitation august 7',
        'precipitation august 9', 'precipitation august 2', 'x', 'y'
        # 'x_min', 'y_min', 'x_max', 'y_max', 'landslide'
    ], axis=1)
    
    # Predict landslide susceptibility
    prediction_prob = classifier.predict_proba(X)
    df['LSM'] = prediction_prob[:, 1]

    # Create unique coordinates and sort them
    unique_x = np.unique(df['x'])
    unique_y = np.unique(df['y'])
    unique_x.sort()
    unique_y.sort()

    # Create a raster array
    num_rows = len(unique_y)
    num_cols = len(unique_x)
    raster_array = np.zeros((num_rows, num_cols), dtype=np.float32)
    
    # Fill the raster array with LSM values
    for index, row in df.iterrows():
        x_index = np.where(unique_x == row['x'])[0][0]
        y_index = num_rows - 1 - np.where(unique_y == row['y'])[0][0]
        raster_array[y_index, x_index] = row['LSM']

    # Define raster metadata
    pixel_width = 10
    pixel_height = 10
    transform = from_origin(unique_x.min(), unique_y.max(), pixel_width, pixel_height)
    crs = "EPSG:25833"
    dtype = raster_array.dtype
    count = 1
    driver = 'GTiff'

    with rasterio.open(output_tif_path, 'w', driver=driver, width=num_cols, height=num_rows, 
                        count=count, dtype=dtype, crs=crs, transform=transform) as dst:
        dst.write(raster_array, 1)

df_study_area = df_study_area
clf = clf
output_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/susceptibility_map_RF_rain_august_8.tif'
create_landslide_susceptibility_map(df_study_area, clf, output_path)
    
# def evaluate_other_days(clf, X_test, y_test, days):
#     for day in days:
#         y_pred_day, y_pred_proba_day, y_test_day = predict_for_day(clf, X_test, y_test, day)
#         print(f"Predictions for Day {day}: {y_pred_day}")
#         print(f"Probabilities for Day {day}: {y_pred_proba_day}")

#         # Calculate accuracy
#         accuracy = accuracy_score(y_test_day, y_pred_day)
#         print(f"Accuracy for Day {day}: {accuracy}")

#         # Calculate confusion matrix and derive TN, FP (specificity calculation)
#         cm = confusion_matrix(y_test_day, y_pred_day, labels=clf.classes_)
#         tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], cm[0][1], 0, 0)  # Handle cases where there are no positives
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 1  # Specificity is trivially 1 if no false positives
#         print(f"Specificity for Day {day}: {specificity}\n")

# # Specify the days for predictions
# non_landslide_days = [2, 7, 10]
# evaluate_other_days(clf, X_test, y_test, non_landslide_days)
#%%

















