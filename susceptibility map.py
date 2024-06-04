# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:05:45 2024

@author: Isak9
"""
#%% libraries
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
import dask_geopandas as ddg
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
#%% RUNNING MODEL AND INVESTIGATING PERFORMANCE
# df_study_area = pd.read_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/study_area_corrected.parquet', engine='fastparquet')
# landslides = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ml_landslide.xlsx')
# read = df_study_area.iloc[:1000]
# # root_cohesion = df_study_area['root cohesion']

# landslides['flow accumulation'] = landslides['flow accumulation'] * 10**4 #to get the right unit, meter squared
# #landslides['slope angle'] = landslides['slope angle'].round(0)
# #landslides['root cohesion'] = landslides['cohesion'].round(0)
# #landslides['root cohesion'] = landslides['cohesion'].replace(-9999, 0)
# landslides['tree type'] = landslides['tree type'].replace(-9999, 0)
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
# #it is only 227 landslides, but that is because we are only counting amount of pixels
# # where landslides happen, and not the amount of landslides. 
# majority = df_study_area[df_study_area['landslide'] == 0]
# minority = df_study_area[df_study_area['landslide'] == 1]

# num_minority_samples = len(minority)
# num_majority_samples_needed = 3 * num_minority_samples
# majority_needed = majority.sample(n=num_majority_samples_needed, random_state=42)
# df_balanced = pd.concat([majority_needed, minority], ignore_index=True)

# #X = df_balanced.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1) #not parameters that decide landslides
# X = df_balanced.drop(['soil saturation august 10', 'soil saturation august 9','soil saturation august 8', 
#                        'precipitation august 10', 'flow accumulation', 'slope aspect',
#                         'precipitation august 9', 'precipitation august 8', 'x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
# y = df_balanced['landslide'] 
# features_list = list(X.columns)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
#%% RF MODEL PERFORMANCE TESTING
# clf = RandomForestClassifier(max_depth = 5, random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_train_pred = clf.predict(X_train)
# y_pred_proba = clf.predict_proba(X_test)[:, 1]

# #TSS
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

# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# auc = roc_auc_score(y_test, y_pred_proba)
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve of all parameters')
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.text(0.6, 0.2, f'AUC = {auc:.2f}', fontsize=12, ha='center')
# plt.show()

# feature_imp = pd.Series(clf.feature_importances_, index=features_list).sort_values(ascending=False)
# print(feature_imp)
# feature_imp.plot.bar()

#Checking for overfitting
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

#%% PREDICTING THE REST OF THE STUDY AREA BASED ON TRAINED RF MODEL
#X_SA = df_study_area.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
# X_SA = df_study_area.drop(['soil saturation august 10', 'soil saturation august 9','soil saturation august 8', 
#                        'soil saturation august 7', 'precipitation august 10', 'precipitation august 7',
#                         'precipitation august 9', 'precipitation august 8', 'x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
# prediction_SA = clf.predict(X_SA)
# prediction_prob=clf.predict_proba(X_SA)
# df_study_area['LSM']= prediction_prob[:,1]
# df_filtered = df_study_area[df_study_area['LSM'] >= 0.25]

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

#%% LR TRAINING MODEL WITH OPTIMIZED PARAMETERS
# logreg = LogisticRegression(max_iter = 1000, random_state=42)
# logreg.fit(X_train, y_train)
# y_pred_lr = logreg.predict(X_test)
# y_pred_proba_lr = logreg.predict_proba(X_test)[:, 1]
# y_train_pred_lr = logreg.predict(X_train)

# #TSS
# conf_matrix = confusion_matrix(y_test, y_pred_lr)
# TP = conf_matrix[1, 1]
# TN = conf_matrix[0, 0]
# FP = conf_matrix[0, 1]
# FN = conf_matrix[1, 0]
# sensitivity = TP / (TP + FN)
# specificity = TN / (TN + FP)
# TSS = sensitivity + specificity - 1
# print("True Skill Statistics (TSS):", TSS)

# print(classification_report(y_test, y_pred_lr))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
# print("Accuracy:", accuracy_score(y_test, y_pred_lr))
# print("Precision", precision_score(y_test, y_pred_lr))

# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr)
# auc = roc_auc_score(y_test, y_pred_proba_lr)
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve of all parameters')
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.text(0.6, 0.2, f'AUC = {auc:.2f}', fontsize=12, ha='center')
# plt.show()

# #Checking for overfitting
# train_accuracy = accuracy_score(y_train, y_train_pred_lr)
# train_precision = precision_score(y_train, y_train_pred_lr)
# train_recall = recall_score(y_train, y_train_pred_lr)
# train_f1 = f1_score(y_train, y_train_pred_lr)
# test_accuracy = accuracy_score(y_test, y_pred_lr)
# test_precision = precision_score(y_test, y_pred_lr)
# test_recall = recall_score(y_test, y_pred_lr)
# test_f1 = f1_score(y_test, y_pred_lr)

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

# coefficients_lr = logreg.coef_[0]
# feature_importance = pd.DataFrame({'Feature LR': X.columns, 'Importance': np.abs(coefficients_lr)})
# feature_importance = feature_importance.sort_values('Importance', ascending=True)
# feature_importance.plot(x='Feature LR', y='Importance', kind='barh', figsize=(10, 6))

# X_SA_lr = df_study_area.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
# X_SA_lr = df_study_area.drop(['soil saturation august 10', 'soil saturation august 9','soil saturation august 8', 
#                          'soil saturation august 7', 'precipitation august 10', 'precipitation august 7',
#                          'precipitation august 9', 'precipitation august 8', 'x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
# prediction_SA_lr = logreg.predict(X_SA_lr)
# prediction_prob=logreg.predict_proba(X_SA_lr)
# df_study_area['LSM']= prediction_prob[:,1]

#%%SENDING Either THE PROBABILITY OF LANDSLIDE FOR RF AND LR TO TIF-FILE FOR ARCGIS
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

# from rasterio.transform import from_origin
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

#%% CREATING HISTOGRAMS OF PERCENTAGE WITHIN LOW, MEDIUM, HIGH LANDSLIDE PROBABILITY
# raster_RF = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/susceptibility maps/susceptibility_map_RF.tif'
# raster_LR = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/susceptibility maps/susceptibility_map_LR.tif'

# with rasterio.open(raster_RF) as src:
#     array_RF = src.read(1).flatten()  
# with rasterio.open(raster_LR) as src:
#     array_LR = src.read(1).flatten()  
    
# bins = [0, 0.25, 0.5, 0.75, 1]
# hist_RF, _ = np.histogram(array_RF, bins=bins)
# hist_LR, _ = np.histogram(array_LR, bins=bins)
# hist_percentage_RF = (hist_RF / hist_RF.sum()) * 100  
# hist_percentage_LR = (hist_LR / hist_LR.sum()) * 100  

# plt.figure(figsize=(10, 6))
# bar_width = 0.4
# r1 = np.arange(len(hist_percentage_RF))
# r2 = [x + bar_width for x in r1]
# plt.bar(r1, hist_percentage_RF, width=bar_width, color='green', edgecolor='black', label='Random Forest')
# plt.bar(r2, hist_percentage_LR, width=bar_width, color='red', edgecolor='black', label='Logistic Regression')
# labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1']
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xticks([r + bar_width/2 for r in range(len(hist_percentage_RF))], labels, fontsize =12)
# plt.xlabel('Landslide Probability', fontsize=12)
# plt.ylabel('Area Percentage [%]', fontsize=12)
# #plt.title('Histogram of Landslide Probabilities')
# plt.legend(fontsize=12)
# plt.show()

#%% Investigating the probability assigned by models on actual landslide pixels. 

# Define paths to the Excel file and raster files
# excel_file = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/mergeddf_master.xlsx'
# raster_files = {
#     'RF': 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/susceptibility maps/susceptibility_map_RF.tif',
#     'LR': 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/susceptibility maps/susceptibility_map_LR.tif'
# }

# # Load the Excel file
# df = pd.read_excel(excel_file)

# # Initialize a DataFrame to store the results
# results_list = []

# # Extract the probability values for each point in the Excel file
# for key in raster_files.keys():
#     with rasterio.open(raster_files[key]) as src:
#         coords = [(x, y) for x, y in zip(df['x_utm'], df['y_utm'])]
#         values = [val[0] for val in src.sample(coords)]
#         # Create a DataFrame with these values and the key
#         temp_df = pd.DataFrame({key: values}, index=df.index)
#         results_list.append(temp_df)

# # Concatenate the results into a single DataFrame
# results_df = pd.concat(results_list, axis=1)

# # Define the bins
# bins = [0, 0.25, 0.5, 0.75, 1]
# bin_labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1']

# # Initialize a DataFrame to store the binned counts
# binned_counts = pd.DataFrame()

# # Categorize the data and count the occurrences in each bin for each raster
# for key in results_df.columns:
#     binned_counts[key] = pd.cut(results_df[key], bins=bins, labels=bin_labels).value_counts(sort=False)

# # Plot the histogram
# ax = binned_counts.plot(kind='bar', figsize=(10, 6))

# # Set font size for all text elements
# plt.xlabel('Probability Bins', fontsize=12)
# plt.ylabel('Number of Landslides', fontsize=12)
# plt.title('Histogram of Landslide Probabilities for Different Rasters', fontsize=12)
# plt.xticks(rotation=0, fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(title='Rasters', fontsize=12, title_fontsize=12)
# plt.grid(True)

# # Adjust layout
# plt.tight_layout()

# # Show the plot
# plt.show()
#%%





























