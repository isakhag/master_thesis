# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:08:39 2024

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression

#%% creating dataframes for ml models
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

df_study_area['slope angle'] =

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
#%% RF
num_minority_samples = len(minority)
num_majority_samples_needed = 3 * num_minority_samples
majority_needed = majority.sample(n=num_majority_samples_needed, random_state=42)
df_balanced = pd.concat([majority_needed, minority], ignore_index=True)

#X = df_balanced.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1) #not parameters that decide landslides
X = df_balanced.drop(['flow accumulation', 'slope aspect','x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
y = df_balanced['landslide']
features_list = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf = RandomForestClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
#CHECK FOR OVERFITTING
y_train_pred = clf.predict(X_train)

#LR
num_majority_needed_lr = 2 * num_minority_samples
majority_needed_lr = majority.sample(n=num_majority_needed_lr, random_state=42)
df_balanced_lr = pd.concat([majority_needed_lr, minority], ignore_index=True)
X_lr = df_balanced_lr.drop(['x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
y_lr = df_balanced_lr['landslide']
features_list_lr = X_lr.columns
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.3, random_state=42, stratify=y_lr)

logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_lr, y_train_lr)
y_pred_lr = logreg.predict(X_test_lr)
y_pred_proba_lr = logreg.predict_proba(X_test_lr)[:, 1]
y_train_pred_lr = logreg.predict(X_train_lr)

# coefficients = logreg.coef_[0]
# feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
# feature_importance = feature_importance.sort_values('Importance', ascending=True)
# feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))

print('Confusion Matrix RF:', classification_report(y_test, y_pred))
print("Confusion Matrix RF:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))

print('Confusion Matrix LR:', classification_report(y_test_lr, y_pred_lr))
print("Confusion Matrix LR:\n", confusion_matrix(y_test_lr, y_pred_lr))
print("Accuracy:", accuracy_score(y_test_lr, y_pred_lr))
print("Precision", precision_score(y_test_lr, y_pred_lr))

#%%CHECK FOR OVERFITTING, SEE HOW THE MODEL DOES ON TRAINED DATA COMPARED TO TEST DATA
# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_precision = precision_score(y_train, y_train_pred)
# train_recall = recall_score(y_train, y_train_pred)
# train_f1 = f1_score(y_train, y_train_pred)
# # For regression models, you can use mean_squared_error:
# train_mse = mean_squared_error(y_train, y_train_pred_lr)
# # Calculate metrics for validation/test data
# test_accuracy = accuracy_score(y_test, y_pred)
# test_precision = precision_score(y_test, y_pred)
# test_recall = recall_score(y_test, y_pred)
# test_f1 = f1_score(y_test, y_pred)
# test_mse = mean_squared_error(y_test, y_pred_lr)

# train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
# train_precision_lr = precision_score(y_train, y_train_pred_lr)
# train_recall_lr = recall_score(y_train, y_train_pred_lr)
# train_f1_lr = f1_score(y_train, y_train_pred_lr)

# test_accuracy_lr = accuracy_score(y_test, y_pred_lr)
# test_precision_lr = precision_score(y_test, y_pred_lr)
# test_recall_lr = recall_score(y_test, y_pred_lr)
# test_f1_lr = f1_score(y_test, y_pred_lr)

# print("Training Data Metrics RF:")
# print(f"Accuracy: {train_accuracy}")
# print(f"Precision: {train_precision}")
# print(f"Recall: {train_recall}")
# print(f"F1 Score: {train_f1}")
# print(f"Mean Squared Error: {train_mse}")

# print("\nValidation/Test Data Metrics RF:")
# print(f"Accuracy: {test_accuracy}")
# print(f"Precision: {test_precision}")
# print(f"Recall: {test_recall}")
# print(f"F1 Score: {test_f1}")
# print(f"Mean Squared Error: {test_mse}")

# print("Training Data Metrics LR:")
# print(f"Accuracy: {train_accuracy_lr}")
# print(f"Precision: {train_precision_lr}")
# print(f"Recall: {train_recall_lr}")
# print(f"F1 Score: {train_f1_lr}")

# print("\nValidation/Test Data Metrics LR:")
# print(f"Accuracy: {test_accuracy_lr}")
# print(f"Precision: {test_precision_lr}")
# print(f"Recall: {test_recall_lr}")
# print(f"F1 Score: {test_f1_lr}")

#%% VIF
X_with_const = add_constant(majority_needed.drop(['flow accumulation', 'slope aspect', 'x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1))
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print('VIF data: ', vif_data)

#IG
info_gains = mutual_info_classif(X_train, y_train)
feature_name_map = {i: feature_name for i, feature_name in enumerate(X_train.columns)}
for i, info_gain in enumerate(info_gains):
    feature_name = feature_name_map[i]
    print(f"'{feature_name}': Information Gain = {info_gain}")

#TSS
conf_matrix = confusion_matrix(y_test, y_pred)
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
TSS = sensitivity + specificity - 1
print("True Skill Statistics (TSS):", TSS)

# #ROC CURVE WITH CORRESPONDING AUC VALUE
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test_lr, y_pred_proba_lr)
auc_lr = roc_auc_score(y_test_lr, y_pred_proba_lr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='green', lw=2, label=f'RF ROC curve (AUC = {auc:.2f})')
plt.plot(fpr_lr, tpr_lr, color='red', lw=2, label=f'LR ROC curve (AUC = {auc_lr:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1])
plt.xlabel('False Positive Rate', fontsize = 12)
plt.ylabel('True Positive Rate', fontsize = 12)
plt.title('ROC Curve', fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend(loc='lower right', fontsize = 14)
plt.grid(True)
plt.show()

#VISUALIZING THE DECISION TREES IN THE RANDOM FOREST
# num_trees_to_visualize = 5

# for i in range(min(num_trees_to_visualize, len(clf.estimators_))):
#     tree = clf.estimators_[i]
#     plt.figure(figsize=(20, 10))
#     plot_tree(tree, feature_names=features_list, filled=True, rounded=True, class_names=['Not Landslide', 'Landslide'])
#     plt.title(f'Tree {i+1}')
#     plt.show()

#MOST IMPORTANT FEATURES RF
# plt.figure()
# feature_imp = pd.Series(clf.feature_importances_, index=features_list).sort_values(ascending=False)
# feature_imp.plot.bar()
# plt.xticks(rotation=45) 
# plt.tight_layout() 
# plt.show()  

#%%PREDICTING THE REST OF THE STUDY AREA
# X_SA = df_study_area.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
# prediction_SA = logreg.predict(X_SA)
# prediction_prob=logreg.predict_proba(X_SA)
# df_study_area['LSM']= prediction_prob[:,1]
# df_filtered = df_study_area[df_study_area['LSM'] >= 0.5]

#PLOTTING THE DF_FILTERED
# colors = []
# for prob in df_filtered['LSM']:
#     if 0.5 <= prob <= 0.75:
#         colors.append('yellow')
#     elif 0.75 <= prob <= 1:
#         colors.append('red')         
# plt.figure(figsize=(10, 8))
# plt.scatter(df_filtered['x'], df_filtered['y'], c=colors, marker='.')

# plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='0.5-0.75 PoL'),
#                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='0.75-1 PoL')])

# plt.title('Landslide Susceptibility Map')
# plt.xlabel('East [m]')
# plt.ylabel('North [m]')
# plt.show()

#SENDING THE PREDICTION OF THE TOTAL STUDY AREA TO A TIF-FILE
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
#%%
