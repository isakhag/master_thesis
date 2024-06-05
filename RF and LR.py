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
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skopt.space import Integer, Categorical, Real
from skopt import BayesSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
#%% creating dataframes for ml models
# df_study_area = pd.read_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/study_area_corrected.parquet', engine='fastparquet')
#landslides = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ml_landslide.xlsx')
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
# num_majority_samples_needed = 3 * num_minority_samples
# majority_needed = majority.sample(n=num_majority_samples_needed, random_state=42)
# df_balanced = pd.concat([majority_needed, minority], ignore_index=True)

# df_balanced.to_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/minority_majority/df_balanced.xlsx', index=False)
#%% reading in the df_balanced dataframe
df_balanced = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/minority_majority/df_balanced.xlsx')
#%%
X = df_balanced.drop(['soil saturation august 8', 'soil saturation august 9', 'soil saturation august 10', 'soil saturation august 7',
                      'precipitation august 8', 'precipitation august 9', 'precipitation august 7', 'precipitation august 10',
                      'x', 'y', 'x_min', 'y_min', 'x_max', 'y_max', 'landslide'], axis=1)
y = df_balanced['landslide']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ensure the same train-test split for both models
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

### Random Forest Evaluation ###
# Train Random Forest
clf_rf = RandomForestClassifier(max_depth=5, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
y_pred_proba_rf = clf_rf.predict_proba(X_test)[:, 1]

print("Confusion Matrix RF (Single Split):\n", confusion_matrix(y_test, y_pred_rf))

# Evaluate model performance on the test set for RF
print("Single Split Metrics RF:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf)}")
print(f"Recall: {recall_score(y_test, y_pred_rf)}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf)}")

# Evaluate model performance on the training set for RF
y_train_pred_rf = clf_rf.predict(X_train)
print("\nTraining Data Metrics RF:")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_rf)}")
print(f"Precision: {precision_score(y_train, y_train_pred_rf)}")
print(f"Recall: {recall_score(y_train, y_train_pred_rf)}")
print(f"F1 Score: {f1_score(y_train, y_train_pred_rf)}")

# Cross-validation for Random Forest
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores_rf = []

for train_index, test_index in skf.split(X_scaled, y):
    X_train_cv, X_test_cv = X_scaled[train_index], X_scaled[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    clf_rf_cv = RandomForestClassifier(max_depth=5, random_state=42)
    clf_rf_cv.fit(X_train_cv, y_train_cv)
    
    # Predict on the test set
    y_pred_cv_rf = clf_rf_cv.predict(X_test_cv)
    y_pred_proba_cv_rf = clf_rf_cv.predict_proba(X_test_cv)[:, 1]

    # Evaluate the model
    cv_scores_rf.append({
        'accuracy': accuracy_score(y_test_cv, y_pred_cv_rf),
        'precision': precision_score(y_test_cv, y_pred_cv_rf),
        'recall': recall_score(y_test_cv, y_pred_cv_rf),
        'f1': f1_score(y_test_cv, y_pred_cv_rf)
    })

# Calculate average cross-validation scores for Random Forest
cv_scores_df_rf = pd.DataFrame(cv_scores_rf)
average_cv_scores_rf = cv_scores_df_rf.mean()
print('\nAverage scores after 10 K-fold cross-validation RF:')
print(average_cv_scores_rf)

### Logistic Regression Evaluation ###
# Train Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
y_pred_proba_lr = logreg.predict_proba(X_test)[:, 1]

print("Confusion Matrix LR (Single Split):\n", confusion_matrix(y_test, y_pred_lr))

# Evaluate model performance on the test set for LR
print("Single Split Metrics LR:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print(f"Precision: {precision_score(y_test, y_pred_lr)}")
print(f"Recall: {recall_score(y_test, y_pred_lr)}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr)}")

# Evaluate model performance on the training set for LR
y_train_pred_lr = logreg.predict(X_train)
print("\nTraining Data Metrics LR:")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_lr)}")
print(f"Precision: {precision_score(y_train, y_train_pred_lr)}")
print(f"Recall: {recall_score(y_train, y_train_pred_lr)}")
print(f"F1 Score: {f1_score(y_train, y_train_pred_lr)}")

# Cross-validation for Logistic Regression
cv_scores_lr = []

for train_index, test_index in skf.split(X_scaled, y):
    X_train_cv, X_test_cv = X_scaled[train_index], X_scaled[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    logreg_cv = LogisticRegression(max_iter=1000, random_state=42)
    logreg_cv.fit(X_train_cv, y_train_cv)
    
    # Predict on the test set
    y_pred_cv_lr = logreg_cv.predict(X_test_cv)
    y_pred_proba_cv_lr = logreg_cv.predict_proba(X_test_cv)[:, 1]

    # Evaluate the model
    cv_scores_lr.append({
        'accuracy': accuracy_score(y_test_cv, y_pred_cv_lr),
        'precision': precision_score(y_test_cv, y_pred_cv_lr),
        'recall': recall_score(y_test_cv, y_pred_cv_lr),
        'f1': f1_score(y_test_cv, y_pred_cv_lr)
    })

# Calculate average cross-validation scores for Logistic Regression
cv_scores_df_lr = pd.DataFrame(cv_scores_lr)
average_cv_scores_lr = cv_scores_df_lr.mean()
print('\nAverage scores after 10 K-fold cross-validation LR:')
print(average_cv_scores_lr)

### ROC Curve Generation ###
# Generate ROC curve data
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_proba_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_proba_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

# Plot the ROC curves
plt.figure(figsize=(8, 8))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'RF ROC curve (AUC = {auc_rf:.2f})')
plt.plot(fpr_lr, tpr_lr, color='red', lw=2, label=f'LR ROC curve (AUC = {auc_lr:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower right', fontsize=14)
plt.grid(True)
plt.show()

#%% Feature importance, hyperparameter tuning, VIF
features_list = X.columns
plt.figure()
feature_imp = pd.Series(clf_rf.feature_importances_, index=features_list).sort_values(ascending=False)
feature_imp.plot.bar()
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.show()

# Logistic Regression feature importance
coefficients_lr = logreg.coef_[0]
feature_importance_lr = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients_lr)})
feature_importance_lr = feature_importance_lr.sort_values('Importance', ascending=True)
feature_importance_lr.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
plt.tight_layout()
plt.show()

### Hyperparameter Tuning ###
# Random Forest hyperparameter tuning
search_spaces_rf = {
    'max_depth': Integer(3, 10),
    'n_estimators': Integer(10, 130),
}

rf = RandomForestClassifier(random_state=42)

opt_rf = BayesSearchCV(
    estimator=rf,
    search_spaces=search_spaces_rf,
    n_iter=30,  
    cv=5,       
    n_jobs=-1, 
    return_train_score=True,
    random_state=42
)

opt_rf.fit(X_train, y_train)
print("Best parameters RF:", opt_rf.best_params_)
print("Best score RF:", opt_rf.best_score_)

# Logistic Regression hyperparameter tuning
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search_lr = GridSearchCV(logreg, param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)
print("Best parameters LR:", grid_search_lr.best_params_)
print("Best score LR:", grid_search_lr.best_score_)

### VIF Calculation ###
# Add constant to the features for VIF calculation
X_with_const = add_constant(X)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print('VIF data: ', vif_data)


#%%PREDICTING THE STUDY AREA
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
















