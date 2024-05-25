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
# df_study_area = pd.read_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/study_area.parquet', engine='fastparquet')
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
#%% RF
#X = df_balanced.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1) #not parameters that decide landslides
X = df_balanced.drop(['soil saturation august 8', 'soil saturation august 9', 'soil saturation august 10','soil saturation august 7',
                      'precipitation august 8','precipitation august 9', 'precipitation august 7','precipitation august 10',
                        'x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)

y = df_balanced['landslide']
features_list = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf = RandomForestClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
#CHECK FOR OVERFITTING
y_train_pred = clf.predict(X_train)
print("Confusion Matrix RF:\n", confusion_matrix(y_test, y_pred))

#Feature importance RF
# features_list = X.columns
# plt.figure()
# feature_imp = pd.Series(clf.feature_importances_, index=features_list).sort_values(ascending=False)
# feature_imp.plot.bar()
# plt.xticks(rotation=45) 
# plt.tight_layout() 
# plt.show()  

# search_spaces = {
#     'max_depth': Integer(3, 10),
#     'n_estimators': Integer(10, 130),
# }

# rf = RandomForestClassifier(random_state=42)

# opt = BayesSearchCV(
#     estimator=rf,
#     search_spaces=search_spaces,
#     n_iter=30,  
#     cv=5,       
#     n_jobs=-1, 
#     return_train_score=True,
#     random_state=42
# )

# opt.fit(X_train, y_train)
# print("Best parameters RF:", opt.best_params_)
# print("Best score RF:", opt.best_score_)

#%% LR
#X_lr = df_balanced.drop(['x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
X_lr = df_balanced.drop(['soil saturation august 8', 'soil saturation august 9', 'soil saturation august 10', 'soil saturation august 7',
                      'precipitation august 8', 'precipitation august 9', 'precipitation august 7', 'precipitation august 10',
                         'x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)

y_lr = df_balanced['landslide']
scaler = StandardScaler()
X_lr_scaled = scaler.fit_transform(X_lr)
X_train_lr_scaled, X_test_lr_scaled, y_train_lr, y_test_lr = train_test_split(X_lr_scaled, y_lr, test_size=0.3, random_state=42, stratify=y_lr)
logreg = LogisticRegression(max_iter=1000, random_state=42)
#, C= 0.1, penalty= 'l2'
logreg.fit(X_train_lr_scaled, y_train_lr)
y_pred_lr = logreg.predict(X_test_lr_scaled)
y_pred_proba_lr = logreg.predict_proba(X_test_lr_scaled)[:, 1]
y_train_pred_lr = logreg.predict(X_train_lr_scaled)

coefficients_lr = logreg.coef_[0]
feature_importance = pd.DataFrame({'Feature LR': X_lr.columns, 'Importance': np.abs(coefficients_lr)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature LR', y='Importance', kind='barh', figsize=(10, 6))
print("Confusion Matrix LR:\n", confusion_matrix(y_test_lr, y_pred_lr))

# X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.3, random_state=42, stratify=y_lr)
# logreg = LogisticRegression(max_iter = 1000, random_state=42)
# logreg.fit(X_train_lr, y_train_lr)
# y_pred_lr = logreg.predict(X_test_lr)
# y_pred_proba_lr = logreg.predict_proba(X_test_lr)[:, 1]
# y_train_pred_lr = logreg.predict(X_train_lr)
# coefficients_lr = logreg.coef_[0]
# feature_importance = pd.DataFrame({'Feature LR': X_lr.columns, 'Importance': np.abs(coefficients_lr)})
# feature_importance = feature_importance.sort_values('Importance', ascending=True)
# feature_importance.plot(x='Feature LR', y='Importance', kind='barh', figsize=(10, 6))
# print("Confusion Matrix LR:\n", confusion_matrix(y_test, y_pred_lr))

# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear']  
# }
# # hyperparameter tuning
# grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train_lr_scaled, y_train_lr)
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)

#%%CHECK FOR OVERFITTING, SEE HOW THE MODEL DOES ON TRAINED DATA COMPARED TO TEST DATA
#RF
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Calculate metrics for validation/test data
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

#LR
train_accuracy_lr = accuracy_score(y_train_lr, y_train_pred_lr)
train_precision_lr = precision_score(y_train_lr, y_train_pred_lr)
train_recall_lr = recall_score(y_train_lr, y_train_pred_lr)
train_f1_lr = f1_score(y_train_lr, y_train_pred_lr)

test_accuracy_lr = accuracy_score(y_test_lr, y_pred_lr)
test_precision_lr = precision_score(y_test_lr, y_pred_lr)
test_recall_lr = recall_score(y_test_lr, y_pred_lr)
test_f1_lr = f1_score(y_test_lr, y_pred_lr)

print("Training Data Metrics RF:")
print(f"Accuracy: {train_accuracy}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
print(f"F1 Score: {train_f1}")

print("\nValidation/Test Data Metrics RF:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")

print("Training Data Metrics LR:")
print(f"Accuracy: {train_accuracy_lr}")
print(f"Precision: {train_precision_lr}")
print(f"Recall: {train_recall_lr}")
print(f"F1 Score: {train_f1_lr}")

print("\nValidation/Test Data Metrics LR:")
print(f"Accuracy: {test_accuracy_lr}")
print(f"Precision: {test_precision_lr}")
print(f"Recall: {test_recall_lr}")
print(f"F1 Score: {test_f1_lr}")

#%% performance
#X_with_const = add_constant(majority_needed.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1))
X_with_const = df_balanced.drop(['soil saturation august 8', 'soil saturation august 9', 'soil saturation august 10', 'soil saturation august 7',
                      'precipitation august 8', 'precipitation august 9', 'precipitation august 7','precipitation august 10',
                      'x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)

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

#TSS RF
conf_matrix = confusion_matrix(y_test, y_pred)
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
TSS = sensitivity + specificity - 1
print("True Skill Statistics (TSS):", TSS)

#ROC CURVE WITH CORRESPONDING AUC VALUE
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

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = []

# Cross-validation loop
for train_index, test_index in skf.split(X_lr_scaled, y_lr):
    X_train_lr_scaled, X_test_lr_scaled = X_lr_scaled[train_index], X_lr_scaled[test_index]
    y_train_lr, y_test_lr = y_lr.iloc[train_index], y_lr.iloc[test_index]

    # Train the model
    logreg.fit(X_train_lr_scaled, y_train_lr)
    # Predict on test set
    y_pred_lr = logreg.predict(X_test_lr_scaled)

    # Evaluate the model
    scores.append({
        'accuracy': accuracy_score(y_test_lr, y_pred_lr),
        'precision': precision_score(y_test_lr, y_pred_lr),
        'recall': recall_score(y_test_lr, y_pred_lr),
        'f1': f1_score(y_test_lr, y_pred_lr)
    })

# Calculate average scores across all folds
scores_df = pd.DataFrame(scores)
average_scores = scores_df.mean()
print('Average scores after 10 K-fold:', average_scores)

#VISUALIZING THE DECISION TREES IN THE RANDOM FOREST
# num_trees_to_visualize = 5

# for i in range(min(num_trees_to_visualize, len(clf.estimators_))):
#     tree = clf.estimators_[i]
#     plt.figure(figsize=(20, 10))
#     plot_tree(tree, feature_names=features_list, filled=True, rounded=True, class_names=['Not Landslide', 'Landslide'])
#     plt.title(f'Tree {i+1}')
#     plt.show()

