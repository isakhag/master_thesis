# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:46:47 2024

@author: Isak9
"""
#%% imported libraries
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
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#%% 
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

majority = df_study_area[df_study_area['landslide'] == 0]
minority = df_study_area[df_study_area['landslide'] == 1]

num_minority_samples = len(minority)
num_majority_samples_needed = 2 * num_minority_samples
majority_needed = majority.sample(n=num_majority_samples_needed, random_state=42)
df_balanced = pd.concat([majority_needed, minority], ignore_index=True)

X = df_balanced.drop(['x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
y = df_balanced['landslide']
features_list = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
#%%Initializing RF and LR models, checking prec, acc, auc, roc 
# clf = RandomForestClassifier(max_depth = 5, random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_pred_proba = clf.predict_proba(X_test)[:, 1]
# #CHECK FOR OVERFITTING
# y_train_pred = clf.predict(X_train)

logreg = LogisticRegression(max_iter = 1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
y_pred_proba_lr = logreg.predict_proba(X_test)[:, 1]
y_train_pred_lr = logreg.predict(X_train)

# logreg = LogisticRegression(max_iter=1000, random_state=42)
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],  
#     'penalty': ['l1', 'l2'],  
# }

# grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)
# best_model = LogisticRegression(C=grid_search.best_params_['C'], penalty=grid_search.best_params_['penalty'])
# best_model.fit(X_train, y_train)
# y_pred_lr = best_model.predict(X_test)
# y_pred_proba_lr = best_model.predict_proba(X_test)[:, 1]
# y_train_pred_lr = best_model.predict(X_train)

# #CHECK FOR OVERFITTING, SEE HOW THE MODEL DOES ON TRAINED DATA COMPARED TO TEST DATA
# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_precision = precision_score(y_train, y_train_pred)
# train_recall = recall_score(y_train, y_train_pred)
# train_f1 = f1_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_pred)
# test_precision = precision_score(y_test, y_pred)
# test_recall = recall_score(y_test, y_pred)
# test_f1 = f1_score(y_test, y_pred)

train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
train_precision_lr = precision_score(y_train, y_train_pred_lr)
train_recall_lr = recall_score(y_train, y_train_pred_lr)
train_f1_lr = f1_score(y_train, y_train_pred_lr)
test_accuracy_lr = accuracy_score(y_test, y_pred_lr)
test_precision_lr = precision_score(y_test, y_pred_lr)
test_recall_lr = recall_score(y_test, y_pred_lr)
test_f1_lr = f1_score(y_test, y_pred_lr)

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

# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# auc = roc_auc_score(y_test, y_pred_proba)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_proba_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

plt.figure(figsize=(8, 8))
#plt.plot(fpr, tpr, color='green', lw=2, label=f'RF ROC curve (AUC = {auc:.2f})')
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

# print("Confusion Matrix RF:\n", confusion_matrix(y_test, y_pred))

print("Confusion Matrix LR:\n", confusion_matrix(y_test, y_pred_lr))
#%%CHECKING HOW THE MODELS DO WITH SMOTE AND CORRESPONDING PERFORMANCE
# counter = Counter(y_train)
# print('Before', counter)
# smt = SMOTE(sampling_strategy=1, random_state=42)
# X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
# counter = Counter(y_train_res)
# print('After', counter)

# clf = RandomForestClassifier(max_depth=5, random_state=42)
# clf.fit(X_train_res, y_train_res)
# y_pred = clf.predict(X_test)
# y_pred_proba = clf.predict_proba(X_test)[:, 1]
# #CHECK FOR OVERFITTING
# y_train_pred = clf.predict(X_train_res)

# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression(random_state=42)
# logreg.fit(X_train_res, y_train_res)
# y_pred_lr = logreg.predict(X_test)
# y_pred_proba_lr = logreg.predict_proba(X_test)[:, 1]
# y_train_pred_lr = logreg.predict(X_train_res)

# #CHECK FOR OVERFITTING, SEE HOW THE MODEL DOES ON TRAINED DATA COMPARED TO TEST DATA
# train_accuracy = accuracy_score(y_train_res, y_train_pred)
# train_precision = precision_score(y_train_res, y_train_pred)
# train_recall = recall_score(y_train_res, y_train_pred)
# train_f1 = f1_score(y_train_res, y_train_pred)
# # For regression models, you can use mean_squared_error:
# #train_mse = mean_squared_error(y_train, y_train_pred_lr)
# # Calculate metrics for validation/test data
# test_accuracy = accuracy_score(y_test, y_pred)
# test_precision = precision_score(y_test, y_pred)
# test_recall = recall_score(y_test, y_pred)
# test_f1 = f1_score(y_test, y_pred)
# #test_mse = mean_squared_error(y_test, y_pred_lr)

# train_accuracy_lr = accuracy_score(y_train_res, y_train_pred_lr)
# train_precision_lr = precision_score(y_train_res, y_train_pred_lr)
# train_recall_lr = recall_score(y_train_res, y_train_pred_lr)
# train_f1_lr = f1_score(y_train_res, y_train_pred_lr)

# test_accuracy_lr = accuracy_score(y_test, y_pred_lr)
# test_precision_lr = precision_score(y_test, y_pred_lr)
# test_recall_lr = recall_score(y_test, y_pred_lr)
# test_f1_lr = f1_score(y_test, y_pred_lr)

# print("Training Data Metrics RF:")
# print(f"Accuracy: {train_accuracy}")
# print(f"Precision: {train_precision}")
# print(f"Recall: {train_recall}")
# print(f"F1 Score: {train_f1}")
# #print(f"Mean Squared Error: {train_mse}")

# print("\nValidation/Test Data Metrics RF:")
# print(f"Accuracy: {test_accuracy}")
# print(f"Precision: {test_precision}")
# print(f"Recall: {test_recall}")
# print(f"F1 Score: {test_f1}")
# #print(f"Mean Squared Error: {test_mse}")

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

# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# auc = roc_auc_score(y_test, y_pred_proba)
# fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_proba_lr)
# auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='green', lw=2, label=f'RF ROC curve (AUC = {auc:.2f})')
# plt.plot(fpr_lr, tpr_lr, color='red', lw=2, label=f'LR ROC curve (AUC = {auc_lr:.2f})')
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

# print('Confusion Matrix RF:', classification_report(y_test, y_pred))
# print("Confusion Matrix RF:\n", confusion_matrix(y_test, y_pred))

# print('Confusion Matrix LR:', classification_report(y_test, y_pred_lr))
# print("Confusion Matrix LR:\n", confusion_matrix(y_test, y_pred_lr))

#%% HYPERPARAMETER TUNING OF RANDOM FOREST MODEL WITH BAYESIAN OPTIMIZATION
# search_spaces = {
#     'max_depth': Integer(3, 5),
#     'n_estimators': Integer(10, 130),
# }

# rf = RandomForestClassifier(random_state=42)

# opt = BayesSearchCV(
#     estimator=rf,
#     search_spaces=search_spaces,
#     n_iter=30,  # Number of iterations
#     cv=5,       # 5-fold cross-validation
#     n_jobs=-1,  # Use all available cores
#     return_train_score=True,
#     random_state=42
# )

# opt.fit(X_train, y_train)
# print("Best parameters:", opt.best_params_)
# print("Best score:", opt.best_score_)

# clf = RandomForestClassifier(**opt.best_params_, random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_pred_proba = clf.predict_proba(X_test)[:, 1]
# #CHECK FOR OVERFITTING
# y_train_pred = clf.predict(X_train)

# clf = RandomForestClassifier(max_depth=5, random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_pred_proba = clf.predict_proba(X_test)[:, 1]
# #CHECK FOR OVERFITTING
# y_train_pred = clf.predict(X_train)

# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_precision = precision_score(y_train, y_train_pred)
# train_recall = recall_score(y_train, y_train_pred)
# train_f1 = f1_score(y_train, y_train_pred)

# # Calculate metrics for validation/test data
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

# print("Confusion Matrix RF:\n", confusion_matrix(y_test, y_pred))

#%%HYPERPARAMETER TUNING OF LOGISTIC REGRESSION GRID SEARCH
# logreg = LogisticRegression(max_iter=1000, random_state=42)
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],  
#     'penalty': ['l1', 'l2'],  
# }

# grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)
# best_model = LogisticRegression(C=grid_search.best_params_['C'], penalty=grid_search.best_params_['penalty'])
# best_model.fit(X_train, y_train)
# y_pred_lr = best_model.predict(X_test)
# y_pred_proba_lr = best_model.predict_proba(X_test)[:, 1]
# y_train_pred_lr = best_model.predict(X_train)

# train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
# train_precision_lr = precision_score(y_train, y_train_pred_lr)
# train_recall_lr = recall_score(y_train, y_train_pred_lr)
# train_f1_lr = f1_score(y_train, y_train_pred_lr)
# test_accuracy_lr = accuracy_score(y_test, y_pred_lr)
# test_precision_lr = precision_score(y_test, y_pred_lr)
# test_recall_lr = recall_score(y_test, y_pred_lr)
# test_f1_lr = f1_score(y_test, y_pred_lr)

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

# print("Confusion Matrix LR:\n", confusion_matrix(y_test, y_pred_lr))

#%%

















