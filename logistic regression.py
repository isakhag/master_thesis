# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:08:39 2024

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import mutual_info_classif

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
majority_needed = majority.sample(n=num_majority_samples_needed, random_state=16)
df_balanced = pd.concat([majority_needed, minority], ignore_index=True)

X = df_balanced.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1) #not parameters that decide landslides
# X = df_balanced.drop(['slope aspect', 'flow accumulation', 'tree type', 'soil', 'cohesion','x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)
y = df_balanced['landslide'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

X_with_const = add_constant(majority_needed.drop(['x', 'y', 'x_min','y_min','x_max','y_max' ,'landslide'], axis=1))
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print('VIF data: ', vif_data)

# X = df_balanced.drop(['slope aspect', 'flow accumulation', 'tree type', 'soil', 'cohesion','x', 'y','x_min','y_min','x_max','y_max' ,'landslide'], axis=1)

y = df_balanced['landslide'] 
features_list = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#IG
info_gains = mutual_info_classif(X_train, y_train)
feature_name_map = {i: feature_name for i, feature_name in enumerate(X_train.columns)}
for i, info_gain in enumerate(info_gains):
    feature_name = feature_name_map[i]
    print(f"'{feature_name}': Information Gain = {info_gain}")
    
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

#PERFORMANCE CONSTANTS AND CONFUSION MATRIX
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))

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

#ROC CURVE WITH CORRESPONDING AUC VALUE
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1])
plt.xlabel('False Positive Rate',fontsize = 12)
plt.ylabel('True Positive Rate', fontsize = 12)
plt.title('ROC Curve', fontsize = 14)
plt.legend(loc='lower right', fontsize = 14)
plt.grid(True)
plt.text(0.6, 0.2, f'AUC = {auc:.2f}', fontsize=12, ha='center')
plt.show()

#MOST IMPORTANT FEATURES
plt.figure()
feature_imp = pd.Series(clf.feature_importances_, index=features_list).sort_values(ascending=False)
feature_imp.plot.bar()
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.show()  





