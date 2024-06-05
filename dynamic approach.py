# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:25:47 2024

@author: Isak9
"""

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
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
#%% creating subsets of total study area to train model.
#df_study_area = pd.read_parquet('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/study_area.parquet', engine='fastparquet')
# landslides = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ml_landslide.xlsx')
# #read = df_study_area.iloc[:1000]

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
# df.to_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/dataset_ratios/dynamic_ratio.xlsx', index=False)
# %% implementing SMOTE in order to get more landslide points. 
df_1 = pd.read_excel('C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/dataset_ratios/dynamic_ratio.xlsx')
X = df_1.drop(['x', 'y', 'x_min', 'y_min', 'x_max', 'y_max', 'landslide', 'soil'], axis=1)
#'flow accumulation', 'soil', 'slope aspect'
y = df_1[['landslide', 'Day']]
day_counts = df_1['Day'].value_counts()
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y['landslide'])

count_landslide = y_train['landslide'].sum()
count_non_landslide = len(y_train) - count_landslide  # or y_train['landslide'].count() - count_landslide

# Define the number of synthetic samples you want to add
additional_samples = 170

# Calculate the new desired count for the minority class
new_minority_count = count_landslide + additional_samples

# Calculate the ratio for SMOTE
desired_ratio = new_minority_count / count_non_landslide

# Initialize SMOTE with the calculated ratio
smote = SMOTE(sampling_strategy=desired_ratio, random_state=42)

# Apply SMOTE
# Note: Ensure that 'Day' is not included in the features used to fit SMOTE.
X_train_sm, y_train_sm = smote.fit_resample(X_train.drop('Day', axis=1), y_train['landslide'])

#%%Hyperparameter tuning
# search_spaces = { 
#     'max_depth': Integer(3, 100),
#     'n_estimators': Integer(10, 200),
#     'min_samples_split': Integer(2, 20),
#     'min_samples_leaf': Integer(1, 20),
#     'max_features': Categorical(['sqrt', 'log2', None, 0.5]),
#     'bootstrap': Categorical([True, False]),
#     'criterion': Categorical(['gini', 'entropy']),
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

# opt.fit(X_train_sm, y_train_sm)
# print("Best parameters:", opt.best_params_)
# print("Best score:", opt.best_score_)

#%%Train the model
clf = RandomForestClassifier(
    n_estimators=180,       # Number of trees in the forest
    max_depth=8,           # Maximum depth of each tree
    min_samples_split=15,   # Minimum number of samples required to split an internal node
    min_samples_leaf=10,     # Minimum number of samples required to be at a leaf node
    max_features='log2',    # Number of features to consider when looking for the best split
    bootstrap=False,        # Whether bootstrap samples are used when building trees
    criterion='gini',       # The function to measure the quality of a split
    random_state=42         # Seed used by the random number generator
)
clf.fit(X_train_sm, y_train_sm)


# # Cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = []

# Cross-validation loop on X_train_sm
for train_index, test_index in skf.split(X_train_sm, y_train_sm):
    X_train_fold, X_test_fold = X_train_sm.iloc[train_index], X_train_sm.iloc[test_index]
    y_train_fold, y_test_fold = y_train_sm.iloc[train_index], y_train_sm.iloc[test_index]

    # Train the model with the chosen hyperparameters
    clf.fit(X_train_fold, y_train_fold)
    # Predict on test set
    y_pred = clf.predict(X_test_fold)

    # Evaluate the model
    scores.append({
        'accuracy': accuracy_score(y_test_fold, y_pred),
        'precision': precision_score(y_test_fold, y_pred, average='weighted'),
        'recall': recall_score(y_test_fold, y_pred, average='weighted'),
        'f1': f1_score(y_test_fold, y_pred, average='weighted')
    })

# Calculate average scores across all folds
scores_df = pd.DataFrame(scores)
average_scores = scores_df.mean()
print('Average scores after 10 K-fold:', average_scores)

#Prediction function 
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

#Loop for predictions 
for day in [7, 8, 9, 10]:
    y_pred_day, y_pred_proba_day, y_test_day = predict_for_day(clf, X_test, y_test, day)
    # print(f"Predictions for Day {day}: {y_pred_day}")
    # print(f"Probabilities for Day {day}: {y_pred_proba_day}")
    cm = confusion_matrix(y_test_day, y_pred_day)
    print(f"Confusion Matrix for Day {day}:\n{cm}\n")
    
colors = {8: '#ff7f0e',  # Orange
          9: '#2ca02c'}  # Green
    
def plot_roc_curve(y_test, y_scores, day_label, color):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, color=color, label=f'{day_label} ROC curve (AUC = {auc_score:.2f})')
    
# Start plotting
plt.figure(figsize=(10, 8))

for day in [8, 9]:  # Days for which to plot ROC curves
    y_pred_day, y_pred_proba_day, y_test_day = predict_for_day(clf, X_test, y_test, day)
    cm = confusion_matrix(y_test_day, y_pred_day)
    print(f"Confusion Matrix for Day {day}:\n{cm}\n")
    # Calculate and print day-specific metrics
    day_accuracy = accuracy_score(y_test_day, y_pred_day)
    day_precision = precision_score(y_test_day, y_pred_day)
    day_recall = recall_score(y_test_day, y_pred_day)
    day_f1 = f1_score(y_test_day, y_pred_day)
    
    print(f"Metrics for Day {day}:")
    print(f"Accuracy: {day_accuracy}, Precision: {day_precision}, Recall: {day_recall}, F1 Score: {day_f1}")
    plot_roc_curve(y_test_day, y_pred_proba_day, f'August {day}', colors[day])

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for August 8 and August 9', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#checking to see amount of landslides in each day
landslide_sums_test = y_test.groupby('Day')['landslide'].sum()
print(landslide_sums_test)
landslide_sums_total = y.groupby('Day')['landslide'].sum()
print(landslide_sums_total)

# Visualization of feature importance
features_list = X_train_sm.columns
plt.figure(figsize=(10, 8))
feature_imp = pd.Series(clf.feature_importances_, index=features_list).sort_values(ascending=False)
feature_imp.plot(kind='bar', title='Feature Importances')
plt.xticks(rotation=45)
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

#%% susceptibility map function

# def create_landslide_susceptibility_map(df, classifier, output_tif_path):
#     df = df.rename(columns={
#         'precipitation august 9': 'precipitation',
#         'soil saturation august 9': 'soil saturation'
#     })
#     X = df.drop([
#         'soil saturation august 8', 'soil saturation august 10', 'soil saturation august 2',
#         'soil saturation august 7', 'precipitation august 8', 'precipitation august 7',
#         'precipitation august 10', 'precipitation august 2', 'x', 'y', 'soil'
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
# output_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/SMOTE/no_soil/aug9/susceptibility_map_RF_august_9_less_overfitting.tif'
# create_landslide_susceptibility_map(df_study_area, clf, output_path)


#%% discrimination diagram
# all_y_pred_proba = []
# all_y_test = []

# for day in [7, 8, 9, 10]:
#     _, y_pred_proba_day, y_test_day = predict_for_day(clf, X_test, y_test, day)
    
#     # Assuming y_test_day is a DataFrame, we take its only column and convert it to numpy array
#     y_test_day = y_test_day['landslide']  # This ensures y_test_day is a 1D numpy array

#     # Extend the lists
#     all_y_pred_proba.extend(y_pred_proba_day)  # y_pred_proba_day is already a 1D numpy array
#     all_y_test.extend(y_test_day)

# # Convert lists to numpy arrays for easier processing
# all_y_pred_proba = np.array(all_y_pred_proba)
# all_y_test = np.array(all_y_test)

# # Define bins for forecast probabilities
# bins = np.linspace(0, 1, 11)  # 10 bins

# # Compute histograms for observed and not observed without density
# observed_histogram, _ = np.histogram(all_y_pred_proba[all_y_test == 1], bins=bins)
# not_observed_histogram, _ = np.histogram(all_y_pred_proba[all_y_test == 0], bins=bins)

# # Normalize the histograms to show probabilities instead of counts
# observed_probability = observed_histogram / observed_histogram.sum()
# not_observed_probability = not_observed_histogram / not_observed_histogram.sum()

# # Plot the histograms as probabilities
# plt.figure(figsize=(10, 6))
# width = bins[1] - bins[0]
# plt.bar(bins[:-1], observed_probability, width=width, alpha=0.5, label='Observed (Landslide)')
# plt.bar(bins[:-1], not_observed_probability, width=width, alpha=0.5, label='Not Observed (No Landslide)', edgecolor='black')

# plt.xlabel('Predicted Probability', fontsize=12)
# plt.ylabel('Likelihood', fontsize=12)
# plt.title('Discrimination Diagram for Landslide Prediction August 7-10', fontsize=12)

# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.legend(fontsize=12)

# # Display the plot
# plt.show()
#%% CREATING HISTOGRAMS OF PERCENTAGE WITHIN LOW, MEDIUM, HIGH LANDSLIDE PROBABILITY
# raster_aug7 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/SMOTE/no_soil/aug7/RF_susceptibility_map_august_7.tif'
# raster_aug8 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/SMOTE/no_soil/aug8/RF_susceptibility_map_august_8.tif'
# raster_aug9 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/SMOTE/no_soil/aug9/RF_susceptibility_map_august_9.tif'
# raster_aug10 = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/SMOTE/no_soil/aug10/RF_susceptibility_map_august_10.tif'
 
# with rasterio.open(raster_aug7) as src:
#     array_aug7 = src.read(1).flatten()  
# with rasterio.open(raster_aug8) as src:
#     array_aug8 = src.read(1).flatten()  
# with rasterio.open(raster_aug9) as src:
#     array_aug9 = src.read(1).flatten()  
# with rasterio.open(raster_aug10) as src:
#     array_aug10 = src.read(1).flatten() 
    
# bins = [0, 0.25, 0.5, 0.75, 1]
# # hist_RF, _ = np.histogram(array_RF, bins=bins)
# # hist_LR, _ = np.histogram(array_LR, bins=bins)
# hist_aug7, _ = np.histogram(array_aug7, bins=bins)
# hist_aug8, _ = np.histogram(array_aug8, bins=bins)
# hist_aug9, _ = np.histogram(array_aug9, bins=bins)
# hist_aug10, _ = np.histogram(array_aug10, bins=bins)
# # hist_percentage_RF = (hist_RF / hist_RF.sum()) * 100  
# # hist_percentage_LR = (hist_LR / hist_LR.sum()) * 100  
# hist_percentage_aug7 = (hist_aug7 / hist_aug7.sum()) * 100  
# hist_percentage_aug8 = (hist_aug8 / hist_aug8.sum()) * 100  
# hist_percentage_aug9 = (hist_aug9 / hist_aug9.sum()) * 100  
# hist_percentage_aug10 = (hist_aug10 / hist_aug10.sum()) * 100

# # Define bin edges and bar width
# bins = [0, 0.25, 0.5, 0.75, 1]
# bar_width = 0.2  # width of the bars

# # Set up the positions for the bars
# positions = np.arange(len(bins) - 1)
# offset = bar_width  # space between sets of bars

# # Plot the histograms as bars
# plt.figure(figsize=(10, 6))

# # Plot each histogram
# plt.bar(positions - 1.5 * offset, hist_percentage_aug7,  edgecolor='black', width=bar_width, label='August 7')
# plt.bar(positions - 0.5 * offset, hist_percentage_aug8, edgecolor='black',width=bar_width, label='August 8')
# plt.bar(positions + 0.5 * offset, hist_percentage_aug9, edgecolor='black',width=bar_width, label='August 9')
# plt.bar(positions + 1.5 * offset, hist_percentage_aug10, edgecolor='black',width=bar_width, label='August 10')

# # Add labels and title
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xlabel('Landslide Probability', fontsize=12)
# plt.ylabel('Area Percentage [%]', fontsize=12)
# #plt.title('Susceptibility Histograms', fontsize=12)

# # Add x-ticks
# plt.xticks(positions, labels=["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"])
# plt.yticks(np.arange(0, max(hist_percentage_aug7.max(), hist_percentage_aug8.max(),
#                             hist_percentage_aug9.max(), hist_percentage_aug10.max()) + 1, 10), fontsize=12)

# # Set the font sizes for tick labels on both axes to 12
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# # Add a legend
# plt.legend(fontsize=12)

# # Show the plot
# plt.show()
#%% Investigating the probability assigned by models on actual landslide pixels. 
# excel_files = {
#     'Aug 8': 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/landslide initiation points/august_8.xlsx',
#     'Aug 9': 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/landslide initiation points/august_9.xlsx'
# }

# # Define paths to raster files
# raster_files = {
#     'Aug 8': 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/SMOTE/no_soil/aug8/RF_susceptibility_map_august_8.tif',
#     'Aug 9': 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/resultater/SMOTE/no_soil/aug9/RF_susceptibility_map_august_9.tif'
# }

# # Initialize a list to store the results
# results_list = []

# # Loop through each Excel file and corresponding raster
# for key in excel_files.keys():
#     # Load the Excel file
#     df = pd.read_excel(excel_files[key])

#     # Extract the probability values for each point in the Excel file
#     with rasterio.open(raster_files[key]) as src:
#         coords = [(x, y) for x, y in zip(df['x'], df['y'])]
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
# plt.grid()

# # Adjust layout
# plt.tight_layout()

# # Show the plot
# plt.show()
#%%


















































