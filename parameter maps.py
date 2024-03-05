# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:16:45 2024

@author: Isak9
"""

#VISUALIZING DIFFERENT PARAMETERS AND LANDSLIDE POINTS
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

input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NVE_NGIHVL.shp"

square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
square_polygon = Polygon(square_coords)
pts = gpd.read_file(input_shapefile)  
pts = pts[pts.geometry.within(square_polygon)]

plt.figure(figsize=(10, 8))
pts.plot(ax=plt.gca(), color='red', markersize=5)
slope_raster_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/høydedata/dtm10/data/total_dem.tif'
with rasterio.open(slope_raster_path) as src:
    slope_raster_data, slope_raster_transform = mask(src, [square_polygon], crop=True)
    plt.imshow(slope_raster_data[0], cmap='terrain', extent=(square_polygon.bounds[0], square_polygon.bounds[2], square_polygon.bounds[1], square_polygon.bounds[3]))

plt.title('Slope Raster with Points')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
cbar = plt.colorbar(label='Elevation [m]')
cbar.set_label(label='Elevation [m]', size=8)
plt.show()


plt.figure(figsize=(10, 8))
pts.plot(ax=plt.gca(), color='red', markersize=5)
cohesion_raster_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/cohesion/cohesion_10/cohesion_10.tif'
with rasterio.open(cohesion_raster_path) as src:
    cohesion_raster_data, cohesion_raster_transform = mask(src, [square_polygon], crop=True)
    plt.imshow(cohesion_raster_data[0], cmap='terrain', extent=(square_polygon.bounds[0], square_polygon.bounds[2], square_polygon.bounds[1], square_polygon.bounds[3]))
plt.title('Root cohesion with Points')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
cbar = plt.colorbar(label='Root Cohesion [kPa]')
cbar.set_label(label='Root Cohesion [kPa]', size=8)
plt.show()


plt.figure(figsize=(10, 8))
pts.plot(ax=plt.gca(), color='red', markersize=5)
aspect_raster_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/Aspect/total_aspect.tif'
with rasterio.open(aspect_raster_path) as src:
    aspect_raster_data, aspect_raster_transform = mask(src, [square_polygon], crop=True)
    plt.imshow(aspect_raster_data[0], cmap='hsv', extent=(square_polygon.bounds[0], square_polygon.bounds[2], square_polygon.bounds[1], square_polygon.bounds[3]))
plt.title('Slope Aspect with Points')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.show()

plt.figure(figsize=(10, 8))
pts.plot(ax=plt.gca(), color='red', markersize=5)
tretype_raster_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/tretype/tretype.tif'
with rasterio.open(tretype_raster_path) as src:
    tretype_raster_data, tretype_raster_transform = mask(src, [square_polygon], crop=True)
    plt.imshow(tretype_raster_data[0], cmap='tab10', extent=(square_polygon.bounds[0], square_polygon.bounds[2], square_polygon.bounds[1], square_polygon.bounds[3]))
plt.title('Treetype with Points')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.show()

plt.figure(figsize=(10, 8))
pts.plot(ax=plt.gca(), color='red', markersize=5)
soil_raster_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/ML/soil_raster/soil_raster.tif'
with rasterio.open(soil_raster_path) as src:
    soil_raster_data, soil_raster_transform = mask(src, [square_polygon], crop=True)
    plt.imshow(soil_raster_data[0], cmap='terrain', extent=(square_polygon.bounds[0], square_polygon.bounds[2], square_polygon.bounds[1], square_polygon.bounds[3]))
plt.title('Soil with Points')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.show()

plt.figure(figsize=(10, 8))
pts.plot(ax=plt.gca(), color='red', markersize=5)
flow_raster_path = 'C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/flow_accu/flowacc_tot.tif'
with rasterio.open(flow_raster_path) as src:
    flow_raster_data, flow_raster_transform = mask(src, [square_polygon], crop=True)
    plt.imshow(flow_raster_data[0], cmap='terrain', extent=(square_polygon.bounds[0], square_polygon.bounds[2], square_polygon.bounds[1], square_polygon.bounds[3]))
plt.title('Flow Accumulation with Points')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.show()

