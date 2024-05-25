# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:28:56 2024

@author: Isak9
"""

import numpy as np
import pandas as pd
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
from rasterio.enums import Resampling
#%% cohesion values with tensile strength based on equation
input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NVE_NGIHVL.shp"
input_raster = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRBMU.tif"
input_raster2 = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/sr16 raster/sr16_34_SRRTRESLAG.tif"

square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
square_polygon = Polygon(square_coords)
pts = gpd.read_file(input_shapefile)  
pts = pts[pts.geometry.within(square_polygon)]

shapefile = gpd.read_file(input_shapefile)
square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
square_polygon = Polygon(square_coords)

with rasterio.open(input_raster) as src1:
    raster1, raster1_transform = mask(src1, [square_polygon], crop=True)
    raster1 = raster1[0]

with rasterio.open(input_raster2) as src2:
    raster2, raster2_transform = mask(src2, [square_polygon], crop=True)
    raster2 = raster2[0] 
    
raster1 = raster1.astype(float)
raster2 = raster2.astype(float)    
raster1[raster1 == -9999] = np.nan
raster2[raster2 == -9999] = np.nan

cohesion_gran = 1.2 * 11.55 * (raster1[raster2 == 1] / 451)
cohesion_furu = 1.2 * 9.6 * (raster1[raster2 == 2] / 539)
cohesion_lauv = 1.2 * 15.1 * (raster1[raster2 == 3] / 671)

cohesion_values = np.zeros_like(raster2, dtype=float)
cohesion_values[raster2 == 1] = cohesion_gran
cohesion_values[raster2 == 2] = cohesion_furu
cohesion_values[raster2 == 3] = cohesion_lauv

#%% sending the cohesion_values to a raster file
output_raster_path = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/cohesion_10/cohesion_10.tif"

affine = raster1_transform
crs = src1.crs  

with rasterio.open(
    output_raster_path, 'w',
    driver='GTiff',
    height=cohesion_values.shape[0],
    width=cohesion_values.shape[1],
    count=1,
    dtype=cohesion_values.dtype,
    crs=crs,
    transform=affine,
    nodata=np.nan  
) as dst:
    dst.write(cohesion_values, 1)
    
with rasterio.open(output_raster_path) as src:
    crs = src.crs
    print(f"CRS: {crs}")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    show(src, ax=ax, title='Cohesion Map', cmap='viridis')  

    ax.set_xlabel('Longitude' if crs.is_geographic else 'Easting')
    ax.set_ylabel('Latitude' if crs.is_geographic else 'Northing')
    ax.set_title('Root Cohesion Values in CRS')

plt.show()

#%% kohesjonsverdier i ulike landslide points basert på nye kohesjonsverdier!
input_shapefile = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/arcGIS/merged_points/NVE_NGIHVL.shp"
input_raster = "C:/Users/Isak9/OneDrive - NTNU/5. året NTNU/2. semester/Masteroppgave/cohesion_10/cohesion_10.tif"

square_coords = [(140438, 6886745), (140438, 6845000), (214000, 6845000), (214000, 6886745)]
square_polygon = Polygon(square_coords)
pts = gpd.read_file(input_shapefile)  
pts = pts[pts.geometry.within(square_polygon)]

with rasterio.open(input_raster) as src1:

    coords = [(geom.x, geom.y) for geom in pts.geometry]
    pts['cohesion'] = [x[0] for x in rasterio.sample.sample_gen(src1, coords)]



#%%
