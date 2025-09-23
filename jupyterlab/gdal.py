# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# GDAL
# GOES data subsetting
import os
from osgeo import gdal
import glob

# %%
years = ['2017', '2023', '2024']
for y in years:
    oldpath = '/stor/soteria/hydro/shared/data/GOES/GOES-16-EPSG4326/%s/' % y
    newpath = '/stor/soteria/hydro/private/pjg25/GOES_holes/'

    options = gdal.WarpOptions(
        format = 'GTiff',
        outputBounds = (-97.9882,36.1077,-96.9882,37.1077)
    )
    
    for file in os.listdir(oldpath):
        old_raster = os.path.join(oldpath, file)
        new_raster = os.path.join(newpath, file)
        gdal.Warp(new_raster, old_raster, options=options)
        pass
    print(newpath)

# %%
path = '/stor/soteria/hydro/private/pjg25/GOES_holes/*'
files = glob.glob(path)
len(files)

# %%
