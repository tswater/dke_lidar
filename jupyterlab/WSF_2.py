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
# Playing with WSF
# Use for deeper analysis before implementing

# Some info:
'''
-Bounding Box: [-97.9882090,37.1077130] to [-96.9882090,36.1077130] 
-from the metadata in the LST tiffs

Candidate Conditions: (timeframe between 10 AM and 4-5 PM, over the region)
-Surface and 500 hPa winds: already included in some form, refine 
-Fronts and other features: check for fronts, try different distances around both the box and the sites
-Dew Point Temperature: higher humidity values are of interest
-Shallow Cumulus: check rainfall means, implicitly screening for scattered/isolated storms
-Background DKE/Vorticity: check in morning to make sure there's no background effects from mean flow
'''

# better screen first --> then add more data

# %%
# Imports
import netCDF4 as nc
import glob
import numpy as np
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
from osgeo import gdal
import pandas as pd
from herbie import Herbie
import metpy.calc as mpcalc # calculate vorticity ourselves
from metpy.units import units

# %%
# Herbie!
# Herbie object
H = Herbie(
    '2021-07-15 15:00', # datetime
    model='hrrr', # model
    product='sfc', # produce name?
    fxx=0, # lead time, want 0
    save_dir='/home/pjg25/tyche/data'
)
# H.inventory(search='') --> search for data
# Variables we want
# ':APCP:' --> all accumulated precip (surface)
# ':RELV:' --> vorticity (850 mb) ???
# ':DPT:' --> dew point temp (2m)

# Reading Data
# Regex: 
#ds = H.xarray('(:DPT:2 m*|:APCP:*|:RELV:*)', remove_grib=False) # what does this give?
ds = H.xarray(':[UV]GRD:', remove_grib=False)
winds = ds[2]

# (:DPT:2 m*|:APCP:|:RELV:) --> will this work?

#data = ds.sel(heightAboveGroundLayer=1, x=610, y=899, method="nearest")
u = winds['u'].values[2,591:630,887:916] * (units.meter / units.second)
v = winds['v'].values[2,591:630,887:916] * (units.meter / units.second)
lat = winds['latitude'].values[591:630,0] 
lon = winds['longitude'].values[0,887:916] 
dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
vort = mpcalc.vorticity(u, v, dx=dx, dy=dy)


# %%
vort

# %%
a = np.abs(np.array(vort)) > 12e-5
np.sum(a)

# %%
# Iterate the above
# Incorporate into the main script
# Test iteration (one month)

# for day in date:
#    create Herbie object
#    read variables
#    central site
#    pick points
#    check specific variables
#        -vorticity
#        -precip
#        -dew point

# %%
# Xarray Opening
# Test file
filepath = '/stor/tyche/hydro/shared/HRRR/'
hrrr = filepath + 'hrrr2.20230619.t16z.grib2' # 14=10 am, 16=noon

# Latitude and Longitude to Indices --> upscale to take a range?
def xy_from_coord(lat, long, pointer, box=False): 
    ds = xr.open_dataset(pointer, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, backend_kwargs={'indexpath':''})
    ds_lat = ds.coords['latitude'].values
    ds_long = ds.coords['longitude'].values
    l = lat - ds_lat
    m = long - ds_long
    lat_ind = max(np.where(l > 0)[0])
    lon_ind = max(np.where(m > 0)[1])
    return lat_ind, lon_ind

# Central Site: 36.60732, -97.48764
clat, clon = xy_from_coord(36.60732, -97.48764+360, hrrr)

# Bounding box
bl = xy_from_coord(36.1077130, -97.9882090+360, hrrr)
tr = xy_from_coord(37.1077130, -96.9882090+360, hrrr)
latrange = list(range(bl[0], tr[0]+1))
lonrange = list(range(bl[1], tr[1]+1))
'''
# Pressure Slicing (account for the fact we are reading in isobaric)
fp = xr.open_dataset(hrrr, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, backend_kwargs={'indexpath':''})
fp2 = xr.open_dataset(hrrr, engine='cfgrib', filter_by_keys={'stepType':'instant','typeOfLevel': 'surface'}, backend_kwargs={'indexpath':''})
# Slice
press = fp.coords['isobaricInhPa'].values
p500 = np.where(press == 500)[0] # 500 mbar isobar
sp = fp2.variables['sp'].values[bl[0]:tr[0], bl[1]:tr[1]] / 100
surface = np.zeros(np.shape(sp))
for i in range(0, np.shape(sp)[0]):
    for j in range(0, np.shape(sp)[1]):
        p_close_s = np.min(np.abs(sp[i][j] - press))
        surf_ind = np.where(p_close_s == np.abs(sp[i][j] - press))[0][0]
        surface[i][j] = surf_ind

# Dew Point at surface
surf_dpt = fp['dpt'][3,bl[0]:tr[0],bl[1]:tr[1]].values # dew points at surface over the box
mask = np.sum(surf_dpt < 290.15)
'''
print(bl)
print(tr)

# %%
# Wind Speeds (included)
center = xy_from_coord(36.60771, -97.4882+360, hrrr)
center

# %%
# Frontal Systems
rad = 200 # kilometer radius around box (vary)
# midpoint of box --> diagonal is 142.7 km
# upscale

# # fronts? 
# pressure gradients
# temperature gradients

# %%
# Precipitation and ShCu
mm = 5  # millimeters of precipitation, 
fp3 = xr.open_dataset(hrrr, engine='cfgrib', filter_by_keys={'stepType':'accum','typeOfLevel': 'surface'}, backend_kwargs={'indexpath':''})
# units: 1 kg/m2 = 1 mm of rain

precip = fp3['tp'][bl[0]:tr[0],bl[1]:tr[1]].values
# find out how the accumulation is done

# %%
# Background DKE/Vorticity
vort = fp['absv'][3,bl[0]:tr[0],bl[1]:tr[1]].values # s^-1
# Planetary Voticity subtraction

# What value is large enough?

# %%
fp

# %%
f=xr.open_dataset(hrrr, engine='cfgrib', filter_by_keys={'stepType':'accum','typeOfLevel': 'surface'}, backend_kwargs={'indexpath':''})
f['tp'].values

# %%
vort

# %%
