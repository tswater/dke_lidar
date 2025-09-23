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
"""
Weak Synoptic Forcing Screening
"""

# %%
# Data Required: NWS Frontal Maps, Wind Data at surface and 500 hPa, Dew Point Temperature
# Fronts: NWS --> NA Surface Analysis
# Dew Point: NOAA Dewpoint Forecasts
# Winds: NWS/SPC Rawinsonde/Surface Observations

# Conditions for Weak Synoptic Forcing
# 1: No frontal features within 500 km
# 2: Surface winds < 5 m/s
# 3: 500 hPa winds < 10 m/s
# 4: Dew Point > 17 C

# %%
####
# Data for WSF Screening will come from the HRRR
# Get a box of lat/long for analysis
# wrfpython diagnostics?

# %%
import netCDF4 as nc
import glob
import numpy as np
import xarray as xr

# %%
# Lat/Long Box for region (change as needed)
# Lat: 36 - 37 N
# Lon: 96 - 98 W (to 360)

newpath = '/stor/tyche/hydro/shared/HRRR/'
test = newpath + 'hrrr2.20230619.t12z.grib2'
    
def xy_from_coord(lat, long, pointer): # pass latitude and longitude, output indices in the simulation
    ds = xr.open_dataset(pointer, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, backend_kwargs={'indexpath':''})
    ds_lat = ds.coords['latitude'].values
    ds_long = ds.coords['longitude'].values
    l = lat - ds_lat
    m = long - ds_long
    lat_ind = max(np.where(l > 0)[0])
    lon_ind = max(np.where(m > 0)[1])
    return lat_ind, lon_ind

# Central Site
#central = xy_from_coord(36.60732, -97.48764+360, test)
#print(central)


# %%
# Structure of HRRR Data
# (1, 40, 1059, 1799)
# 1 time step, 50 vertical layers, latitude, longitude

# Input: Date as YYYYMMDD
# Box: [36,-98+360], [37,-96+360]

def WSF(date, box): 
    path = '/stor/tyche/hydro/shared/HRRR/hrrr2.'+date+'.t16z.grib2' # change how this iterates
    files = glob.glob(path) # files by hour (first 24)
    # Bounding Box (single file call)
    test = files[0]
    b_l = xy_from_coord(box[0][0], box[0][1], test) # bottom left
    t_r = xy_from_coord(box[1][0], box[1][1], test) # top right
    latrange = list(range(b_l[0], t_r[0]))
    lonrange = list(range(b_l[1], t_r[1]))
    
    # Frontal Systems (Mode 1)
    # Hardest one?
   
    for i in range(0, len(files)): # winds only
        marker = 0
        test = files[i]

        # 500 hPa Wind
        ds = xr.open_dataset(test, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, backend_kwargs={'indexpath':''})
        press = ds.coords['isobaricInhPa'].values
        press_close = np.min(np.abs(500 - press))
        press_ind = np.where(press_close == np.abs(500 - press))[0][0]
        u = ds.variables['u'].values[press_ind,610,899]
        v = ds.variables['v'].values[press_ind,610,899]
        wspd500 = np.sqrt(u**2 + v**2)
        # Mask
        mask500 = (wspd500 <= 10)
        # locate Falses
        bools500 = mask500.flatten()
        for i in range(0, len(bools500)):
            bool = bools500[i]
            if bool == True:
                pass
            elif bool == False:
                #print('Failed 500 hPa check')
                marker = 1
                break
        if marker == 1:
            return 0
            break

        # Surface Wind
        ds2 =  xr.open_dataset(test, engine='cfgrib', filter_by_keys={'stepType':'instant','typeOfLevel': 'surface'}, backend_kwargs={'indexpath':''})
        sp = ds2.variables['sp'].values[610,899]
        # closest isobar to surface? 
        p_close_s = np.min(np.abs(sp - press))
        surf_ind = np.where(p_close_s == np.abs(sp - press))[0][0]
        usf = ds.variables['u'].values[surf_ind,610,899]
        vsf = ds.variables['v'].values[surf_ind,610,899]
        swspd = np.sqrt(usf**2 + vsf**2)
        # Mask
        masksurf = (swspd <= 5)
        # Falses
        boolsurf = masksurf.flatten()
        for i in range(0, len(boolsurf)):
            bool = boolsurf[i]
            if bool == True:
                pass
            elif bool == False:
                #print('Failed Surface check')
                marker = 1
                break
        if marker == 1:
            return 0
            break
        ds.close()
        ds2.close()
    return 1
WSF('20210601', [[36,-98+360], [37,-96+360]])

# %%
# TO DO:
# Close on wind speed checks --> upscale to a region and longer timescale (current state checks the central site at noon only)
# Need to figure the frontal part
# Integrate MKE into the DKE calcs for the ARM lidars --> generate Figure 6 (w/screening)

# Iterate Days (June to August 2023)
months = ['06','07','08']
yrs = ['2023']
date_strings = []
for yr in yrs:
    for mon in months:
        for i in range(1,32):
            # leading zero formatting
            if i<10:
                day = '0' + str(i)
            else: 
                day = str(i)
            run_string = yr + mon + day
            if mon=='06' and day=='31':
                continue
            date_strings += [run_string]

# Check
screened = []
for s in date_strings:
    mark = WSF(s)
    if mark == 1:
        screened += [s]

# %%
# NEXT
# Look at Nate's MsKE script

# Incorporate WSF checks at surface and 500 hPa
#      try conditions that cover the area (what percentage of pixels?)

# Dew Point Temperature Check?

# 
