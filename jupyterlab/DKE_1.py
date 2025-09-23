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
DKE Calculations
"""

# %%
# Data: Doppler Lidar products from ARM --> u,v,w profiles
# Central Site + 2 supplemental sites 
# Compute Integrated DKE

# LIDAR DATA (2023)
# C1 = central site
# E37 = Waukomis, OK
# E39 = Morrison, OK

# filenames: sgpdlprofwind4news*.c1.
# * = site number
# ^ = date, YYYYMMDD

# %%
import netCDF4 as nc
import glob
import numpy as np
import wrf
import cfgrib
import pandas as pd
import xarray as xr
from scipy.ndimage import interpolation
from herbie import Herbie

path = "/stor/tyche/hydro/private/pjg25/ARM Lidar" # Access folder where the data is
C1 = glob.glob(path+'/sgpdlprofwind4newsC1.c1.20230501*')
E37 = glob.glob(path+'/sgpdlprofwind4newsE37.c1.20230501*')
E39 = glob.glob(path+'/sgpdlprofwind4newsE39.c1.20230501*')

# %%
# Herbie for HRRR archive
#H1 = Herbie(
#    '2018-05-01 12:00',
#    model='hrrr',
#    product='sfc',
#    fxx=0 # hour 0, no lead
#)
#H1.inventory(search=':TMP:')

# checks
# vorticity (at central site)
center = pd.DataFrame( #coordinates of center
    {
        'latitude': [36.6077],
        'longitude': [-97.4882]
    }
)
H1 = Herbie('2017-05-01 10:00', model='hrrr', fxx=0)
#ds = H1.xarray(':vo:') --> 

# precipitation
# dew point? 

# %%
ds

# %%
fp = C1[0]
a = nc.Dataset(fp,'r') 
for v in a.variables: # Keep for reference, description of every variable
    try:
        desc=a[v].description
    except:
        desc=''
    print(v+': '+desc)


# %%
# u,v,w shapes
# (96, 164)
# 96 time steps, measured seconds from midnight
# other dimension is the step in the z direction

# %%
# General DKE Calculation (with MKE)
# DATE --> RELEVANT FILES
def gen_date(date): # date as a str, YYYYMMDD; returns site files
    arm_path = '/stor/tyche/hydro/private/pjg25/ARM Lidar'
    # Sites
    C1 = glob.glob(path+'/sgpdlprofwind4newsC1.c1.' + date + '*')[0]
    E37 = glob.glob(path+'/sgpdlprofwind4newsE37.c1.' + date + '*')[0]
    E39 = glob.glob(path+'/sgpdlprofwind4newsE39.c1.' + date + '*')[0]
    sitefiles = [C1, E37, E39]
    return sitefiles

# Coordinates (from WSF function)
def xy_from_coord(lat, long, pointer): # pass latitude and longitude, output indices in the simulation
    ds = xr.open_dataset(pointer, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, backend_kwargs={'indexpath':''})
    ds_lat = ds.coords['latitude'].values
    ds_long = ds.coords['longitude'].values
    l = lat - ds_lat
    m = long - ds_long
    lat_ind = max(np.where(l > 0)[0])
    lon_ind = max(np.where(m > 0)[1])
    ds.close() # added
    return lat_ind, lon_ind

# FILL VALUE SCREENER
def mask_screen(pointer, n): # Input: pointer = file pointer, n = index (for iteration later) 
    u = np.array(pointer['u'][n])
    u = u * (u > -1000)
    v = np.array(pointer['v'][n])
    v = v * (v > -1000)
    w = np.array(pointer['w'][n])
    w = w * (w > -1000)
    profile = [u,v,w]
    return profile

# VERTICAL PROFILE TIME AVERAGES
def averager(pointer, t_i = 96): # Input: pointer = file pointer
    us = []
    vs = []
    ws = []
    for ind in range(0, t_i):
        u = mask_screen(pointer,ind)[0]
        us += [u]
        v = mask_screen(pointer,ind)[1]
        vs += [v]
        w = mask_screen(pointer,ind)[2]
        ws += [w]
    u_avg = np.mean(us, axis=0)[:-1] # neglecting the top
    v_avg = np.mean(vs, axis=0)[:-1]
    w_avg = np.mean(ws, axis=0)[:-1]
    return [u_avg, v_avg, w_avg]

# SPATIAL VARIANCE + MEAN for MKE (3 SITES)
# Return Variances at all heights and one step size array
def spit(date, pointers, profiles): # Input: pointers = file pointers in a list, profiles = corresponding time averaged profiles (list of list of profiles per site)
    # Variance
    vars = np.var(profiles, axis=0)
    means = (np.mean(profiles, axis=0))**2 # squared for MKE
    # Heights
    diffs = []
    for m in range(0, len(pointers)):
        h = pointers[m]['height']
        dh = np.diff(h)
        diffs += [dh]
    avg_diff = np.mean(diffs, axis=0)
    # Density (from heights)
    gribfile = '/stor/tyche/hydro/shared/HRRR/hrrr2.20230601.t12z.grib2'
    slati = [] # indices of site coordinates --> for HRRR pressure reading
    sloni = []
    for p in pointers:
        slat = float(np.array(p['lat']))
        slon = float(np.array(p['lon']))
        ilat, ilon = xy_from_coord(slat, slon+360, gribfile)
        slati += [ilat]
        sloni += [ilon]
    denses = []
    for k in range(0,3):
        r = cent_dens(date, slati[k], sloni[k])
        denses += [r]
    rho = np.mean(denses, axis=0)
    
    return vars, means, avg_diff, h, rho

# DENSITY
def cent_dens(date, lati, loni):
    Rs = 287.050 # J / (kg K)
    grib = '/stor/tyche/hydro/shared/HRRR/hrrr2.' + date + '.t16z.grib2'
    ds = xr.open_dataset(grib, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, backend_kwargs={'indexpath':''})
    ds2 = xr.open_dataset(grib, engine='cfgrib', filter_by_keys={'stepType':'instant','typeOfLevel': 'surface'}, backend_kwargs={'indexpath':''})
    # Read
    sp = ds2.variables['sp'].values[lati,loni]
    press = ds.coords['isobaricInhPa'].values
    p_close_s = np.min(np.abs(sp - press))
    surf_ind = np.where(p_close_s == np.abs(sp - press))[0][0]
    # Variables
    T_k = ds.variables['t'][surf_ind:32,lati,loni].values # k
    P = ds.coords['isobaricInhPa'][surf_ind:32].values # hPa
    rho = (P*100) / (Rs * T_k)
    rho = interpolation.zoom(rho,163/32)
    
    return rho

# DKE INTEGRATION
def DKE(date, pointers): # Input: pointers = list of pointers
    profs = []
    for p in range(0, len(pointers)):
        profs += [averager(pointers[p])]
    (vars, means, dz, h, rho) = spit(date, pointers, profs)
    # DKE
    vals = 0.5 * rho * np.sum(vars, axis=0) * dz
    intDKE = np.sum(vals)
    # MKE
    mvals = 0.5 * rho * np.sum(means, axis=0) * dz
    intMKE = np.sum(mvals)
    
    return intDKE / intMKE # fraction

# STATUS: DKE value obtained (kind of), have not accounted for density yet
# put this in the spit function
# base on temperature from WRF



# %%
# Integrated DKE / Integrated MKE Calculator
# Reports Fraction, uses ARM LIDAR data from 3 sites (2023)
# Enter Date: YYYYMMDD as a string (str)

def general(date): # Data in folder is 2023 only, add more later
    sites = gen_date(date)
    # Site-specific
    C1 = sites[0]
    E37 = sites[1]
    E39 = sites[2]
    # Pointers
    fpC1 = nc.Dataset(C1, 'r')
    fpE37 = nc.Dataset(E37, 'r')
    fpE39 = nc.Dataset(E39, 'r')   
    # Call DKE function
    report = DKE(date,[fpC1, fpE37, fpE39])
    # return the DKE value and close pointers
    fpC1.close()
    fpE37.close()
    fpE39.close()
    return report

general('20230701') # change date

# %%
# GOES LST
# Tailoring Work that Nate has done 


# %%
# NEXT: HRRR DKE Calculator (Moving Window)

# TO DO
# modify DKE calcs as needed (including density)
# extend to MKE

# Utility: Date to Pointers for Daytime Hours
def date_pointers(date, m=10, a=16): # Date = 'YYYYMMDD', m and a = hours in 24-hr time
    # edit to include homogeneous sim
    # Daytime: 10 AM to 4 PM
    daypath = '/stor/tyche/hydro/shared/HRRR/hrrr2.' + date 
    hours = []
    for d in range(m, a+1):
        add = '.t' + str(d+4) + 'z.grib2'
        hours += [daypath + add]
    return hours

# Latitude and Longitude --> Box and spacing for window
def latlon(bl, tr, pointer, box): # Inputs: bottom_left, top right (lat and long), box=window size in km
    coords = bl + tr
    inds = []
    ds = xr.open_dataset(pointer, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, backend_kwargs={'indexpath':''})
    ds_lat = ds.coords['latitude'].values
    ds_long = ds.coords['longitude'].values
    for i in range(0, 2):
        lat = coords[2*i]
        long = coords[1 + (2*i)]
        l = lat - ds_lat
        m = long - ds_long
        lat_ind = max(np.where(l > 0)[0])
        lon_ind = max(np.where(m > 0)[1])
        inds += [lat_ind, lon_ind]
    # Spacing (for box later)
    r_earth = 6371 # km
    # Haversine formulas
    width = (2*r_earth) * np.arcsin(np.sqrt(0.5*np.cos(bl[0]*np.pi/180)*np.cos(bl[0]*np.pi/180)*(1-np.cos((np.abs(bl[1]-tr[1])*np.pi/180))))) # correct
    length = (2*r_earth) * np.arcsin(np.sqrt(0.5*(1-np.cos((np.abs(bl[0]-tr[0])*np.pi/180))))) # correct
    # Splits
    Ws = round(width / box)
    Ls = round(length / box)
    # Format for windows: [list of lats, list of lons]
    latb = []
    lonb = []
    for i in range(0, Ws):
        latb += [round(inds[0] + (((i+1)/Ws)*(inds[2] - inds[0])))]
    for j in range(0, Ls):
        lonb += [round(inds[1] + (((j+1)/Ls)*(inds[3] - inds[1])))]
    #return latb, lonb # indices of windows 
    return latb, lonb

# Density (HRRR), assume ideal gas or humid air?
def HRRR_density(data1, data2, latind, lonind): # Data1 = isobaric layers, Data2 = surface data
    # Find Surface Pressure
    p_all = data1.coords['isobaricInhPa'].values
    sp = data2.variables['sp'].values[latind,lonind] / 100
    p_close_s = np.min(np.abs(sp - p_all))
    surf_ind = np.where(p_close_s == np.abs(sp - p_all))[0][0]

    # Density Calculation (Dry Air)
    Rs = 287.050 # J / (kg K)
    T_k = data1.variables['t'][surf_ind:32,latind,lonind].values # k
    P = data1.coords['isobaricInhPa'][surf_ind:32].values # hPa
    rho = (P*100) / (Rs * T_k)
    
    return rho, P, surf_ind
    
# NEED TO FIGURE OUT HOW TO GET HEIGHT ABOVE GROUND

# DKE Process w/Spatial Averaging
def spit2(file, lats, lons): # INPUTS: indices from latlon, pointer,  
    # get time-averaged profiles over daytime hours (10 AM to 4 PM, edit this too)
    latrange = list(range(lats[0], lats[1]))
    lonrange = list(range(lons[0], lons[1]))

    # find density at the central point (from indices)
    centlat = latrange[round(len(latrange)/2)]
    centlon = lonrange[round(len(lonrange)/2)]
    ds2 =  xr.open_dataset(file, engine='cfgrib', filter_by_keys={'stepType':'instant','typeOfLevel': 'surface'}, backend_kwargs={'indexpath':''})
    centrho, press, surf = HRRR_density(data, centlat, centlon) 
    zs = (np.diff(press, append=press[-1])*100) / (centrho * 9.81)
    
    # find spatial variance within window 
    u_prof = [] 
    v_prof = []
    w_prof = []
    for x in latrange:
        for y in lonrange:
            # xarray
            ds = xr.open_dataset(file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, backend_kwargs={'indexpath':''})
            u = ds.variables['u'][surf:32,x,y].values # to top of troposphere (250 mb)
            v = ds.variables['v'][surf:32,x,y].values
            w = ds.variables['w'][surf:32,x,y].values
            ds.close()
            u_prof += [[u]]
            v_prof += [[v]]
            w_prof += [[w]]    

    # All profiles together [3 x 6 x points]
    all_profiles = [u_prof, v_prof, w_prof]
    # Close any pointer(s)
    ds2.close()
    
    return all_profiles, centrho, zs
    

def HDKE(date, mapbox, window): # other inputs; DATE as a str, bounding box coordinates in a list, spacing in km of window
    files = date_pointers(date)
    # Loop
    counter = 0
    boxes = [] # to store results
    for h in range(0, len(files)):
        # Latitude and longitude index grabber (do once)
        if counter == 0:
            (lats, lons) = latlon([36,-98+360],[37,-96+360], files[h], window)
        # Iterate through boxes --> read in data
        for i in range(0,len(lats)-1):
            for j in range(0,len(lons)-1):
                boxholder = []
                lat_edge = [lat[i], lat[i+1]]
                lon_edge = [lon[i], lon[i+1]]
                (prof, rho, zs) = spit2(files[h], lat_edge, lon_edge) 
                

    # Averaging [6 x number of points x vertical layers]
    # time average first
    # utavg = np.mean(u, axis=0)  # right axis?
    # vtavg = np.mean(v, axis=0)
    # wtavg = np.mean(w, axis=0)
    
    # spatial variance next
    # uvar = np.var(u, axis=1) 
    # vvar = np.var(v, axis=1) 
    # wvar = np.var(w, axis=1) 
    # sumvar = uvar + vvar + wvar  # at each layer
     
    # integrate
    # DKE_i = rho * sumvar * zs (element-wise)
    # INTDKE = 0.5 * np.sum(DKE_i)
    
    return 0

#HDKE('20230601', [[36,-98+360],[37,-96+360]], 60)
# STATUS: DKE calculator fixed and made for HRRR, test on a bounding box
# Work into the ARM DKE calculator
# Very Close on this one
# Incorporate MKE later

# %%
newpath = '/stor/tyche/hydro/shared/HRRR/'
test = newpath + 'hrrr2.20230619.t12z.grib2'
# General Format: newpath + 'hrrr2.'+ date + '.t' + hour + 'z.grib2'
tester = xr.open_dataset(test, engine='cfgrib', filter_by_keys={'typeOfLevel': 'sigma'},backend_kwargs={'indexpath':''})
#tester = xr.open_dataset(test, engine='cfgrib', backend_kwargs={'indexpath':''})
#tester = xr.
#filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, 
#print(test)

# %%
a = tester.variables['sp'].values[610,899] / 100
a

# %%
g, h, i = HRRR_density(tester2, tester, 670, 1006)
w = (np.diff(h, append=h[-1])*100) / (g * 9.81)
p = np.abs(w)*g*5
p

# %%
tester

# %%
tester2

# %%
