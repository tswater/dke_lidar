# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc
import pickle
import numpy as np
from scipy import stats
matplotlib.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %% [markdown]
# # Trying to figure out pickle files 

# %%
proot='/home/tswater/projects/dke_lidar/'
troot='/home/tswater/tyche/data/dke_peter/'
with open(proot+'/pickle_network/20150606dke.pkl', 'rb') as file:
    p20150606dke = pickle.load(file)
with open(proot+'/pickle_network/20150606.pkl', 'rb') as file:
    p20150606 = pickle.load(file)

# %%
dkem=[]
dkes=[]
for k in p20150606dke.keys():
    dkem.append(np.nanmedian(p20150606dke[k]))
    dkes.append(np.nanstd(p20150606dke[k]))

# %%
with open(proot+'/pickle_network/base.pkl', 'rb') as file:
    base = pickle.load(file)
with open(proot+'/pickle_network/corrs.pkl', 'rb') as file:
    corrs = pickle.load(file)
with open(proot+'/pickle_network/dkebase.pkl', 'rb') as file:
    dkebase = pickle.load(file)
with open(proot+'/pickle_network/hets.pkl', 'rb') as file:
    hets = pickle.load(file)
with open(proot+'/pickle_network/nets.pkl', 'rb') as file:
    nets = pickle.load(file)

# %%
ht=[]
dk=[]
mk=[]
rt=[]
for k in hets.keys():
    ht.append(hets[k])
    dk.append(dkebase[k])
    a=dkebase[k]/base[k]
    mk.append(a)
    rt.append(base[k])
rt=np.array(rt)
dk=np.array(dk)
mk=np.array(mk)
ht=np.array(ht)

# %%
color=plt.get_cmap('terrain')((np.array(mk)-900)/50000)
rt2=np.sqrt(dk)/np.sqrt(mk)
plt.scatter(ht,rt2,c=color)
#plt.semilogy([10,.20],[.01,.02],alpha=0)
stats.spearmanr(ht,rt2)
#plt.ylim(0,0.04)

# %%

# %%

# %%

m=mk<5000

color=plt.get_cmap('terrain')((np.array(mk)-900)/50000)
color=color[m]
rt2=np.sqrt(dk)/np.sqrt(mk)
plt.scatter(ht[m],rt2[m],c=color)
#plt.semilogy([10,.20],[.01,.02],alpha=0)
print(stats.spearmanr(ht[m],rt2[m]))
#plt.ylim(0,0.04)
plt.xlim(0,120)
plt.ylim(0,.3)

# %%

m=(mk>5000)&(mk<12500)

color=plt.get_cmap('terrain')((np.array(mk)-900)/50000)
color=color[m]
rt2=np.sqrt(dk)/np.sqrt(mk)
plt.scatter(ht[m],rt2[m],c=color)
#plt.semilogy([10,.20],[.01,.02],alpha=0)
print(stats.spearmanr(ht[m],rt2[m]))
plt.xlim(0,120)
plt.ylim(0,.3)

# %%

m=(mk>12500)

color=plt.get_cmap('terrain')((np.array(mk)-900)/50000)
color=color[m]
rt2=np.sqrt(dk)/np.sqrt(mk)
plt.scatter(ht[m],rt2[m],c=color)
#plt.semilogy([10,.20],[.01,.02],alpha=0)
print(stats.spearmanr(ht[m],rt2[m]))
plt.xlim(0,120)
plt.ylim(0,.3)

# %%

# %%
plt.scatter(mk,dk/mk)

# %%

# %%
from scipy import stats
stats.spearmanr(ht,dk)

# %%
np.max(mk)

# %%
plt.hist(mk)

# %%
plt.hist(dk)

# %% [markdown]
# # MsKE Redo

# %%
import glob
import netCDF4 as nc
import os
import rasterio
#import xarray as xr
import datetime
import scipy

# 2017-2024 (1181 days, allegedly)
tp = nc.Dataset(proot+'/other_data/lst_dat.nc', 'r')
lst_std_ts = tp['std'][:]
lst_mean_ts = tp['mean'][:]
lst_lscale_ts = tp['lscale'][:]
datesnc = tp['date'][:]
#tp.close()

# %%
len(lst_lscale_ts.data[lst_lscale_ts>0])

# %%
# Cell that does the heavy lifting
# Generate Data for Scatter Plot
import glob
import time
import numpy as np
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
from numba import jit
from herbie import Herbie
import pandas as pd
import metpy.calc as mpcalc # calculate vorticity ourselves
from metpy.units import units
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@jit(nopython=True)
def upscale(yin,window): # Upscaling Data?
    #print(np.unique(yin))
    i = 0
    count = 0
    yout = 0
    nt = int(yin.shape[0]/window)
    yout = np.ones((nt,yin.shape[1]))
    while i < yin.shape[0]:
        for j in range(yin.shape[1]):
            m1 = yin[i:i+window,j] != -9999
            if np.sum(m1) > 0:
                yout[count,j] = np.mean(yin[i:i+window,j][m1])
        i = i + window
        count += 1
    return yout 

# Density Function (not important)
def calculate_air_density(z,minz,maxz): #hack
        T = 20 - 0.00649 * z;                        
        P = (101.29) * ((T + 273.15)/288.08)**(5.256);
        rho =  P/(0.2869*(T + 273.15));
        rho[z < minz] = 0.0
        rho[z > maxz] = 0.0
        return rho
                                
#files = glob.glob('/home/nc153/soteria/data/SGP/DL/*.nc')
# ARM Data, substitute my own files here as needed
arm1 = glob.glob(troot+'lidar_wind_profile/245518/*.nc') # ARM
holes = glob.glob(troot+'lidar_wind_profile/ARM/*') # 2016-17, 23 (none for May 24)
sonde = glob.glob(troot+'sgpsondewn/*') # radiosonde (available for all days 2017-22
# combine these, screen duplicates? 

# add data from 2024 if its available
u = []
v = []
DKE_ts = []
morning = []
MKE_ts = []
ug_ts = []

# Range: JJA 2021-23, accounting for the range of HRRR data we have
# NEW: update to 2023/Sep 24, extend back to 2017
fdate = datetime.datetime(2022,10,1) # current extended data
date = datetime.datetime(2017,6,12)
# date = datetime.datetime(2017,5,1)
sd = '%04d%02d%02d' % (date.year,date.month,date.day)

error_tsh = 10.0 # error threshold

# Pre-Screen where the count starts
d = datetime.datetime(2017,6,12) 
all_dates = [d]
while d <= fdate:
    d = d + datetime.timedelta(days=1)
    if ((d.month >= 5) & (d.month < 10)):
        d_lasso = '%04d%02d%02d' % (d.year,d.month,d.day)
        all_dates += [d_lasso]
    else:
        continue
all_dates = np.array(all_dates)
#count_start = np.where(all_dates == sd)[0][0]
#count = count_start # start slice
count = 0

# Attribution Structure 
total = 0 # total day counter (after clear sky)
wind = 0 # days removed in the wind check
report = 0 # days removed in the reporting check
wsf = 0 # days removed by wsf

# Main Loop (edit on nc Data)
while date <= fdate:
    # Dates
    date = date + datetime.timedelta(days=1)
    if date.month < 5:continue # May-Sep, Jun-Aug for HRRR checks
    if date.month > 9:continue #####################################################################################
    # these days break the thing >:(
    if date == datetime.datetime(2021,7,30): continue
    if date == datetime.datetime(2021,8,28): continue
    if date == datetime.datetime(2022,6,17): continue
    if date == datetime.datetime(2022,6,18): continue
    if date == datetime.datetime(2022,6,25): continue
    if date == datetime.datetime(2022,8,8): continue
    if date == datetime.datetime(2022,8,9): continue
    if date == datetime.datetime(2022,9,27): continue
    if date == datetime.datetime(2022,9,1): continue
    if int(lst_std_ts[count]) == -9999: # Screen fill values
        count += 1
        continue
    count += 1
    #if date.year == 2020:continue
    #if date == datetime.datetime(2021,7,30):continue
    #if date == datetime.datetime(2021,8,28):continue
    #if '%04d%02d%02d' % (date.year,date.month,date.day) not in dates_lasso:continue
    print(date)

    # Pulling Data
    u = []
    v = []
    w = []
    ug = []
    if True == True:
        if (date.year < 2018) | (date.year > 2022):
            files = holes
        else: 
            files = arm1
        for file in files:
            doy = '%04d%02d%02d' % (date.year,date.month,date.day)
            if doy not in file:continue
            fp = nc.Dataset(file)
            dates = nc.num2date(fp['time'][:],units=fp['time'].units)
            
            ## THIS IS THE TIME FILTERING 
            
            #m = (dates >= datetime.datetime(date.year,date.month,date.day,15,0)) & (dates <= datetime.datetime(date.year,date.month,date.day,21,0)) # time of day
            m = (dates >= datetime.datetime(date.year,date.month,date.day,15,0)) & (dates <= datetime.datetime(date.year,date.month,date.day,17,0)) # time of day
            ################################################################################# 
            # Accounting for different sizes in time --> scaling together
            if fp['u'].shape[1] != 164:continue
            if dates.size == 144:
                tmp = fp['u'][m,:].data
                tmpu = upscale(tmp,6)
                tmp = fp['v'][m,:].data
                tmpv = upscale(tmp,6)
                tmp = fp['w'][m,:].data
                tmpw = upscale(tmp,6)
            if dates.size == 96:
                tmp = fp['u'][m,:].data
                tmpu = upscale(tmp,4)
                tmp = fp['v'][m,:].data
                tmpv = upscale(tmp,4)
                tmp = fp['w'][m,:].data
                tmpw = upscale(tmp,4)
            z = fp['height'][:]
            fp.close()
            rho = calculate_air_density(z,0,1000) #######################################################################
            ug.append((tmpu**2+tmpv**2)**0.5) # horizontal wind speed
            u.append(tmpu)
            v.append(tmpv)
            w.append(tmpw)
        u = np.array(u) # failing on July 30 2021
        v = np.array(v)
        w = np.array(w)
        ug = np.array(ug)
        
        # Screen, make sure the LiDARs are reporting at the same time (5 for 2017-22, 3 for 23-24)
        # Relax this next: only 3 instead of 5 (for all years)
        # relaxed again?
        nsamples = u != -9999
        if date.year > 2022:
            m = np.sum(nsamples,axis=0) < 3 ########################################################################################################
        else:
            m = np.sum(nsamples,axis=0) < 5
        #m = np.sum(nsamples,axis=0) < 3
        u[:,m] = -9999
        nsamples = v != -9999
        if date.year > 2022:
            m = np.sum(nsamples,axis=0) < 3
        else:
            m = np.sum(nsamples,axis=0) < 5
        #m = np.sum(nsamples,axis=0) < 3
        v[:,m] = -9999
        nsamples = w != -9999
        if date.year > 2022:
            m = np.sum(nsamples,axis=0) < 3
        else:
            m = np.sum(nsamples,axis=0) < 5
        #m = np.sum(nsamples,axis=0) < 3
        w[:,m] = -9999
        m = (u == -9999) | (v == -9999) | (w == -9999)
        u[m] = -9999
        v[m] = -9999
        w[m] = -9999

         # Masking
        u = np.ma.masked_array(u,u==-9999)
        v = np.ma.masked_array(v,v==-9999)
        w = np.ma.masked_array(w,w==-9999)
        ug = np.ma.masked_array(ug,ug==-9999)
        
        c5 = np.sum(ug[:,:,0] > 10) # wind screening, edit
        c10 = np.sum(ug[:,:,-1] > 20) ###########################################################################################################
        
        screened = 0
        '''
        if ((c5 > 0) | (c10 > 0)):
            u[:] = -9999
            v[:] = -9999
            w[:] = -9999
            wind += 1
            screened = 1
        '''
        
        # Herbie + Metpy Vort
        forecast = '%04d-%02d-%02d 15:00' % (date.year,date.month,date.day) # string to herbie
        H = Herbie(
            forecast, # datetime
            model='hrrr', # model
            product='sfc', # produce name?
            fxx=0, # lead time, want 0
            #save_dir='/home/pjg25/tyche/data'
            save_dir = troot+'herbie')
        
        try:
            ds = H.xarray(':[UV]GRD:', remove_grib=False)
            winds = ds[2]
            uvort = winds['u'].values[2,591:630,887:916] * (units.meter / units.second)
            vvort = winds['v'].values[2,591:630,887:916] * (units.meter / units.second)
            lat = winds['latitude'].values[591:630,0] 
            lon = winds['longitude'].values[0,887:916] 
            dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
            vort = mpcalc.vorticity(uvort, vvort, dx=dx, dy=dy)
            for pointer in ds:
                pointer.close()
        except FileNotFoundError:
            vort = 0
        except ValueError:
            vort = 0
        
        if screened == 1:
            pass
        elif type(vort) == int:
            u[:] = -9999
            v[:] = -9999
            w[:] = -9999
            wsf += 1
        else:
            vort_mag = np.abs(np.array(vort))
            vort_mask = vort_mag > 12e-5 ##############################################################################
            if np.sum(vort_mask) > 0:
                u[:] = -9999
                v[:] = -9999
                w[:] = -9999
                wsf += 1
        # Spatial Variance (DKE)
        up2 = np.var(u,axis=0,ddof=1)[:,:]
        vp2 = np.var(v,axis=0,ddof=1)[:,:]
        wp2 = np.var(w,axis=0,ddof=1)[:,:]
        # Spatial Mean and Squaring (MKE)
        u2 = np.mean(u,axis=0)[:,:]**2
        v2 = np.mean(v,axis=0)[:,:]**2
        w2 = np.mean(w,axis=0)[:,:]**2
        ug = np.mean(ug[:,:])
        # Adding
        DKE = up2 + vp2 + wp2
        DKEmorn = up2[:,0] + vp2[:,0] + wp2[:,0]
        MKE = u2 + v2 + w2
        #except: # Fills
    else:
        DKE = -9999
        MKE = -9999
        ug = -9999
    print(date,np.mean(DKE))
    #print(z)
    #mske_ts.append(np.mean(mske))

    # Vertical Integration(Sum)
    bDKE = np.mean(np.sum((z[1:,np.newaxis]-z[0:-1,np.newaxis])*rho[:-1,np.newaxis]*DKE.T[:-1,:],axis=0)/np.sum(rho))
    bDKEmorn = np.mean(np.sum((z[1:,np.newaxis]-z[0:-1,np.newaxis])*rho[:-1,np.newaxis]*DKEmorn.T[:-1],axis=0)/np.sum(rho))
    bMKE = np.mean(np.sum((z[1:,np.newaxis]-z[0:-1,np.newaxis])*rho[:-1,np.newaxis]*MKE.T[:-1,:],axis=0)/np.sum(rho))
    bDKE = np.ma.getdata(bDKE)
    bDKEmorn = np.ma.getdata(bDKEmorn)
    bMKE = np.ma.getdata(bMKE)
    #print(bDKE,bMKE)
    DKE_ts.append(bDKE)
    morning.append(bDKEmorn)
    MKE_ts.append(bMKE)
    #mske_ts.append(np.sum(rho*mske)/np.sum(rho))
    ug_ts.append(ug)
    total += 1
    time.sleep(0.25)
fcount = count # end slice
DKE_ts = np.array(DKE_ts)
#DKE_ts = np.ma.masked_array(DKE_ts,DKE_ts==-9999)
morning = np.array(morning)
MKE_ts = np.array(MKE_ts)
#MKE_ts = np.ma.masked_array(MKE_ts,MKE_ts==-9999)
ug_ts = np.array(ug_ts)

# %%
hetl=lst_lscale_ts.data[lst_lscale_ts>0][0:-5][DKE_ts>0]

# %%
color=plt.get_cmap('terrain')((np.array(MKE_ts[DKE_ts>0]))/4000)
plt.scatter(hetl,DKE_ts[DKE_ts>0],color=color)
plt.semilogy()

# %%
plt.scatter(hetl,1/MKE_ts[DKE_ts>0],color=color)
plt.semilogy()

# %%
plt.scatter(hetl,DKE_ts[DKE_ts>0]/MKE_ts[DKE_ts>0],color=color)
plt.semilogy()

# %%
plt.scatter(hetl,MKE_ts[DKE_ts>0])
plt.semilogy()

# %%
