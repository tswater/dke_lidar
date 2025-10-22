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
import rasterio
import os
from scipy import stats
matplotlib.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %% [markdown]
# # Trying to figure out pickle files 

# %%
proot='/home/tswater/projects/dke_lidar/'
troot='/home/tswater/tyche/data/dke_peter/'
with open(proot+'/old_pickle_network/20150606dke.pkl', 'rb') as file:
    p20150606dke = pickle.load(file)
with open(proot+'/old_pickle_network/20150606.pkl', 'rb') as file:
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
days=[]
for k in hets.keys():
    ht.append(hets[k])
    dk.append(dkebase[k])
    a=dkebase[k]/base[k]
    mk.append(a)
    rt.append(base[k])
    days.append(k)
rt=np.array(rt)
dk=np.array(dk)
mk=np.array(mk)
ht=np.array(ht)

# %%
color=plt.get_cmap('terrain')((np.array(mk)-900)/50000)
plt.scatter(ht,rt,c=color)
#plt.semilogy([10,.20],[.01,.02],alpha=0)
stats.spearmanr(ht,rt2)
#plt.ylim(0,0.04)

# %%
rt[7]

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
ds = H.xarray(':[UV]GRD:', remove_grib=False)

# %%
ds[2]

# %%
ds[2]['isobaricInhPa']

# %%

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

# %% [markdown]
# # Testing New Workflow Stuff

# %% [markdown]
# ### Testing GOES holes data

# %%
fp=rasterio.open(troot+'GOES_holes/OR_ABI-L2-LSTC-M3_G16_s20172140422189_e20172140424561_c20172140426101.tif','r')

# %%
fp.index(-97.9,36.3)

# %%
data=fp.read(1)

# %%
plt.imshow(data,cmap='coolwarm')

# %%
from scipy.ndimage import rotate

# %%
data[np.isnan(data)]=np.nanmean(data)
out=rotate(data,45,cval=float('nan'))

# %%
plt.imshow(out,cmap='coolwarm')

# %%
out[25,25]

# %%
fp.xy(35,35)

# %%
fpp=nc.Dataset(troot+'lidar_wind_profile/ARM/'+'sgpdlprofwind4newsC1.c1.20170515.000046.nc','r')
print('['+str(fpp['lon'][0].data)+', '+str(fpp['lat'][0].data)+']')

# %%
lonlat

# %%
data[yy[0],xx[0]]

# %%
fp.xy(30,30)

# %%
lonlat={'E37':[-97.927376, 36.3109],
        'E41':[-97.08639, 36.879944],
        'E32':[-97.81987, 36.819656],
        'E39':[-97.06912, 36.373775],
        'C1':[-97.48658, 36.605293]}

# %%
grad=np.gradient(data)
gradm=[np.nanmean(grad[0]),np.nanmean(grad[1])]

# %%
gradm


# %%
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi


# %%
angle_between([1,0],[-1,1])


# %%
def angle_diff(a1,a2):
    """ Returns angle indicating how far off of parallel these two angles are degrees"""
    abig=max(a1,a2)
    asml=min(a1,a2)
    if abig>180:
        abig=abig-180
    if asml>180:
        asml=asml-180
    b1=np.abs(abig-asml)
    if b1>90:
        b1=180-b1
    return b1


# %%

# %%
wdir='/home/tswater/tyche/data/dke_peter/lidar_wind_profile/245518/'
lena=[]
mina=[]
maxa=[]
a100=[]
da=[]
for file in os.listdir(wdir):
    fp=nc.Dataset(wdir+file,'r')
    alt=fp['height'][:]
    lena.append(len(alt))
    maxa.append(np.max(alt))
    mina.append(np.min(alt))
    da.append(alt[1]-alt[0])
    break
    try:
        a100.append(alt[100])
    except:
        print(file)

# %%
np.sum(a100!=2689.009)

# %%
np.sum(a100!=2689.009)

# %%
alt

# %%
alt

# %%
np.where(alt<1500)[0]

# %%
for v in fp.variables:
    print(v)

# %% [markdown]
# ### Testing NetCDF file

# %%
fp=nc.Dataset(troot+'lidar_lst_out.nc','r')

# %%
for v in fp.variables:
    data=fp[v][:]
    print(v)
    print('   '+str(data.shape))
    print('   '+str(np.nanmin(data)))
    print('   '+str(np.nanmean(data)))
    print('   '+str(np.nanmax(data)))
    print('   '+str(np.sum(np.isnan(data))/data.size))

# %%
from scipy.stats import circmean


# %%
def angle_diff(a1,a2):
    """ Returns angle indicating how far off of parallel these two angles are degrees"""
    bout=[]
    if hasattr(a1,'__len__'):
        pass
    else:
        a1=[a1]
        a2=[a2]
    for i in range(len(a1)):
        a10=a1[i]
        a20=a2[i]
        abig=max(a10,a20)
        asml=min(a10,a20)
        if abig>180:
            abig=abig-180
        if asml>180:
            asml=asml-180
        b10=np.abs(abig-asml)
        if b10>90:
            b10=180-b10
        bout.append(b10)
    if len(bout)==1:
        return bout[0]
    else:
        return np.array(bout)


# %%
numnan=np.zeros((36,11))
for i in range(36):
    height=fp['height'][:]
    for j in range(11):
        numnan[i,j]=np.sum(np.isnan(fp['DKE_xy'][:,j,i]))

# %%
plt.imshow(numnan,cmap='terrain',origin='lower')
ht=np.linspace(0,35,6).astype(int)
plt.yticks(ht,[height[ht[0]],height[ht[1]],height[ht[2]],height[ht[3]],height[ht[4]],height[ht[5]]])
plt.colorbar()

# %%
h0=5
h=25
winda=circmean(fp['wind_a'][:,2:4,h0:h],axis=(1,2),nan_policy='omit')
lsta=fp['lst_a'][:]
ws=np.nanmean(fp['wind_speed'][:,2:4,h0:h],axis=(1,2))
abet=angle_diff(winda,lsta)
#color=plt.get_cmap('terrain')(fp['lst_std_site'][:]/2)
#color=plt.get_cmap('terrain')(ws/10)
color=plt.get_cmap('terrain')((abet)/90)
dke=np.mean(np.sum(fp['DKE_xy'][:,2:4,h0:h],axis=2),axis=1)
mke=np.mean(np.sum(fp['MKE_xy'][:,2:4,h0:h],axis=2),axis=1)
lhet=fp['lst_lhet'][:]
lstd_site=fp['lst_std_site'][:]
lstd=fp['lst_std'][:]
plt.scatter(lhet*lstd,dke/mke,color=color)
plt.semilogy()

# %%
h=15
dke=np.mean(np.sum(fp['DKE_z'][:,2:4,0:h],axis=2),axis=1)
mke=np.mean(np.sum(fp['MKE_z'][:,2:4,0:h],axis=2),axis=1)
#lhet=fp['lst_lhet'][:]
plt.scatter(lhet,dke/mke,color=color)
plt.semilogy()

# %%
np.nanpercentile(fp['lst_std_site'],5)

# %%
h=15
winda=circmean(fp['wind_a'][:,2:4,0:h],axis=(1,2),nan_policy='omit')
lsta=fp['lst_a'][:]
ws=np.nanmean(fp['wind_speed'][:,2:4,0:h],axis=(1,2))
abet=angle_diff(winda,lsta)
#color=plt.get_cmap('terrain')(fp['lst_std_site'][:]/2)
#color=plt.get_cmap('terrain')(ws/10)
color=plt.get_cmap('terrain')((abet)/90)
dke0=np.mean(np.sum(fp['DKE_z'][:,0:2,0:h],axis=2),axis=1)
mke0=np.mean(np.sum(fp['MKE_z'][:,0:2,0:h],axis=2),axis=1)
dke=np.mean(np.sum(fp['DKE_z'][:,2:8,0:h],axis=2),axis=1)
mke=np.mean(np.sum(fp['MKE_z'][:,2:8,0:h],axis=2),axis=1)
rt0=dke0/mke0
rt=dke/mke0
delta=(rt-rt0)/rt0*100
lhet=fp['lst_lhet'][:]
lstd_site=fp['lst_std_site'][:]
lstd=fp['lst_std'][:]

m=(ws<5)&(abet>22.5)

plt.scatter(lhet[m]*lstd[m],(dke[m]-dke0[m])/mke[m],color=color[m])
#plt.semilogy()
#plt.ylim(0,500)

# %%
for i in range(20):
    plt.figure()
    data=fp['DKE_xy'][i,:,:]/fp['MKE_xy'][i,:,:]
    plt.imshow(data.T,origin='lower',cmap='terrain',extent=(8,24,0,3),vmin=0,vmax=10)
    plt.title(str(ws[i]))

# %%
h=25
winda=circmean(fp['wind_a'][:,2:4,0:h],axis=(1,2),nan_policy='omit')
lsta=fp['lst_a'][:]
ws=np.nanmean(fp['wind_speed'][:,2:4,0:h],axis=(1,2))
abet=angle_diff(winda,lsta)
#color=plt.get_cmap('terrain')(fp['lst_std_site'][:]/2)
color=plt.get_cmap('terrain')(ws/10)
#color=plt.get_cmap('terrain')((abet)/90)
dk=np.mean(np.sum(fp['DKE_xy'][:,2:4,0:h],axis=2),axis=1)
mk=np.mean(np.sum(fp['MKE_xy'][:,2:4,0:h],axis=2),axis=1)
lhet=fp['lst_lhet'][:]
lstd_site=fp['lst_std_site'][:]
lstd=fp['lst_std'][:]

m=(ws<5)&(abet>45)

dke0=np.mean(np.sum(fp['DKE_z'][:,1:3,0:h],axis=2),axis=1)
mke0=np.mean(np.sum(fp['MKE_z'][:,1:3,0:h],axis=2),axis=1)
dke=np.mean(np.sum(fp['DKE_z'][:,3:6,0:h],axis=2),axis=1)
mke=np.mean(np.sum(fp['MKE_z'][:,3:6,0:h],axis=2),axis=1)

plt.scatter(lhet[m],((dke-dke0)/(mke))[m],color=color[m])
#plt.semilogy()
#plt.ylim(0,1)

# %%
dke=np.nanmedian(fp['DKE_xy'][:],axis=0)
dke_mke=np.nanmedian(fp['DKE_xy'][:]/fp['MKE_xy'][:],axis=0)

# %%
dkez=np.nanmedian(fp['DKE_z'][:],axis=0)
dke_mkez=np.nanmedian(fp['DKE_z'][:]/fp['MKE_z'][:],axis=0)

# %%
data=dke
plt.imshow(data.T,origin='lower',cmap='terrain',extent=(8,24,0,3),vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95))

# %%
data=dke_mke
plt.imshow(data.T,origin='lower',cmap='terrain',extent=(8,24,0,3),vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95))

# %%
data=dkez
plt.imshow(data.T,origin='lower',cmap='terrain',extent=(8,24,0,3),vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95))

# %%
data=dke_mkez
plt.imshow(data.T,origin='lower',cmap='terrain',extent=(8,24,0,3),vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95))

# %%
data=dke_mke-dke_mke[:,0][:,np.newaxis]
plt.imshow(data.T,origin='lower',cmap='terrain',extent=(8,24,0,3),vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95))

# %%
data=dke_mkez-dke_mkez[:,0][:,np.newaxis]
plt.imshow(data.T,origin='lower',cmap='terrain',extent=(8,24,0,3),vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95))

# %%

# %% [markdown]
# ### Testing Correlations

# %%
corr=np.zeros((3,3,12,3,3,3))
numpnt=np.zeros((3,3,12,3,3,3))

DKExy=fp['DKE_xy'][:]
DKEz=fp['DKE_z'][:]
MKExy=fp['MKE_xy'][:]
MKEz=fp['MKE_z'][:]
lhet=fp['lst_lhet'][:]*fp['lst_std'][:]/fp['lst_mean'][:]
for i in range(3): # xy, z, xyz
    dke_=[DKExy,DKEz,DKEz+DKExy][i]
    mke_=[MKExy,MKEz,MKEz+MKExy][i]
    dke_=dke_.data
    mke_=mke_.data
    for j in range(3): # DKE, DKE/MKE, DKE-DKE0/MKE
        data=[dke_,dke_/mke_,(dke_/mke_)-(dke_[:,1,:][:,np.newaxis,:]/mke_[:,1,:][:,np.newaxis,:])][j]
        for t in range(3): # Morn After Both
            tmin=[2,4,2][t]
            tmax=[4,8,8][t]
            for k in range(3): # Hmin
                h0=[0,3,6][k]
                for l in range(3): # Hmax
                    h1=[16,25,35][l]
                    a=np.sum(fp['sites_repo'][:,tmin:tmax,h0:h1,:],axis=3)
                    a=np.mean(a,axis=(1,2))
                    ws=np.mean(fp['wind_speed'][:,tmin:tmax,0:5],axis=(1,2))
                    winda=circmean(fp['wind_a'][:,tmin:tmax,0:5],axis=(1,2),nan_policy='omit')
                    lsta=fp['lst_a'][:]
                    abet=angle_diff(winda,lsta)
                    vort=fp['vort'][:]
                    for f in range(12):
                        if f==0:
                            m=(vort<12e-5)
                        elif f in [1,2,3]:
                            m=(a>=[3,4,5][f-1])
                        elif f in [4,5,6]:
                            m=(ws<[2.5,5,10][f-4])
                        elif f in [7,8]:
                            m=abet>[45,75][f-7]
                        elif f in [9,10,11,12]:
                            m=(vort<12e-5)&(a>=3)
                            if f in [10,12]:
                                m=m&(ws<10)
                            if f in [11,12]:
                                m=m&(abet>22.5)
                        m=m.data
                        xx=lhet[m]
                        yy=np.sum(data[:,tmin:tmax,h0:h1],axis=2)
                        yy=np.nanmean(yy,axis=1)
                        yy=yy[m]

                        m2=(~np.isnan(xx))&(~np.isnan(yy))
                        
                        
                        corr[i,j,f,t,k,l]=stats.spearmanr(xx[m2],yy[m2])[0]
                        numpnt[i,j,f,t,k,l]=np.sum(m2)
                            
                            
                          

# %%
for i in range(3):
    for j in range(3):
        plt.figure()
        a=np.reshape(corr[i,j,:],(12,27))
        plt.imshow(a,vmin=-.5,vmax=.5,cmap='coolwarm')
        plt.colorbar()
        plt.title(['DKExy','DKEz','DKExyz'][i]+' '+['d','d/m','d-d/m'][j])

# %%
sm=0
for t in range(3): # Morn After Both
    if sm==16:
        break
    tmin=[2,4,2][t]
    tmax=[4,8,8][t]
    for k in range(3): # Hmin
        h0=[0,3,6][k]
        for l in range(3): # Hmax
            h1=[16,25,35][l]
            sm=sm+1

# %%
tmin

# %%
tmax

# %%
h0

# %%
h1

# %%
for i in range(3):
    for j in range(3):
        plt.figure()
        a=np.reshape(numpnt[i,j,:],(12,27))
        plt.imshow(a,vmin=0,vmax=275,cmap='terrain')
        plt.colorbar()
        plt.title(['DKExy','DKEz','DKExyz'][i]+' '+['d','d/m','d-d/m'][j])

# %%

# %%
nmrpt=np.nanmean(np.nansum(fp['sites_repo'][:],axis=3),axis=(1,2))
color=plt.get_cmap('coolwarm')((nmrpt-2)/3)
plt.scatter(fp['lst_std_site'][:],fp['lst_std'][:],s=10,c=color)
plt.plot([0,5],[0,5],'k--',alpha=.5)

# %%
