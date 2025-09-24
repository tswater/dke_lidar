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
# UPDATE
# Syncing all data --> 2016 - 2023/24

# HRRR archive --> Herbie module (access archived HRRR data)
# https://herbie.readthedocs.io/en/stable/user_guide/tutorial/intro.html
# subsetting HRRR files before reading with xarray?

# GOES LST --> GOES data 2017, 23-24 needs sorting

# ARM Data --> 2016-22 acquired

# WRF-LES --> TBD


# %%
import glob
# Determine LASSO shallow convection days
dir = '/stor/tyche/hydro/shared/clasp/fr2'
dirs = glob.glob('%s/*' % dir)
dates_lasso = []
for dir in dirs:
    #if '2017' not in dir:continue
    tmp = dir.split('/')[-1][4:-3]
    if tmp == '20160530':continue
    dates_lasso.append(tmp)
print(dates_lasso)

# Similar concept with HRRR data --> WSF check
# Iterate and generate dts

# %%
import rasterio
import datetime
import numpy as np

d = datetime.datetime(2018,5,1)
dl = '%04d%02d%02d' % (d.year,d.month,d.day)
a = '/home/nc153/soteria/projects/DOE_BNR_2024/data/goes_lst_hourly/%s%02d00.tif' % (dl,15)
b = rasterio.open(a).read(1)
np.shape(b)

# %%
goes_path = '/stor/soteria/hydro/shared/data/GOES/GOES-16-EPSG4326/2024/'
#goes_all = glob.glob(goes_2017 + '/*')

# before the following
# figure out GOES, subset/transform as needed
# bundle together, follow

# test: iterate globs
# holes: 2017, 2023-24
extra = ['2017', '2023', '2024'] # May-Sep
files = []
for year in extra: # run globs
    # omitting transformation step
    add_glob = glob.glob(goes_path + year + '/*')
    files += add_glob

# read
#d = files[0]
#e = rasterio.open(d).read(1)

# file: OR_ABI-L2-LSTC-M4_G16_s{year}{day}{hour}50225_e{year}{day}{hour}50225_c{year}{day}{hour}57237.tif
# year = YYYY
# day = DDD (integer 1-365)
# hour = HH
#g = 'OR*s'
g = 'L2*s'
f = glob.glob(goes_path + g + '202424312*')
print(len(f))
# day numbers
# May 1 (non-leap)
sdate = datetime.datetime(2017,5,1)  # find earliest
edate = datetime.datetime(2023,9,30) # 273
s = sdate.timetuple().tm_yday 
e = edate.timetuple().tm_yday
print((s,e))

test = '/stor/soteria/hydro/shared/data/GOES/GOES-16-EPSG4326/%04d/%s%s%02d*' % (2017, 'OR*s', '2017230', 15)
yes = glob.glob(test)[0] # exists?

# %%
# Length Scales for GOES LST data 
import gstools as gs
from scipy.optimize import curve_fit

def exponential(x, a, b):
    return a * np.exp(-x/b)

def calculate_length_scale(z,undef=-9999):
    np.random.seed(1)
    ix = np.random.randint(0,z.shape[0],1000)
    iy = np.random.randint(0,z.shape[1],1000)
    x = 2000*ix
    y = 2000*iy
    field = z[iy,ix]
    m = file != undef
    ix = ix[m]
    iy = iy[m]
    field = field[m]
    #var = np.var(sh_array_microscale)#np.var(field)
    bins = np.linspace(0,40000,10)
    bin_center, gamma = gs.vario_estimate((x, y), field, bins)
    cov = np.max(gamma) - gamma
    opt, pcov = curve_fit(exponential, bin_center, cov,bounds=([0.9999*np.max(gamma),0],[np.max(gamma),40000]))
    #plt.plot(bin_center,cov)
    #model = gs.Exponential(dim=2,var=var)
    #fit_model.fit_variogram(bin_center, gamma, nugget=False)
    #ax = fit_model.plot(x_max=20000)
    #ax.scatter(bin_center, gamma)
    #plt.show()
    return opt[1]


# %%
# Screen Days and grab LST spatial statistics (redoing)

import glob
import os
import rasterio
import xarray as xr
import datetime
import scipy
import numpy as np
import netCDF4 as nc
lst_std_ts,lst_mean_ts,lst_lscale_ts = [],[],[] # Length Scales and Statistics
# Edit Start and End Dates (May to September)

# Toggle, try years, this appears to go through 2022 (20-22 next)
#fdate = datetime.datetime(2024,5,10) # end date
#date = datetime.datetime(2023,9,29) # start date
fdate = datetime.datetime(2024,10,1) # target
date = datetime.datetime(2017,6,12) # data starts at June 11, 2017? 

count = 0
while date <= fdate:
    date = date + datetime.timedelta(days=1)
    if date.month < 5:continue
    if date.month > 9:continue
    # May - Sep
    
    #if date.year == 2020:continue
    #if '%04d%02d%02d' % (date.year,date.month,date.day) not in dates_lasso:continue  
    if (date.year >= 2018) & (date.year <= 2022):
        date_lasso = '%04d%02d%02d' % (date.year,date.month,date.day) # 
    else:
        daynum = date.timetuple().tm_yday
        date_lasso = '%04d%03d' % (date.year, daynum)

    # Formatting
    #print(date_lasso)
    #for hour in range(18,24):
    
    lst_std,lst_mean,lst_lscale = [],[],[]
    for hour in range(15,16): #morning/near clear sky
        if (date.year >= 2018) & (date.year <= 2022):
            file = '/home/nc153/soteria/projects/DOE_BNR_2024/data/goes_lst_hourly/%s%02d00.tif' % (date_lasso,hour)
        elif date.year == 2017:
            pre = 'OR*s'
            file = '/stor/soteria/hydro/private/pjg25/GOES_holes/%s%s%02d*' % (pre, date_lasso, hour)
            file = glob.glob(file)
            if len(file) == 0:
                lst_std.append(-9999)
                lst_mean.append(-9999)
                lst_lscale.append(-9999)
                continue
            file = file[0]
        elif date.year >= 2023:
            pre = 'L2*s'
            file = '/stor/soteria/hydro/private/pjg25/GOES_holes/%s%s%02d*' % (pre, date_lasso, hour)
            file = glob.glob(file)
            if len(file) == 0:
                lst_std.append(-9999)
                lst_mean.append(-9999)
                lst_lscale.append(-9999)
                continue
            file = file[0]
                
        if os.path.exists(file):
            #print(file)
            data = rasterio.open(file).read(1)
            # data2 --> split surface and 500 hPa winds
            #print(np.sum(data==0),0.5*data.size)
            
            # Screening for undefined values (implicitly checking for clouds)
            # in the holes data --> nans
            if np.sum((data==0) | (data!=data)) > 0.1*data.size: ############################################################################################
                lst_std.append(-9999)
                lst_mean.append(-9999)
                lst_lscale.append(-9999)
                continue

           
            else: # Morning Sky Check Passed
             tmp = np.copy(data)
             tmp[tmp != tmp] = np.mean(tmp[tmp == tmp])
             tmp[tmp == 0] = np.mean(tmp[tmp != 0])
             lst_mesoscale = scipy.ndimage.gaussian_filter(tmp, sigma=3, mode='reflect') # smoother
             lst_microscale = tmp-lst_mesoscale
             meso_lh = calculate_length_scale(lst_mesoscale,undef=0)
             micro_lh = calculate_length_scale(lst_microscale,undef=0)
             '''plt.subplot(131)
             plt.imshow(tmp)
             plt.subplot(132)
             plt.imshow(lst_mesoscale)
             plt.subplot(133)
             plt.imshow(lst_microscale)
             plt.show()'''
             lh = calculate_length_scale(data,undef=0)
             #data = np.ma.masked_array(data,data==0)
             #lst_std.append(np.std(data))
             #lst_mean.append(np.mean(data))
             #lst_lscale.append(meso_lh)
             #lst_std.append(np.std(lst_mesoscale))
             #lst_mean.append(np.mean(lst_mesoscale))
             lst_lscale.append(meso_lh)
             lst_std.append(np.std(lst_mesoscale))
             lst_mean.append(np.mean(lst_mesoscale))
                
        else: # Fill if no file found
            lst_std.append(-9999)
            lst_mean.append(-9999)
            lst_lscale.append(-9999)
            
    lst_std = np.array(lst_std)
    lst_mean = np.array(lst_mean)
    lst_lscale = np.array(lst_lscale)
    #if np.sum(lst_std != -9999) > 0:print(date,np.mean(lst_std[lst_std != -9999]))
    lst_std = np.mean(lst_std[lst_std != -9999])
    lst_mean = np.mean(lst_mean[lst_mean != -9999])
    lst_lscale = np.mean(lst_lscale[lst_lscale != -9999])
    if np.isnan(lst_std) == 1:lst_std = -9999
    if np.isnan(lst_mean) == 1:lst_mean = -9999
    if np.isnan(lst_lscale) == 1:lst_lscale = -9999
    lst_std_ts.append(lst_std)
    lst_mean_ts.append(lst_mean)
    #print(date_lasso,(lst_lscale*lst_std)/lst_mean,count,len(lst_std_ts))
    lst_lscale_ts.append(lst_lscale)
    count += 1
    
lst_std_ts = np.array(lst_std_ts)
lst_mean_ts = np.array(lst_mean_ts)
lst_lscale_ts = np.array(lst_lscale_ts)
#files = glob.glob('/home/nc153/soteria/projects/DOE_BNR_2024/data/goes_lst_hourly/*.tif')
#print(lst_std_ts.shape)

# Write to netCDF for convenience
f = 'lst_dat.nc'
os.system('rm -f %s' % f)
fp = nc.Dataset(f, 'w')

# Dims
fp.createDimension('date')

# Vars
fp.createVariable('date', 'i8', dimensions=('date')) 
fp['date'].units = 'calendar date'
fp.createVariable('std', 'f', dimensions=('date'))
fp['std'].units = 'Kelvin'
fp.createVariable('mean', 'f', dimensions=('date'))
fp['mean'].units = 'Kelvin'
fp.createVariable('lscale', 'f', dimensions=('date'))
fp['lscale'].units = 'm'


# Write
# Dates
fd = fdate
d = date
dates = []
while d <= fd:
    d = d + datetime.timedelta(days=1)
    if ((d.month >= 5) & (d.month < 10)):
        d_lasso = '%04d%02d%02d' % (d.year,d.month,d.day)
        dates += [d_lasso]
    else:
        continue
fp['date'][:] = dates
fp['std'][:] = lst_std_ts
fp['mean'][:] = lst_mean_ts
fp['lscale'][:] = lst_lscale_ts
fp.close()


# %%
# Data above writen to netCDF file
#print(np.shape(lst_lscale_ts))
'''
f = 'lst_dat.nc'
os.system('rm -f %s' % f)
fp = nc.Dataset(f, 'w')

# Dims
fp.createDimension('date', size=1069)

# Vars
fp.createVariable('date', 'i8', dimensions=('date')) # RE-RUN THE WHOLE THING!!!!!
fp['date'].units = 'calendar date'
fp.createVariable('std', 'f', dimensions=('date'))
fp['std'].units = 'Kelvin'
fp.createVariable('mean', 'f', dimensions=('date'))
fp['mean'].units = 'Kelvin'
fp.createVariable('lscale', 'f', dimensions=('date'))
fp['lscale'].units = 'm'


# Write
# Dates
fd = datetime.datetime(2022,10,31)
d = datetime.datetime(2016,5,2)
dates = []
while d <= fd:
    d = d + datetime.timedelta(days=1)
    if ((d.month >= 5) & (d.month < 10)):
        d_lasso = '%04d%02d%02d' % (d.year,d.month,d.day)
        dates += [d_lasso]
    else:
        continue
fp['date'][:] = dates
fp['std'][:] = lst_std_ts
fp['mean'][:] = lst_mean_ts
fp['lscale'][:] = lst_lscale_ts
fp.close()
'''

# %%
# FROM NOW ON, START HERE
import glob
import netCDF4 as nc
import os
import rasterio
import xarray as xr
import datetime
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pywt

# 2017-2024 (1181 days, allegedly)y
tp = nc.Dataset('lst_dat.nc', 'r')
lst_std_ts = tp['std'][:]
lst_mean_ts = tp['mean'][:]
lst_lscale_ts = tp['lscale'][:]
datesnc = tp['date'][:]
tp.close()

#hets = lscales * (stds / means)

# Next, centralize all Lidar data, create the massive plot
# 2017-2024
# 2023 on --> only 3 lidars, change m condition
# accurately reference all data
# create our new and improved scatter
# incorporate Herbie

# %%
# boundary layer?
test = glob.glob('/home/nc153/soteria/ftp2.archive.arm.gov/chaneyn1/245518/*.nc') # ARM
file = test[0]
fp = nc.Dataset(file)

w = fp['w'][0][0:41]
z = fp['height'][0:41]
wprime = w - np.mean(w)

###########################################
arm1 = glob.glob('/home/nc153/soteria/ftp2.archive.arm.gov/chaneyn1/245518/*.nc') # ARM
holes = glob.glob('/stor/soteria/hydro/private/pjg25/ARM/*') # 2016-17, 23 (none for May 24)

#fdate = datetime.datetime(2024,10,1) # current extended data
fdate = datetime.datetime(2022,9,30)
date = datetime.datetime(2017,6,12)
sd = '%04d%02d%02d' % (date.year,date.month,date.day)

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
    #print(date)

    # Pull W data
    '''
    w = []
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
            m = (dates >= datetime.datetime(date.year,date.month,date.day,15,0)) & (dates <= datetime.datetime(date.year,date.month,date.day,17,0)) # time of day
        samples = fp['w'][m,:].data
    '''


# %%
# sgpsondewn and boundary layer
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# file pointers
sondewn = '/stor/soteria/hydro/private/pjg25/sgpsondewn/*202106*.11*'
g = glob.glob(sondewn)
g = sorted(g)
sfile = g[0]

# variables
# pres - pressure (hPa)
# tdry - dry bulb (deg C)
# dp - dew point (deg C)
# wind speed and direction
# rh - rel. humidity (%)
# u,v, ascent rate

def bulk_r(f):
    fp = nc.Dataset(f)
    # temp and pressure
    td = fp['tdry'][:]
    dew = fp['dp'][:]
    p = fp['pres'][:]

    # wind speeds
    uw = fp['u_wind'][:]
    vw = fp['v_wind'][:]
    alt = fp['alt'][:]

    x = (7.5 * dew) / (237.3 + dew)
    tv = (td + 273.15) / (1 - (0.379 * (6.11 * 10**x / (237.3 + dew)))) # virtual potential temp

    # Bulk Richardson
    u_diff = np.diff(uw)
    v_diff = np.diff(vw)
    z_diff = np.diff(alt)
    tv_diff = np.diff(tv)

    bulk = ((9.81 / tv[:-1]) * tv_diff * z_diff) / (u_diff**2 + v_diff**2)

    fp.close()

    return bulk, alt

# determine boundary layer height from this information
# incorporate this process into the big cell
# extend analysis to include metric with BLH
#bulk_r(sfile)



# %%
# thresholding
c1 = 0.25
c2 = 0.5

br = bulk_r(sfile)[0]
z = bulk_r(sfile)[1][:-1]
b1 = np.ma.getdata(br)
z1 = np.ma.getdata(z)

# Boundary Layer (consecutive exceedance)
def consec_exceed(array, val, cons): # 5-15 consecutive layers (~500 m)
    i = 0 
    for a in array:
        if a > val:
            i += 1
            if i >= cons:
                return True
        else:
            count = 0
    return False

def check_br(br_array, threshold, intervals):
    check = []
    #intervals = 15
    #threshold = 0.5
    for j in range(len(br_array)):
        check += [consec_exceed(br_array[0:10+j], threshold, intervals)]
    bi = np.min(np.where(np.array(check) == True))
    
    return bi # index of altitude array



# %%
#n1 = check_br(b1, 0.25, 25)
#z1[n1]
for day in g:
    b,z = bulk_r(day)
    s = find_peaks(b, height=0.5, distance=20)
    a = s[0][1]
    print(z[a])

# fidget with this method as much as you can

# %%

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
arm1 = glob.glob('/home/nc153/soteria/ftp2.archive.arm.gov/chaneyn1/245518/*.nc') # ARM
holes = glob.glob('/stor/soteria/hydro/private/pjg25/ARM/*') # 2016-17, 23 (none for May 24)
sonde = glob.glob('/stor/soteria/hydro/private/pjg25/sgpsondewn/*') # radiosonde (available for all days 2017-22
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
            save_dir = '/home/pjg25/soteria/herbie')
        
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
     # HRRR pointer calls (untested code)
    """
     if True == True:
      for file2 in files2:
        doy = '%04d%02d%02d' % (date.year,date.month,date.day)
        if doy not in file:continue 
        hfp = xr.open_dataset(file2, engine='cfgrib', filter_by_keys={'typeOfLevel':'isobaricInhPa'}, backend_kwargs={'indexpath':''}) # vertical
        hfp_s = xr.open_dataset(file2, engine='cfgrib', filter_by_keys={'stepType':'instant','typeOfLevel':'surface'}, backend_kwargs={'indexpath':''}) # at the surface
        hfp_a = xr.open_dataset(file2, engine='cfgrib', filter_by_keys={'stepType':'accum','typeOfLevel':'surface'}, backend_kwargs={'indexpath':''}) # accumulated
    """
    """ 
     # press = hfp.coords['isobaricInhPa'].values
     # p500 = np.where(press == 500)[0] # 500 mbar isobar
     # sp = hfp_s.variables['sp'].values[bl[0]:tr[0], bl[1]:tr[1]] / 100 # substitute values for the box (get elsewhere)
     # surface = np.zeros(np.shape(sp))
     # for i in range(0, np.shape(sp)[0]):
     #  for j in range(0, np.shape(sp)[1]):
     #   p_close_s = np.min(np.abs(sp[i][j] - press))
     #   surf_ind = np.where(p_close_s == np.abs(sp[i][j] - press))[0][0]
     #   surface[i][j] = surf_ind
    """
     # Vorticity
     # need to call HRRR file
     # vort = hfp['absv'][surface index,bl[0]:tr[0],bl[1]:tr[1]].values # s^-1 --> fill in box indices
     # check and fill
 
     # Precipitation / ShCu
     # precip = hfp_a['tp'][bl[0]:tr[0],bl[1]:tr[1]].values # mm, box values
     # check and fill

     # Fronts (requires more analysis)
     # Dew Point


# %%
# Set each of these and manipulate in the large cell below
set_DKE = DKE_ts
set_MKE = MKE_ts
set_morning = morning

# %%
# DKE threshold experiments
# 0 to high, start at 0-10 and increase by 10 to 150
# fixed window sizes (move from 0 to 150 or 200): 10, 25, 50?
# other ranges as needed

# Goal: 
# -iterate through and find the correlation at each window (with and without the WSF check active)
# -find the best performing criteria --> combine with literature to explain findings
# -use boundary layer stuff here as well
np.arange(10,150,10)

# %%
import seaborn as sns

def DKE_screen(dke,mke,morn,l,h):
    dke[dke == 0.0] = -9999
    mke[dke == -9999] = -9999
    morn[dke == -9999] = -9999

    DKE_ts_masked = dke[dke!=-9999]
    MKE_ts_masked = mke[dke!=-9999]
    morning_masked = morn[dke!=-9999]

    # x stuff
    x_mean = lst_mean_ts#[count_start:fcount]
    x_var = lst_std_ts**2#[count_start:fcount]**2
    x_lscale = lst_lscale_ts#[count:fcount]
    x_mean = x_mean[x_lscale!=-9999][:-5][dke!=-9999]
    x_var = x_var[x_lscale!=-9999][:-5][dke!=-9999]
    x_lscale = x_lscale[x_lscale!=-9999][:-5][dke!=-9999] # [:-5] is set for data to Oct 2022
    x = x_var**0.5*x_lscale/x_mean

    # pct change
    pct_change = ((DKE_ts_masked - morning_masked) / morning_masked) * 100
    
    DKE_ts_masked[(pct_change < l) | (pct_change > h)] = -9999
    MKE_ts_masked[(pct_change < l) | (pct_change > h)] = -9999
    morning_masked[(pct_change < l) | (pct_change > h)] = -9999

    DKE_ts_masked2 = DKE_ts_masked[DKE_ts_masked!=-9999]
    MKE_ts_masked2 = MKE_ts_masked[DKE_ts_masked!=-9999]
    morning_masked2 = morning_masked[DKE_ts_masked!=-9999]
    x_mask = x[DKE_ts_masked!=-9999] # baseline
    #x_mask_wind = x[DKE_ts_masked!=-9999] # wind removed
    #x_mask_wsf = x[DKE_ts_masked!=-9999] # wsf removed
    #x_mask_dke = x[DKE_ts_masked!=-9999] # baseline

    y_mask = DKE_ts_masked2/MKE_ts_masked2

    if len(x_mask) < 2:
        rho_pr = -9999
        rho_sr = -9999
    else:
        rho_pr = scipy.stats.pearsonr(x_mask,y_mask)[0]
        rho_sr = scipy.stats.spearmanr(x_mask,y_mask)[0]
    return rho_pr, rho_sr, len(x_mask)
#y_mask_wind = DKE_ts_masked2/MKE_ts_masked2
#y_mask_wsf = DKE_ts_masked2/MKE_ts_masked2
#y_mask_dke = DKE_ts_masked2/MKE_ts_masked2

#x = x_var

#print(scipy.stats.pearsonr(x,DKE_ts_masked)[0])
#plt.plot(x,DKE_ts_masked,'bo')
#plt.show()
#print(scipy.stats.pearsonr(x,MKE_ts_masked)[0])
#plt.plot(x,MKE_ts_masked,'bo')
#plt.show()

#rho_pr = scipy.stats.pearsonr(x_mask,y_mask)[0]
#rho_sr = scipy.stats.spearmanr(x_mask,y_mask)[0]
#plt.plot(x,DKE_ts_masked/MKE_ts_masked,'bo')
#plt.show()

DKE_screen(set_DKE, set_MKE, set_morning, 80, 90)

# %%
# Left set to zero, growing threshold to 150
b = np.arange(0, 210, 10)
b[1:]

# %%
# enumerate all combinations and check 
alltens = np.arange(0, 210, 10)

full_array = []

for n in range(len(alltens)-1):
    lowt = alltens[n]
    rest = alltens[n+1:]
    for m in range(len(rest)):
        hight = rest[m]
        results = DKE_screen(set_DKE, set_MKE, set_morning, lowt, hight)
        add_array = list(results) + [lowt, hight]
        full_array += [add_array]

    #DKE_screen(set_DKE, set_MKE, set_morning, 0, his[i])

# array line = pearson, spearmann, number of points, thresholds

# %%
# isolate to more than 10 data points
def find_bw(full, minpts, minwind=None):
    enough = []
    for j in range(len(full)):
        if full[j][2] < minpts:
            continue

        if full[j][4] - full[j][3] < minwind:
            continue            
        
        enough += [full[j]]
            

    # find best and worst
    full_list = []
    for k in range(len(enough)):
        #cond = enough[k][0]
        cond = (enough[k][0] + enough[k][1]) / 2
        full_list += [cond]

    max_condition = np.where(full_list == max(full_list))[0][0]
    min_condition = np.where(full_list == min(full_list))[0][0]
    best = enough[max_condition]
    worst = enough[min_condition]
    
    return best, worst


# %%
pts = np.arange(5,45,5)

for p in range(len(pts)):
    res = find_bw(full_array, pts[p],minwind=0)
    print(res)

# %%
for row in full_array:
    if (row[-1] <= 60) | (row[-2] == 0):
        print(row)

# %%
#create scatterplot with regression line and confidence interval lines
plt.figure(figsize=(12,8))
sns.regplot(x=x_mask,y=y_mask,scatter_kws={'s':100})
#sns.regplot(x=x_mask,y=DKE_ts_masked/MKE_ts_masked,scatter_kws={'s':100})
plt.ylabel(r'$\overline{DKE}/\overline{MKE}$',fontsize=25)
plt.xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s} \,(m)$',fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax = plt.gca()
ax.text(0.78, 0.95, r'$\rho_{P} = %.2f$' % rho_pr , transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
ax.text(0.78, 0.85, r'$\rho_{S} = %.2f$' % rho_sr , transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
plt.show()# 4 panel subplot as a placeholder

# %%
import matplotlib.gridspec as gridspec 

scatter_x = [x_mask, x_mask_wind, x_mask_wsf, x_mask_dke]
scatter_y = [y_mask, y_mask_wind, y_mask_wsf, y_mask_dke]
titles = ['Baseline', 'Wind Removed', 'WSF Removed', 'DKE Removed']

fig = plt.figure(figsize=(15,15))
G = gridspec.GridSpec(2,2)

for i in range(len(scatter_x)):
    col = i % 2
    row = i // 2
    plt.subplot(G[col,row])
    #
    rho_pr = scipy.stats.pearsonr(scatter_x[i],scatter_y[i])[0]
    rho_sr = scipy.stats.spearmanr(scatter_x[i],scatter_y[i])[0]
    sns.regplot(x=scatter_x[i],y=scatter_y[i],scatter_kws={'s':50})
    plt.ylabel(r'$\overline{DKE}/\overline{MKE}$',fontsize=15)
    plt.xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s} \,(m)$',fontsize=15)
    plt.title(titles[i], fontsize=18)
    ax = plt.gca()
    ax.text(0.75, 0.95, r'$\rho_{P} = %.2f$' % rho_pr , transform=ax.transAxes, fontsize=15,
        verticalalignment='top')
    ax.text(0.75, 0.85, r'$\rho_{S} = %.2f$' % rho_sr , transform=ax.transAxes, fontsize=15,
        verticalalignment='top')
plt.show()

# %%
# checking numbers
scheme = 'Base' #################################################################
all_days = 884 # every possible calendar day
report = wind + wsf + len(x)

print('Total Calendar Days: {}'.format(all_days))
print('Candidate Days: {}'.format(total)) # all candidate days (passed LST)
print('Days Passing Reporting Restriction: {}'.format(report)) # reporting check
print('Days Passing Wind Speed Restriction: {}'.format(report - wind)) # days removed by wind speed check
print('Days Passing WSF: {}'.format(len(x))) # days after all simulation checks
print('Days Passing Background DKE: {}'.format(len(x_mask))) # final

# Remove certain checks and see how the behavior changes, CHANGE BELOW LINE
checks[scheme] = {
    'total': total,
    'reporting': total - report,
    'wind': total - report - wind,
    'wsf': len(x),
    'dke': len(x_mask),
    'pearson': rho_pr,
    'spearmann': rho_sr
}

# %%
len(x)

# %%
xax = x
yax = DKE_ts_masked/MKE_ts_masked

# Print Days and pairs for ease
for i in range(len(xax)):
    print((xax[i], yax[i])) 


# %%
np.shape(x_mask[0][0])

# %%
# The above code screens days with minimal clouds in the morning (Cloud screen)
# Weak Synoptic Forcing needs to be included here (take what I've done and incorporate it)
# The rest of the code computes the integrated DKE and MKE from the ARM Data (currently 2018 and 2019)
# Use this and expand to other years

# %%
files = glob.glob('/home/nc153/soteria/ftp2.archive.arm.gov/chaneyn1/245519/*.cdf')
#files = glob.glob('/home/nc153/soteria/data/SGP/EC/*.cdf')
date = datetime.datetime(2016,5,1)
fdate = datetime.datetime(2019,10,31)
#date = datetime.datetime(2016,5,1)
#fdate = datetime.datetime(2017,10,31)
varsh_ts,meansh_ts = [],[]
fdate = datetime.datetime(2019,10,31)
date = datetime.datetime(2018,5,1)
#fdate = datetime.datetime(2017,10,31)
#date = datetime.datetime(2016,5,1)
error_tsh = 0.1
while date <= fdate:
    date = date + datetime.timedelta(days=1)
    if date.month < 6:continue
    if date.month > 9:continue
    #if '%04d%02d%02d' % (date.yea
    #if '%04d%02d%02d' % (date.year,date.month,date.day) not in dates_lasso:continue
    print(date)
    sh = []
    try:
        for file in files:
            doy = '%04d%02d%02d' % (date.year,date.month,date.day)
            if doy not in file:continue
            fp = nc.Dataset(file)
            dates = nc.num2date(fp['time'][:],units=fp['time'].units)
            m = (dates >= datetime.datetime(date.year,date.month,date.day,18)) & (dates <= datetime.datetime(date.year,date.month,date.day,23))
            if (np.sum(m) < 11):continue
            try:
                sh.append(-fp['sensible_heat_flux'][m])
                #plt.plot(-fp['sensible_heat_flux'][m])
                #plt.show()
                #break
            except:
                continue
    except:
        continue
    sh = np.array(sh)
    sh[sh > 1000] = 9999
    sh[sh < -1000] = 9999
    m == sh != 9999
    print(sh.shape)
    sh = np.ma.masked_array(sh,sh==9999)
    varsh = np.var(sh,axis=0,ddof=1)
    meansh = np.mean(sh,axis=0)
    #except:
    # varsh = -9999
    varsh_ts.append(np.mean(varsh))
    meansh_ts.append(np.mean(meansh))
varsh_ts = np.array(varsh_ts)
meansh_ts = np.array(meansh_ts)
'''
u = []
v = []
for file in files:
 if '20170920' not in file:continue
 fp = nc.Dataset(file)
 try:
  sh = -fp['sensible_heat_flux'][:]
  dates = nc.num2date(fp['time'][:],units=fp['time'].units)
  m = (dates >= datetime.datetime(2017,9,20,14)) & (dates <= datetime.datetime(2017,9,20,23))
  plt.plot(sh[m])
 except:
  continue
'''
''' dates = nc.num2date(fp['time'][:],units=fp['time'].units)
 m = (dates >= datetime.datetime(2017,9,19,10)) & (dates <= datetime.datetime(2017,9,19,18))
 if (np.sum(m) < 48):continue
 u.append(fp['u'][m,:])
 v.append(fp['v'][m,:])
u = np.array(u)
v = np.array(v)
u = np.ma.masked_array(u,u==-9999)
v = np.ma.masked_array(v,v==-9999)
up2 = np.var(u,axis=0)
vp2 = np.var(v,axis=0)
mske = up2 + vp2
mske = np.mean(mske[10:40])
print(mske)'''

# %%
import scipy.stats
#plt.imshow(np.flipud(mske.T))
#plt.colorbar(orientation='horizontal')
m = (mske_ts < 4) & (varsh_ts < 20000)
#m = (varsh_ts < 100000)
plt.figure(figsize=(10,5))
plt.plot(varsh_ts[m]**0.5,mske_ts[m],'bo',alpha=0.3,ms=10)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
print(scipy.stats.pearsonr(varsh_ts[m],mske_ts[m])[0])
print(scipy.stats.spearmanr(varsh_ts[m],mske_ts[m])[0])
plt.xlabel('$\sigma(H)$',fontsize=30)
plt.ylabel('MsKE',fontsize=30)
plt.title(r'$\rho = $' + '%.2f' % scipy.stats.pearsonr(varsh_ts[m],mske_ts[m])[0],fontsize=30)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(meansh_ts[m],mske_ts[m],'bo',alpha=0.3,ms=10)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
print(scipy.stats.pearsonr(meansh_ts[m],mske_ts[m])[0])
plt.xlabel('$\mu(H)$',fontsize=30)
plt.ylabel('MsKE',fontsize=30)
plt.title(r'$\rho = $' + '%.2f' % scipy.stats.pearsonr(meansh_ts[m],mske_ts[m])[0],fontsize=30)
plt.show()

#phtg = varsh_ts/meansh_ts
phtg = varsh_ts**0.5/meansh_ts#/ug_ts
plt.figure(figsize=(10,5))
plt.plot(phtg[m],mske_ts[m],'bo',alpha=0.3,ms=10)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
print(scipy.stats.pearsonr(phtg[m],mske_ts[m])[0])
print(scipy.stats.spearmanr(phtg[m],mske_ts[m])[0])
plt.xlabel('phtg',fontsize=30)
plt.ylabel('MsKE',fontsize=30)
plt.title(r'$\rho = $' + '%.2f' % scipy.stats.pearsonr(phtg[m],mske_ts[m])[0],fontsize=30)
plt.show()

# %%
import scipy.stats
#plt.imshow(np.flipud(mske.T))
#plt.colorbar(orientation='horizontal')
m = (mske_ts < 3) & (lst_std_ts != -9999) & (lst_lscale_ts != -9999)
#m = (varsh_ts < 100000)
plt.figure(figsize=(10,5))
plt.plot(lst_std_ts[m],mske_ts[m],'bo',alpha=0.3,ms=10)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
print(scipy.stats.pearsonr(lst_std_ts[m],mske_ts[m])[0])
print(scipy.stats.spearmanr(lst_std_ts[m],mske_ts[m])[0])
plt.xlabel('$\sigma(LST)$',fontsize=30)
plt.ylabel('MsKE',fontsize=30)
plt.title(r'$\rho = $' + '%.2f' % scipy.stats.pearsonr(lst_std_ts[m],mske_ts[m])[0],fontsize=30)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(lst_mean_ts[m],mske_ts[m],'bo',alpha=0.3,ms=10)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
print(scipy.stats.pearsonr(lst_mean_ts[m],mske_ts[m])[0])
print(scipy.stats.spearmanr(lst_mean_ts[m],mske_ts[m])[0])
plt.xlabel('$\mu(LST)$',fontsize=30)
plt.ylabel('MsKE',fontsize=30)
plt.title(r'$\rho = $' + '%.2f' % scipy.stats.pearsonr(lst_mean_ts[m],mske_ts[m])[0],fontsize=30)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(lst_lscale_ts[m],mske_ts[m],'bo',alpha=0.3,ms=10)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
print(scipy.stats.pearsonr(lst_lscale_ts[m],mske_ts[m])[0])
print(scipy.stats.spearmanr(lst_lscale_ts[m],mske_ts[m])[0])
plt.xlabel('$L(LST)$',fontsize=30)
plt.ylabel('MsKE',fontsize=30)
plt.title(r'$\rho = $' + '%.2f' % scipy.stats.pearsonr(lst_lscale_ts[m],mske_ts[m])[0],fontsize=30)
plt.show()


phtg = lst_std_ts/lst_mean_ts*lst_lscale_ts/ug_ts**2
#phtg = 9.8*lst_std_ts/lst_mean_ts/lst_lscale_ts/ug_ts**2
phtg = varsh_ts**0.5/meansh_ts*lst_lscale_ts#/ug_ts**2
plt.figure(figsize=(10,5))
plt.plot(phtg[m],mske_ts[m],'bo',alpha=0.3,ms=10)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
print(scipy.stats.pearsonr(phtg[m],mske_ts[m])[0])
print(scipy.stats.spearmanr(phtg[m],mske_ts[m])[0])
plt.xlabel('phtg',fontsize=30)
plt.ylabel('MsKE',fontsize=30)
plt.title(r'$\rho = $' + '%.2f' % scipy.stats.pearsonr(phtg[m],mske_ts[m])[0],fontsize=30)
plt.show()

# %%
import scipy.stats
#plt.imshow(np.flipud(mske.T))
#plt.colorbar(orientation='horizontal')
phtg = varsh_ts/meansh_ts


# %%
import glob
import numpy as np
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt

def calculate_air_density(z,minz,maxz): #hack
        T = 20 - 0.00649 * z;                        
        P = (101.29) * ((T + 273.15)/288.08)**(5.256);
        rho =  P/(0.2869*(T + 273.15));
        rho[z < minz] = 0.0
        rho[z > maxz] = 0.0
        return rho

@jit(nopython=True)
def upscale(yin,window):
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


files = glob.glob('/home/nc153/soteria/ftp2.archive.arm.gov/chaneyn1/245518/*.nc')
files2 = []
u = []
v = []
doy = '20170405'#dates_lasso[1]
date = datetime.datetime(int(doy[0:4]),int(doy[4:6]),int(doy[6:8]))
print(date)
u = []
v = []
ug = []
for file in files:
    doy = '%04d%02d%02d' % (date.year,date.month,date.day)
    if doy not in file:continue
    fp = nc.Dataset(file)
    dates = nc.num2date(fp['time'][:],units=fp['time'].units)
    print(dates)
    m = (dates >= datetime.datetime(date.year,date.month,date.day,12,0)) & (dates <= datetime.datetime(date.year,date.month,date.day,21,0))
    #print(np.sum(m))
    #if (np.sum(m) < 48):continue
    z = fp['height'][:]
    rho = calculate_air_density(z,0,100)
    error_tsh = 0.1
    if dates.size == 144:
        u_error = fp['u_error'][m,:]
        tmp = fp['u'][m,:]
        tmp[u_error > error_tsh] = -9999
        tmpu = upscale(tmp,6)
        v_error = fp['v_error'][m,:]
        tmp = fp['v'][m,:]
        tmp[v_error > error_tsh] = -9999
        tmpv = upscale(tmp,6)
        #tmpv = fp['v'][m,:][0::6,:]
    if dates.size == 96:
        u_error = fp['u_error'][m,:]
        tmp = fp['u'][m,:]
        tmp[u_error > error_tsh] = -9999
        tmpu = upscale(tmp,4)
        v_error = fp['v_error'][m,:]
        tmp = fp['v'][m,:]
        tmp[v_error > error_tsh] = -9999
        tmpv = upscale(tmp,4)
    ug.append((tmpu**2+tmpv**2)**0.5)
    u.append(tmpu)
    v.append(tmpv)
u = np.array(u)
v = np.array(v)
ug = np.array(ug)
nsamples = u != -9999
m = np.sum(nsamples,axis=0) < 5
u[:,m] = -9999
nsamples = v != -9999
m = np.sum(nsamples,axis=0) < 5
v[:,m] = -9999
m = (u == -9999) | (v == -9999)
u[m] = -9999
v[m] = -9999
u = np.ma.masked_array(u,u==-9999)
v = np.ma.masked_array(v,v==-9999)
ug = np.ma.masked_array(ug,ug==-9999)
up2 = np.var(u,axis=0)[:,:]
vp2 = np.var(v,axis=0)[:,:]
ug = np.mean(ug[:,0:10])
mske = up2 + vp2
print(mske.shape)
print(rho.shape,mske.shape)
print(mske[:,0])
print(rho)
print(date,np.mean(np.sum(rho[:,np.newaxis]*mske.T,axis=0)/np.sum(rho)))
#print(date,np.mean(mske))
#plt.subplot(211)
plt.imshow(np.flipud(mske.T))
plt.colorbar()
#plt.subplot(212)
#plt.imshow(np.flipud(vp2.T))
#plt.colorbar()

# %%

# %%
import numpy as np

# %%
a = np.array([[0,2,1],[2,3,4,1]],dtype='object')
print(a[0,:])

# %%
