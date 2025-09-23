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
import rasterio
dir = '/stor/soteria/hydro/shared/data/GOES_LST+SST/hourly'
file = '%s/%s' % (dir,'202109301100.tif')
da = rasterio.rasterio(file)

# %%
import glob
#retrieve files
files = glob.glob('/stor/soteria/hydro/shared/data/GOES_LST+SST/weekly/*.nc')
#print(files)
#file = '/stor/soteria/hydro/shared/data/GOES_LST+SST/weekly/GOES16(LST+SST)_2021_7day_s253_e259.nc'


# %%
#fp['LST'][0,:,:].plot()
lat = 34.37
lon = -87.35
minlat = lat + 0.5#34
maxlat = lat - 0.5#33
minlon = lon - 0.5#-90
maxlon = lon + 0.5#-89

# %%
import matplotlib.pyplot as plt
import datetime
import numpy as np
import xarray 
tmp = []
dates = []
for file in files[:]:
    if 'nccopy' in file:continue
    if '(LST+SST)' not in file:continue
    start = int(file.split('/')[-1].split('_')[3][1:])
    end = int(file.split('/')[-1].split('_')[4][1:].split('.')[0])
    if start < 121:continue
    if end > 273:continue
    print(file)
    fp = xarray.open_dataset(file)
    cfp = fp.sel(lat=slice(minlat,maxlat), lon=slice(minlon,maxlon))
    for i in range(cfp['LST'].shape[0]):
        cfp['LST']
        if (cfp['time.month'][i].data < 5) | (cfp['time.month'][i].data > 9):continue
        #if (cfp['time.hour'][i].data < 14) | (cfp['time.hour'][i].data > 20):continue
        if (cfp['time.hour'][i].data < 16) | (cfp['time.hour'][i].data > 16):continue
        m = np.isnan(cfp['LST'][i,:,:])==1
        if np.sum(m) > 0.01*m.size:continue
        dates.append(cfp['time'][i].data)
        tmp.append(cfp['LST'][i,:,:])
        #print(cfp['time.hour'][i])
        #plt.imshow(cfp['LST'][i,:,:],cmap='coolwarm')
        #plt.colorbar()
        #plt.show()
tmp = np.array(tmp)
dates = np.array(dates)
print(dates)
#fp

# %%
print(tmp.shape)
#print(tmp)

# %%
plt.subplot(121)
plt.imshow(np.nanmean(tmp,axis=0),cmap='coolwarm')#,vmin=300,vmax=305)
plt.colorbar(orientation='horizontal')
plt.subplot(122)
plt.imshow(np.nanstd(tmp,axis=0),cmap='coolwarm')#,vmin=1,vmax=3)
plt.colorbar(orientation='horizontal')

# %%
for i in range(tmp.shape[0]):
    print(dates[i])
    vmin = np.nanpercentile(tmp[i,:,:],5)
    vmax = np.nanpercentile(tmp[i,:,:],95)
    print(vmax-vmin)
    if (vmax-vmin)<6:continue
    plt.imshow(tmp[i,:,:],cmap='coolwarm',vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.show()

# %%
