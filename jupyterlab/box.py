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
# Plot for motivating future discussion
# Plot full bounding box
# All 5 LiDAR sites
# read LES data here

# LES
# Path: /stor/tyche/hydro/shared/clasp/fr2 (will change to soteria)
# Individual day folders
# 4D arrays = stuff we want
# Figure out indexes

# 00 = het
# 01 = hom

# %%
import numpy as np
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
import glob
import rasterio
import scipy as sp
import xarray as xr
import scipy

# %%
# One Set of LiDAR data
files = glob.glob('/home/nc153/soteria/ftp2.archive.arm.gov/chaneyn1/245518/*.nc')
files = files[0:4] + [files[5]] # one file from each site

# Read Latitude/Longitude
sites = [] # Latitude and Longitude of each site
site_loc = []
lats = []
lons = []

for file in files:
    sites += [file[-25:-22]] # site number
    fp = nc.Dataset(file)
    lat = np.array(fp['lat'])
    lon = np.array(fp['lon'])
    lats += [float(lat)]
    lons += [float(lon)]
    site_loc += [[float(lon), float(lat)]]


# %%
lst2 = glob.glob('/home/nc153/soteria/projects/DOE_BNR_2024/data/goes_lst_hourly/*1000.tif')
#lst = glob.glob('/stor/soteria/hydro/private/pjg25/GOES_holes/*201716715*')
#tiff = lst[0] # to illustrate
tiff2 = lst2[4]
tiff3 = '/home/nc153/soteria/projects/DOE_BNR_2024/data/goes_lst_hourly/201807091500.tif'
#data = rasterio.open(tiff).read(1) # change to 10 AM
data2 = rasterio.open(tiff3).read(1)
lst_mesoscale = scipy.ndimage.gaussian_filter(data2, sigma=3, mode='reflect')
# raster: [-97.9882090,37.1077130] to [-96.9882090,36.1077130]

# %%
lst_mesoscale

# %%
x = []
y = []
#im = plt.imshow(lst_mesoscale, cmap='coolwarm', interpolation='nearest', extent=[-97.9882,-96.9882,36.1077,37.1077])
im2 = plt.imshow(data2, cmap='coolwarm',extent=[-97.9882,-96.9882,36.1077,37.1077])
for n in range(0, len(sites)):
    lonx = lons[n]
    laty = lats[n]
    x += [lonx]
    y += [laty]
#plt.scatter(x,y, c='k')
plt.colorbar(im2, label='Temperature (K)')
plt.title('GOES-16 LST, 07/09/2018, 10 AM')
plt.show()

# %%
# Stats for the above
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


l_mean = np.mean(data2)
l_std = np.std(data2)
lh = calculate_length_scale(data2,undef=0)

# %%
h = (l_std*lh) / l_mean

print('Mean: {:0.2f}'.format(l_mean))
print('St. Deviation: {:0.2f}'.format(l_std))
print('Length Scale: {:0.2f}'.format(lh))
print('Het: {:0.2f}'.format(h))

# %%
# Looking ahead, thinking about sampling over this area
# LiDAR sampling, rough draft
bounds = [-97.9882090,37.1077130,-96.9882090,36.1077130]

# Do LHS and random sampling from a uniform distribution, compare
sampler = sp.stats.qmc.LatinHypercube(d=2) # LHS
un = np.random.uniform(size=2) # Uniform

# Distance Checker for restriction


# Transform to bounding box

# Stuff to think about:
# -do we want to constrain the spacing of the sites when sampling?
# -start with 5 sites, increase by one each time 
# -quantify uncertainty --> quantify 'performance'?
# -keep x, add y --> keep x sites from original, add y new ones

# %%
# N new sites (no prior info) 
def n_new_sites(n, sampler, plot=True, restrict=False): # Input: number of sites (n), sampler (LHS for now), plot, restrict
    xscale = -97.9882090  
    yscale = 36.1077130

    # Sample Step
    while 1:
        draw = list(sampler.random(n=n))
        for i in range(len(draw)): # transform
            draw[i][0] = draw[i][0] + xscale
            draw[i][1] = draw[i][1] + yscale
        # Restricter
        if restrict == False:
            break
        else:
            clear = True
            for n in range(len(draw)):
                pin = draw[n]
                other = draw[:n] + draw[n+1:]
                for m in range(len(other)):
                    dist = np.sqrt((pin[0] - other[m][0])**2 + (pin[1] - other[m][1])**2)
                    if dist < restrict:
                        clear = False
        if clear == False:
            continue
        else:
            break
    # 
    xs = []
    ys = []
    if plot == True:
        im2 = plt.imshow(data2, cmap='coolwarm', interpolation='none', extent=[-97.9882,-96.9882,36.1077,37.1077])
    for j in range(len(draw)):
        xs += [draw[j][0]]
        ys += [draw[j][1]]
    if plot == True:
        plt.scatter(xs,ys, c='k')
        plt.colorbar(im2,  label='Temperature (K)')
    return xs, ys

#n_new_sites(n=7, sampler=sampler, plot=True, restrict=0.2) # Plots map


# %%
# Keep N, pick M
# Randomly save N sites (2-4)
# Pick M new sites (3-5)

site_loc = site_loc
def keep_n_m_new(sites, n, m, sampler, plot=True, restrict=False):
    # Keep
    draw1 = np.random.choice(len(sites), n, replace=False)
    sites = np.array(sites)[draw1]
    
    xscale = -97.9882090  
    yscale = 36.1077130

    # New
    while 1:
        draw2 = list(sampler.random(m))
        for i in range(len(draw2)): # transform
            draw2[i][0] = draw2[i][0] + xscale
            draw2[i][1] = draw2[i][1] + yscale

        # Restricter
        if restrict == False:
            break
        else:
            clear = True
            for k in range(len(draw2)):
                pin = draw2[k]
                other = draw2[:k] + draw2[k+1:]
                for l in range(len(other)):
                    dist = np.sqrt((pin[0] - other[l][0])**2 + (pin[1] - other[l][1])**2)
                    if dist < restrict:
                        clear = False
        if clear == False:
            continue
        else:
            break
    
    sites = np.append(sites, draw2, axis=0)
    xs = []
    ys = []
    if plot == True:
        im3 = plt.imshow(data2, cmap='coolwarm', interpolation='none', extent=[-97.9882,-96.9882,36.1077,37.1077])
    for j in range(len(sites)):
        xs += [sites[j][0]]
        ys += [sites[j][1]]
    if plot == True: 
        plt.scatter(xs,ys, c='k')
        plt.colorbar(im3,  label='Temperature (K)')
    return xs, ys

#keep_n_m_new(site_loc, 4, 3, sampler, plot=True, restrict=0.15) # how many sites total?


# %%
# take sites from selector
# find points in the LES (try on HRRR for now)
# compute DKE from the artificial network

# Lat-lon to indices
def latlon(coords, pointer): # Inputs: lat,lon,file
    inds = []
    ds_lat = pointer.coords['latitude'].values
    ds_long = pointer.coords['longitude'].values
    lat = coords[0]
    long = coords[1]
    l = lat - ds_lat
    m = long - ds_long
    lat_ind = max(np.where(l > 0)[0])
    lon_ind = max(np.where(m > 0)[1])
    inds += [lat_ind, lon_ind]
    #return indices 
    return inds



# %%
# HRRR from JJA 2021-2023 --> will be LES
files = glob.glob('/stor/tyche/hydro/shared/HRRR/*.grib2')

# Site Selection from test file
# Site to index (HRRR), once
a_file = files[0]
a_pointer = xr.open_dataset(a_file, engine='cfgrib', filter_by_keys={'typeOfLevel':'isobaricInhPa'}, backend_kwargs={'indexpath':''})
site1 = keep_n_m_new(site_loc, 4, 3, sampler, plot=False)
site_ind = []
for i in range(len(site1[0])):
    latind, lonind = latlon([site1[1][i],site1[0][i]+360], a_pointer)
    site_ind += [[latind, lonind]]

# while loop 
# Date Range
# fdate

date = datetime.datetime(2021,6,20)
doy = '%04d%02d%02d' % (date.year,date.month,date.day)

us,vs,ws=[],[],[]
#DKEts,MKEts = [],[]
for hour in range(14,21):
    test = '/stor/tyche/hydro/shared/HRRR/hrrr2.' + doy + '.t' + str(hour) +'z.grib2'
    # Pointers
    hfp_a = xr.open_dataset(test, engine='cfgrib', filter_by_keys={'stepType':'accum','typeOfLevel':'surface'}, backend_kwargs={'indexpath':''})
    hfp_s = xr.open_dataset(test, engine='cfgrib', filter_by_keys={'stepType':'instant','typeOfLevel':'surface'}, backend_kwargs={'indexpath':''})
    hfp_iso = xr.open_dataset(test, engine='cfgrib', filter_by_keys={'typeOfLevel':'isobaricInhPa'}, backend_kwargs={'indexpath':''})

    u, v, w, ug,h,t,p = [],[],[],[],[],[],[]
    for site in site_ind:
        lat = site[0]
        lon = site[1]
    
    # Surface
    #press = hfp_iso.coords['isobaricInhPa'].values # Pressure coordinates
    #sp = hfp_s.variables['sp'].values[lat,lon] / 100
    #p_close_s = np.min(np.abs(sp - press))
    #surf_ind = np.where(p_close_s == np.abs(sp - press))[0][0]
    #surface = surf_ind
        surface = 3

        # Wind Profiles
        up = hfp_iso.variables['u'][surface+1:32,lat,lon].values # to top of troposphere (250 mb)
        vp = hfp_iso.variables['v'][surface+1:32,lat,lon].values
        wp = hfp_iso.variables['w'][surface+1:32,lat,lon].values
        # other
        press = hfp_iso.coords['isobaricInhPa'][surface+1:32].values * 100 # pressure
        geo = hfp_iso.variables['gh'][surface+1:32,lat,lon].values # gh
        temp = hfp_iso.variables['t'][surface+1:32, lat,lon].values # temp
        wspd = (up**2+vp**2)**0.5

        u.append(up)
        v.append(vp)
        w.append(wp)
        ug.append(wspd)   
        h.append(geo)
        t.append(temp)
        p.append(press)
    us.append(u)
    vs.append(v)
    ws.append(w)

# Spatial Variance
uvar = np.var(u,axis=0,ddof=1)
vvar = np.var(v,axis=0,ddof=1)
wvar = np.var(w,axis=0,ddof=1)

# Mean + Square
um = np.mean(u,axis=0)**2
vm = np.mean(v,axis=0)**2
wm = np.mean(w,axis=0)**2  



# %%
# DKE and MKE 
DKE = uvar + vvar + wvar
MKE = um + vm + wm

# density
ps = hfp_iso.coords['isobaricInhPa'][3+1:32].values * 100
ts = np.mean(t, axis=0)
hs = np.mean(h, axis=0)
rho = ps/((8.314 / 0.0289)*ts)

# Integrate
iDKE = np.sum((hs[1:]-hs[0:-1])*rho[:-1]*DKE.T[:-1]) / np.sum(rho)
iMKE = np.sum((hs[1:]-hs[0:-1])*rho[:-1]*MKE.T[:-1]) / np.sum(rho)

# %%
# Is this right?
# Start turning into a function?
# Calculate for each pointer, time average over days (need to add axes as in MsKE script)

# %%
draw = list(sampler.random(n=5))
list(draw[0])

# %%
# Lat_Lon Converter (LES)
xscale = [-97.9882090, -96.9882090]
yscale = [36.1077130, 37.1077130]
def les_lat_lon(coords, xscale, yscale, dimsize): # input = coordinates in lon,lat
    # dimsize = 520 for LES
    # dimsize = 41 for LST
    latspace = np.linspace(yscale[0], yscale[1], dimsize)
    lonspace = np.linspace(xscale[0], xscale[1], dimsize)
    # Nearest
    lat = coords[1] # south_north
    lon = coords[0] # west_east
    lati = np.min(np.where(latspace >= lat)[0])
    loni = np.min(np.where(lonspace >= lon)[0])
    return lati, loni # indexes in the LES

# Density Shit
# 226 vertical layers
def density(z,minz,maxz): #hack
        T = 20 - 0.00649 * z;                        
        P = (101.29) * ((T + 273.15)/288.08)**(5.256);
        rho =  P/(0.2869*(T + 273.15));
        rho[z < minz] = 0.0
        rho[z > maxz] = 0.0
        return rho



# %%
# LES
# Screening info will be carried over from previous task

# Flow (write out then format into a function)
# Read in LES data --> wind profiles
LES_path = '/stor/soteria/hydro/shared/LES_3D/clasp/fr2/fr2_20170924_00/*' # edit day
lesfiles = glob.glob(LES_path)
day = []
for file in lesfiles:
    if (int(file[-8:-6]) >= 15) & (int(file[-8:-6]) <= 21):
        day += [file]


# %%
# LESDKE Function --> calculate DKE and MKE from given sites over the LES

def LESDKE(sites, files, base=False): # input site indices and profiles
    z_top = 14.5e3 # meters, subject to change
    z_layer = 226
    zs = np.linspace(0,z_top,z_layer)
    #
    u_array = []
    v_array = []
    w_array = []
    for hour in files:
        fp = nc.Dataset(hour, 'r')
        u_wind = fp['AVV_U'] # Size: 226(bottom to top), 520(latitude), 520(longitude) 
        v_wind = fp['AVV_V']
        w_wind = fp['AVV_W']

        if base==True:
            u_array += [u_wind]
            v_array += [v_wind]
            w_array += [w_wind]
            continue

        # Access winds (repeat across hours)
        us = []
        vs = []
        ws = []
        for pair in sites:
            us += [u_wind[0,:,pair[0],pair[1]]]
            vs += [v_wind[0,:,pair[0],pair[1]]]
            ws += [w_wind[0,:,pair[0],pair[1]]]

        u_array += [us]
        v_array += [vs]
        w_array += [ws]

    # Averaging
    utm = np.mean(u_array, axis=0)
    vtm = np.mean(v_array, axis=0)
    wtm = np.mean(w_array, axis=0)

    # DKE
    uvar = np.var(utm, axis=0)
    vvar = np.var(vtm, axis=0)
    wvar = np.var(wtm, axis=0)

    DKE = uvar + vvar + wvar
    rhos = density(zs,0,1000) 
    bDKE = np.mean(np.sum((zs[1:,np.newaxis]-zs[0:-1,np.newaxis])*rhos[:-1,np.newaxis]*DKE[:],axis=0)/np.sum(rhos))

    # MKE
    umke = np.mean(utm, axis=0)**2
    vmke = np.mean(vtm, axis=0)**2
    wmke = np.mean(wtm, axis=0)**2        

    MKE = umke + vmke + wmke
    bMKE = np.mean(np.sum((zs[1:,np.newaxis]-zs[0:-1,np.newaxis])*rhos[:-1,np.newaxis]*MKE[:],axis=0)/np.sum(rhos))
  
    return bDKE / bMKE # return DKE/MKE or DKE


# %%
# Experiment Function (wind speed)
def LES_exp(LES_files, num, method=0, existing=None): # LES_files = hours from the day; num=number of sites; method = sampling algorithm (n_new by default, =0)
    sampler = sp.stats.qmc.LatinHypercube(d=2) # LHS
    # sample sites from method
    if method == 0:
        site_coord = n_new_sites(n=num, sampler=sampler, plot=False, restrict=0.25)
    elif method == 1:
        if existing == None:
            return 0
        m = 3 # sites to keep
        site_coord = keep_n_m_new(existing, m, num-m, sampler=sampler, plot=False, restrict=0.25)
    elif method == 2:
        site_coord = [list(np.array(existing)[:,0]), list(np.array(existing)[:,1])]
    
    # convert to indices
    xscale = [-97.9882090, -96.9882090]
    yscale = [36.1077130, 37.1077130]
    site_inds = []
    for n in range(len(site_coord)):
        coords = [site_coord[0][n], site_coord[1][n]]
        site_inds += [les_lat_lon(coords, xscale, yscale, dimsize=520)]

    # pass to LESDKE function
    value = LESDKE(site_inds, LES_files)
    
    return value


# %%
existing = site_loc
a = LES_exp(day, 10, method=1, existing=existing)
b = LES_exp(day, 10)
c = LES_exp(day, 5, method=2, existing=existing)
print(a)
print(b)
print(c)


# %%
site_loc

# %%
# LES Experiment
# Setup/Parameters
n_iter = 10 # number of iterations (per site number)
day = day # test file (1 day only)
num_sites = list(range(1,11)) # number of sites to iterate

# Outer for loop for number of sites
full_n = []
full_keep = []
for i in range(4,11): 
    # initialize array/list for respective site number
    test_runs_n = []
    test_runs_keep = []
    for j in range(0, n_iter):
        test_runs_n += [LES_exp(day, i)]
        test_runs_keep += [LES_exp(day, i, method=1, existing=existing)]
    full_n += [test_runs_n]
    full_keep += [test_runs_keep]

# report statistics of each run
# confidence interval/box-whisker over number of sites (normalize?)

# Report results

# Goal: increase iterations, see if convergence gets better
# try both sampling methods

# %%
# Plot
# ground truth: meshgrid the lat-lon into sites for LES_DKE 
# sites = entire grid, how to do this? (vertical part unchanged)
# size = 1/520 degrees --> take endpoints
x_ends = [-97.9882090, -96.9882090]
y_ends = [36.1077130, 37.1077130]
x_space = np.linspace(x_ends[0], x_ends[1], 520)
y_space = np.linspace(y_ends[0], y_ends[1], 520)
grid = np.meshgrid(x_space, y_space)
grid = np.reshape(grid, newshape=[2,520**2])
grid_sites = []
for n in range(len(grid[0])):
    grid_sites += [[grid[0][n], grid[1][n]]]
baseline = LES_exp(day, 5, method=2, existing=grid_sites)

plt.boxplot(full_n)
plt.axhline(baseline)
plt.show()

plt.boxplot(full_keep)
plt.axhline(baseline)
plt.show()


# %%
# Getting mixed results here
# Increase iterations
# look at DKE specifically (not the fraction)
# Try and find a LES day with high DKE
# Brainstorm
i

# %%
# Length Scales for GOES LST data 
import gstools as gs
from scipy.optimize import curve_fit
import scipy

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
# Sampling Experiment --> LST Heterogeneity
def LST_sample(file, num, method=0, existing=None):
    sampler = sp.stats.qmc.LatinHypercube(d=2) # LHS
    # sample sites from method
    if method == 0:
        site_coord = n_new_sites(n=num, sampler=sampler, plot=False, restrict=0.25)
    elif method == 1:
        if existing == None:
            return 0
        site_coord = keep_n_m_new(existing, num, m=2, sampler=sampler, plot=False, restrict=0.25)
        pass

    # convert to indices
    xscale = [-97.9882090, -96.9882090]
    yscale = [36.1077130, 37.1077130]
    site_inds = []
    for n in range(len(site_coord)):
        coords = [site_coord[0][n], site_coord[1][n]]
        site_inds += [les_lat_lon(coords, xscale, yscale, dimsize=41)]

    # statistics
    # process + subset
    data = rasterio.open(file).read(1)
    tmp = np.copy(data)
    tmp[tmp != tmp] = np.mean(tmp[tmp == tmp])
    tmp[tmp == 0] = np.mean(tmp[tmp != 0])
    lst_mesoscale = scipy.ndimage.gaussian_filter(tmp, sigma=3, mode='reflect')
    lstmsub = lst_mesoscale[np.array(site_inds)] # this line
    # length scale
    meso_lh = calculate_length_scale(lst_mesoscale,undef=0)
    
    # mean and std (from points only)
    lst_mean = np.mean(lstmsub)
    lst_std = np.std(lstmsub)

    het = (meso_lh * lst_std) / lst_mean
    
    return het


# %%
#LST_sample(sampler, 5)
file = glob.glob('/home/nc153/soteria/projects/DOE_BNR_2024/data/goes_lst_hourly/*.tif')[0]
data = rasterio.open(file).read(1)
a = scipy.ndimage.gaussian_filter(data, sigma=3, mode='reflect')
calculate_length_scale(a,undef=0)

# %%
LST_sample(file, 2)

# %%
