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
# LES Experiment

# Full and Robust Version


# %%
# Imports
import numpy as np
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import glob
import rasterio
import scipy as sp
import xarray as xr
import os
from sklearn.metrics import root_mean_squared_error
import pickle
import seaborn as sns

# %%
# File path + LES days
LES_path = '/stor/soteria/hydro/shared/LES_3D/clasp/fr2/'
all_days = [x[0] for x in os.walk(LES_path)]
het = '_00'
all_het = [s for s in all_days if het in s]

# Extract hour files from day
def day_files(path): # prescribe filepath
    day = glob.glob(path + '/*')
    hours = []
    for file in day:
        if (int(file[-8:-6]) >= 15) & (int(file[-8:-6]) <= 17):
            hours += [file]
    return hours


# %%
# Existing Network (if needed)
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

# site_loc = lons and lats of existing network

# %%
# Sampling
# Bounds
bounds = [-97.9882090,37.1077130,-96.9882090,36.1077130]
sampler = sp.stats.qmc.LatinHypercube(d=2) # Latin Hypercube

def n_new_sites(n, sampler, plot=True, restrict=False): # Input: number of sites (n), sampler , plot, restrict
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

#n_new_sites(n=5, sampler=sampler, plot=False, restrict=False)


# %%
# Converters
# Lat-Lon to indices
xscale = [-97.9882090, -96.9882090] # rewritten bounds
yscale = [36.1077130, 37.1077130]
def les_lat_lon(coords, xscale, yscale, dimsize=520): # input = coordinates in lon,lat
    # dimsize = 520 for LES
    latspace = np.linspace(yscale[0], yscale[1], dimsize)
    lonspace = np.linspace(xscale[0], xscale[1], dimsize)
    # Nearest
    lat = coords[1] # south_north
    lon = coords[0] # west_east
    lati = np.min(np.where(latspace >= lat)[0])
    loni = np.min(np.where(lonspace >= lon)[0])
    return lati, loni # indexes in the LES

def reverse(coords,xscale,yscale,dimsize=520):
    # dimsize = 520 for LES
    latspace = np.linspace(yscale[0], yscale[1], dimsize)
    lonspace = np.linspace(xscale[0], xscale[1], dimsize)
    # Nearest
    latc = latspace[coords[1]] # south_north
    lonc = lonspace[coords[0]] # west_east
    return latc, lonc # indexes in the LES

# Density with height
# 226 vertical layers
def density(z,minz,maxz): #hack
        T = 20 - 0.00649 * z;                        
        P = (101.29) * ((T + 273.15)/288.08)**(5.256);
        rho =  P/(0.2869*(T + 273.15));
        rho[z < minz] = 0.0
        rho[z > maxz] = 0.0
        return rho


# %%
# DKE Calculator 

# NOTES: CHANGE FROM 10 AM TO NOON IN ACCORDANCE WITH PREVIOUS ANALYSIS

def LES_reader(files): # pass file names, read arrays (u,v,w)
    u_array = []
    v_array = []
    w_array = []
    z_array = []
    rho_array = []
    # check this --> append each hour
    for hour in files:
        fp = nc.Dataset(hour, 'r')
        u_wind = fp['AVV_U'] # Size: 226(bottom to top), 520(latitude), 520(longitude) 
        v_wind = fp['AVV_V']
        w_wind = fp['AVV_W']
        #print('before',u_wind.shape)
        z_les = fp['AVP_Z']
        rho_les = fp['AVP_RHO']
        #plt.imshow(u_wind[0,24,:,:])
        #plt.show()

        #u = np.reshape(np.array(u_wind), newshape=(520,520,-1)) 
        #print('after',u.shape)
        #plt.imshow(u[:,:,24])
        #plt.show()
        #v = np.reshape(np.array(v_wind), newshape=(520,520,-1)) 
        #w = np.reshape(np.array(w_wind), newshape=(520,520,-1)) 
        u = np.moveaxis(u_wind[0,:,:,:],0,2)
        v = np.moveaxis(v_wind[0,:,:,:],0,2)
        w = np.moveaxis(w_wind[0,:,:,:],0,2)

        u_array += [u]
        v_array += [v]
        w_array += [w]
        z_array += [np.array(z_les)[0]]
        rho_array += [np.array(rho_les)[0]]

    return u_array, v_array, w_array, z_array, rho_array # selected profiles
    #return u_array

# pass u,v,w to LESDKE and calculate (single day?)
def LESDKE(u,v,w,z,rho,sites): # input profiles (before site selection)
    # check this for _all variables
    if sites==None:
        u_all = np.reshape(u, newshape=(-1,226))
        v_all = np.reshape(v, newshape=(-1,226))
        w_all = np.reshape(w, newshape=(-1,226))

    else:
        # Access Sites
        us = []
        vs = []
        ws = []
        for pair in sites:
            us += [u[pair[0],pair[1],:]]
            vs += [v[pair[0],pair[1],:]]
            ws += [w[pair[0],pair[1],:]]

        u_all = np.array(us)
        v_all = np.array(vs)
        w_all = np.array(ws)
    
    ztop = np.min(np.where(z > 1000)[0]) # meters, subject to change
    #z_layer = 226
    zs = z[0:ztop]
    #pull heights directly --> up to 1 km

    # DKE
    uvar = np.var(u_all, axis=0, ddof=1) # CHANGE ddof to 1
    vvar = np.var(v_all, axis=0, ddof=1)
    wvar = np.var(w_all, axis=0, ddof=1)

    DKE = uvar + vvar + wvar
    #rhos = density(zs,0,z[-1]) # change to top for trim comparison
    rhos = rho[0:ztop]
    bDKE = np.mean(np.sum((zs[1:,np.newaxis]-zs[0:-1,np.newaxis])*rhos[:-1,np.newaxis]*DKE[:],axis=0)/np.sum(rhos))

    # MKE
    umke = np.mean(np.abs(u_all), axis=0)**2
    vmke = np.mean(np.abs(v_all), axis=0)**2
    wmke = np.mean(np.abs(w_all), axis=0)**2        

    MKE = umke + vmke + wmke
    bMKE = np.mean(np.sum((zs[1:,np.newaxis]-zs[0:-1,np.newaxis])*rhos[:-1,np.newaxis]*MKE[:],axis=0)/np.sum(rhos))
    
    return bDKE / bMKE
    #return u_all


# %%
# Boundary Layer Height
# Temperature Gradient Method (Dai et al 2014)
# where gradient begins to spike (>1e-2?)

test_day = all_het[40]
tf = day_files(test_day)[0] # should be 15 for later

tp = nc.Dataset(tf, 'r')
vpt = tp['AVV_THV'][0,:,0,0] # at a point 
grad_vpt = np.diff(vpt)
top = 110
z = tp['AVP_Z'][0, 0:top]
threshold = 1e-2
bi = min(np.where(grad_vpt[0:top] > threshold)[0]) # first inversion (for morning only)
blh = z[bi]

plt.plot(vpt[0:top], z)
plt.xlabel(r'$\theta_v$')
plt.ylabel('z')
plt.show()


plt.plot(grad_vpt[0:top], z)
plt.vlines(0.01, 0,3500, colors='k')
plt.xlabel(r'$\partial \theta_v / \partial z$')
plt.ylabel('z')
plt.show()

# %%
blh

# %%
# Sanity Checks
s = all_het[-1]
sf = day_files(s)
a,b,c,d,e = LES_reader(sf)

# %%
# Check (all) 1000 network
path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
with open(path + 'nets.pkl', 'rb') as f:
    nets = pickle.load(f)

tn = nets[1000][0]
r2 = []
for l in range(len(a)):
    r_add = LESDKE(a[l], b[l], c[l], d[l], e[l], sites=tn)
    r2 += [r_add]
np.mean(r2)

# %%
#test = []
#for i in range(7):
#    test += [LESDKE(a[i], b[i], c[i], sites=None)]
y = LESDKE(a[1], b[1], c[1], d[1],sites=None)

# %%
y

# %%
np.var(np.abs(y[:,0]))

# %%
np.var(y[:,0])

# %%
# Network Generation
net_num = list(np.arange(3, 11, 1)) + list(np.arange(15, 105, 5)) + [500,1000]

all_nets = {} 
for n in net_num:
    test_nets = [] # indices
    iter = 100
    for i in range(iter):
        site_coords = n_new_sites(n=n, sampler=sampler, plot=False, restrict=False)
        xscale = [-97.9882090, -96.9882090]
        yscale = [36.1077130, 37.1077130]
        site_inds = []
        for m in range(len(site_coords[0])):
            coords = [site_coords[0][m], site_coords[1][m]]
            site_inds += [les_lat_lon(coords, xscale, yscale, dimsize=520)]
        test_nets.append(site_inds)
    all_nets[n] = test_nets

# Apply all networks to one day

# %%
# Write to pickle
path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
netfile = open(path + 'nets.pkl', 'wb')
pickle.dump(all_nets, netfile)
netfile.close()

# %%
plt.figure(figsize=(20,5))
plt.plot(DKEs, 'ko', label='Measured')
#plt.plot(bases, 'rs', label='Truth')
plt.legend(loc='best')
plt.title('Single Sample, All Days')
plt.show()

# %%
# Full Thing [DONT TOUCH]

# pre-generate networks (3-10, by 5 up to 100, 500, 1000)
# save to pickle file
# iterate through days
# read LES data
# apply all networks
# save output per day (pickle file) --> DKE values only

# Load Networks
path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
with open(path + 'nets.pkl', 'rb') as f:
    nets = pickle.load(f)

net_num = list(np.arange(3, 11, 1)) + list(np.arange(15, 105, 5)) + [500,1000]
for day in all_het:
    #read in data (READ ONCE, split data read from calculation in function, split LESDKE to two functions)
    LES_files = day_files(day)
    ud,vd,wd = LES_reader(LES_files)
    day_dict = {}
    for j in net_num:
        vals = []
        for network in nets[j]:
            sites = network
            dkstep = []
            for t in range(len(ud)):
                dkstep += [LESDKE(ud[t],vd[t],wd[t],sites)]
            vals += [np.mean(dkstep)]
        day_dict[j] = vals
    daystr = day[-11:-3]
    daypckl = open(path + daystr + '.pkl', 'wb')
    pickle.dump(day_dict, daypckl)
    daypckl.close()

# 3011 seconds (entire analysis)

# %%
# ABOVE, MODIFIED FOR DKE AND NEW Z SCHEME
path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
with open(path + 'nets.pkl', 'rb') as f:
    nets = pickle.load(f)
net_num = list(np.arange(3, 11, 1)) + list(np.arange(15, 105, 5)) + [500,1000]
for day in all_het:
    LES_files = day_files(day)
    #morning = [list(sorted(LES_files))[0]]
    #afternoon = [list(sorted(LES_files))[-2]]
    ud,vd,wd,zd,rhod = LES_reader(LES_files)
    #ua,va,wa,za,rhoa = LES_reader(afternoon)
    #um,vm,wm,zm,rhom = LES_reader(morning)
    day_dict = {}
    for j in net_num:
        vals = []
        for network in nets[j]:
            sites = network
            dkstep = []
            for t in range(len(ud)):
                dkstep += [LESDKE(ud[t],vd[t],wd[t],zd[t], rhod[t],sites)]
            #aft = LESDKE(ua[0],va[0],wa[0],za[0],rhoa[0],sites)
            #vals += [aft / morn]
            vals += [np.mean(dkstep)]
        day_dict[j] = vals
    daystr = day[-11:-3]
    daypckl = open(path + daystr + '.pkl', 'wb')
    pickle.dump(day_dict, daypckl)
    daypckl.close()

# %%
# Redo baseline calculations (Make another pickle)
bases = {}
for day in all_het:
    LES_files = day_files(day)
    daystr = day[-11:-3]
    ub,vb,wb = LES_reader(LES_files)
    calcs = []
    for t in range(len(ub)):
        calcs += [LESDKE(ub[t],vb[t],wb[t],sites=None)]
    bases[daystr] = np.mean(calcs)

# similar timing to previous cell
# Base Values [DONT TOUCH]
path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
basefile = open(path + 'base.pkl', 'wb')
pickle.dump(bases, basefile)
basefile.close()
# 1772 s

# %%
# MODIFIED BASELINE, Z SCHEME, AFTERNOON ONLY (fraction)
bases = {}
for day in all_het:
    LES_files = day_files(day)
    #morning = [list(sorted(LES_files))[1]]
    #afternoon = [list(sorted(LES_files))[-2]]
    daystr = day[-11:-3]
    #ubm,vbm,wbm,zbm,rhobm = LES_reader(morning)
    ub,vb,wb,zb,rhob = LES_reader(LES_files)
    vals = []
    for t in range(len(ub)):
        vals += [LESDKE(ub[t],vb[t],wb[t],zb[t],rhob[t], sites=None)]
        #vals += [np.mean(dkstep)]
    #calc_m = LESDKE(ubm[0],vbm[0],wbm[0],zbm[0],rhobm[0],sites=None)
    #calc_a = LESDKE(uba[0],vba[0],wba[0],zba[0],rhoba[0],sites=None)
    bases[daystr] = np.mean(vals)

# similar timing to previous cell
# Base Values [DONT TOUCH]
path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
#basefile = open(path + 'dke' + 'base.pkl', 'wb')
basefile = open(path + 'base.pkl', 'wb')
pickle.dump(bases, basefile)
basefile.close()

# %%
# Base Values [DONT TOUCH]
#path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
#basefile = open(path + 'base.pkl', 'wb')
#pickle.dump(bases, basefile)
#basefile.close()
len(rhob)

# %%
# RMSE Analysis (Almost there)
# Read base file (all values for each day)
path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
#base_read = open(path +'dke' + 'base.pkl', 'rb')
base_read = open(path + 'base.pkl', 'rb')
base_dict = pickle.load(base_read)
day_keys = base_dict.keys()
base_read.close()

# Restructure Data from Pickles
net_num = list(np.arange(3, 11, 1)) + list(np.arange(15, 105, 5)) + [500,1000]
new_dict = {} # store each iteration
for x in net_num:
    new_dict[x] = []
    for m in range(100):
        new_dict[x] += [[]]
full_base = []
for d in day_keys:
    full_base += [base_dict[d]]
    d_file = path + d + '.pkl'
    #d_file = path + d + 'dke' + '.pkl' # dke only
    d_read = open(d_file, 'rb')
    d_vals = pickle.load(d_read)
    #
    d_read.close()
    for y in net_num:
        for m in range(100):
            value = d_vals[y][m]
            new_dict[y][m] += [value] 

# final step = plot of RMSE statistics vs. site numbers

# %%

# %%
# Final Step
score_dict = {}
mean_norm_scores = {}
range_norm_scores = {}
for z in net_num:
    rmses = []
    mn_rmses = []
    r_rmses = []
    for k in range(100):
        one_iter = new_dict[z][k]
        rmse = root_mean_squared_error(full_base, one_iter)
        mean_norm = (rmse / np.mean(full_base)) * 100
        range_norm = (rmse / (np.max(full_base) - np.min(full_base))) * 100
        rmses += [rmse]
        mn_rmses += [mean_norm]
        r_rmses += [range_norm]
    score_dict[z] = rmses
    mean_norm_scores[z] = mn_rmses
    range_norm_scores[z] = r_rmses

# %%
# Plot
fig = plt.figure(num=1, figsize=(15,8),clear=True)
ax = plt.subplot(1,1,1)
ax.boxplot(score_dict.values())
ax.set_xticklabels(list(score_dict.keys()))
ax.set(
    ylabel='RMSE',
    xlabel='Sites in Network')
fig.savefig(path + 'rmse.png')

# %%
# Normalized RMSE (Mean)
fig = plt.figure(num=1, figsize=(15,8),clear=True)
ax = plt.subplot(1,1,1)
ax.boxplot(mean_norm_scores.values())
ax.set_xticklabels(list(mean_norm_scores.keys()))
ax.set_ylabel('Mean-Normalized RMSE (%)', fontsize=15)
ax.set_xlabel('Sites in Network', fontsize=15)
#fig.savefig(path + 'mn_rmse.png')
plt.show()

# %%
# Normalized RMSE (Range)
fig = plt.figure(num=1, figsize=(15,8),clear=True)
ax = plt.subplot(1,1,1)
ax.boxplot(range_norm_scores.values())
ax.set_xticklabels(list(range_norm_scores.keys()))
ax.set(
    ylabel='Range-Normalized RMSE (%)',
    xlabel='Sites in Network')
#fig.savefig(path + 'mn_rmse.png')
plt.show()

# %%
netsize = 5
net_id = 1
dk = new_dict[netsize][net_id]
path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
with open(path + 'nets.pkl', 'rb') as f:
    nets = pickle.load(f)
spec_net = nets[netsize][net_id]
plot_net = np.array(spec_net)
coord_net = []
for n in range(len(plot_net)):
    coord_net += [reverse(plot_net[n], xscale, yscale, dimsize=520)]
coord_net = np.array(coord_net)
# temperature overlay

fig_n = plt.figure(figsize=(20,6))
G = gridspec.GridSpec(1,2, width_ratios=[2.5,1])

plt.subplot(G[0,0])
plt.plot(dk, 'ko', label='Measured')
plt.plot(full_base, 'rs', label='Truth')
plt.legend(loc='best')
plt.title('Single Network over LES', fontsize=20)
plt.ylabel('$\overline{DKE} / \overline{MKE}$', fontsize=14)
plt.xlabel('LES Days', fontsize=14)

plt.subplot(G[0,1])
# convert to coordinates
plt.scatter(coord_net[:,1], coord_net[:,0], color='k')
#plt.imshow(, extent=(xscale[0], xscale[1], yscale[0], yscale[1]))
plt.title('Site Locations', fontsize=20)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.tight_layout()
plt.show()
# work gridspec here

# %%
xscale

# %%
# LES Het (funny name)
# Skin Temperature field --> AVT_TSK
# Correlation length (how to do this for lower site numbers?)
# St dev normalized by the mean

skin = day_files(all_het[4])[0]
fp = nc.Dataset(skin, 'r')
tsk = fp['AVS_TSK'] 
tsk = np.array(tsk)[0]

# %%
# Correlation length
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
    m = field != undef
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
corr = calculate_length_scale(tsk,undef=0)
het = corr * (np.std(tsk) / np.mean(tsk))
het

# %%
# Next experiment --> DKE vs. Het for all LES days (entire LES, we have the baseline DKE already)
# Narrow down to site specific (WITH ERROR BARS)
# Decide structure of subplots
# Run in full soon

hets = {}
#tvar = {}
corrs = {}
for day in all_het:
    morning = sorted(day_files(day))[0]
    fp = nc.Dataset(morning, 'r')
    tsk = fp['AVS_TSK'] 
    tsk = np.array(tsk)[0]
    fp.close()

    # Statistics
    corr = calculate_length_scale(tsk,undef=0)
    het = corr * (np.std(tsk) / np.mean(tsk))
    #tv = np.std(tsk)**2 # variance

    daystr = day[-11:-3]
    hets[daystr] = het
    corrs[daystr] = corr
    #tvar[daystr] = tv

# Write to Pickle
path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
hetfile = open(path + 'hets.pkl', 'wb')
pickle.dump(hets, hetfile)
hetfile.close()

corrfile = open(path + 'corrs.pkl', 'wb')
pickle.dump(corrs, corrfile)
corrfile.close()


# %%
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


# %%
# Calculate BLH (Bulk Richardson), emphasize how shaky this is
blh_vals = {}
for day in all_het:
    morning = sorted(day_files(day))[-1]
    fp = nc.Dataset(morning, 'r')

    # Need: VPT, u, v, z
    vpt = fp['AVV_THV'][0,:,:,:] # at a point --> chose central point?
    mean_vpt = np.mean(np.mean(vpt, axis=2), axis=1)
    uwind = fp['AVV_U'][0,:,:,:]
    mean_u_wind = np.mean(np.mean(uwind, axis=2), axis=1)
    vwind = fp['AVV_V'][0,:,:,:]
    mean_v_wind = np.mean(np.mean(vwind, axis=2), axis=1) 
    z = fp['AVP_Z'][0, :]

    # Calculate Rib
    vpt_diff = np.diff(mean_vpt)
    uw_diff = np.diff(mean_u_wind)
    vw_diff = np.diff(mean_v_wind)
    z_diff = np.diff(z)

    Rib = (9.81 * z_diff / mean_vpt[0:-1]) * (vpt_diff / (uw_diff**2 + vw_diff**2))
    
    check_r = []
    intervals = 15
    threshold = 0.5
    for j in range(len(Rib)):
        check_r += [consec_exceed(Rib[0:10+j], threshold, intervals)]
    bi = np.min(np.where(np.array(check_r) == True))
    blh = z[bi]

    daystr = day[-11:-3]
    blh_vals[daystr] = blh

    fp.close()

# %%
#blh_vals (morning)
a = list(blh_vals.values())
np.mean(a) # ~450-500 m tall

# %%
# # noon?
a = list(blh_vals.values())
np.mean(a)

# %%
check_r = []
intervals = 15
for j in range(len(Rib)):
    check_r += [consec_exceed(Rib[0:10+j], 0.5, intervals)]
minz = np.min(np.where(np.array(check_r) == True))
z[minz] # most reasonable

# %%
# Loading hets/corrs from pkl
path = '/stor/soteria/hydro/private/pjg25/THESIS/Pickle Network/'
het_read = open(path + 'hets.pkl', 'rb')
het_dict = pickle.load(het_read)
het_vals = list(het_dict.values())
het_read.close()

corr_read = open(path + 'corrs.pkl', 'rb')
corr_dict = pickle.load(corr_read)
corr_vals = list(corr_dict.values())
corr_read.close()

# %%
plt.scatter(list(blh_vals.values()), corr_vals)
x_range = np.linspace(0,2500,100)
plt.plot(x_range, 4*x_range)
plt.show()

# %%
daykey = list(het_dict.keys())
base_plot = []
tvar_plot = []
for key in daykey:
    base_plot += [base_dict[key]]
    tvar_plot += [len_vals[key]]

# %%
rho_pr = sp.stats.pearsonr(tvar_plot,base_plot)[0]
rho_sr = sp.stats.spearmanr(tvar_plot,base_plot)[0]

plt.figure(figsize=(12,8))
sns.regplot(x=tvar_plot,y=base_plot,scatter_kws={'s':100})
#plt.ylabel(r'$\overline{DKE}$',fontsize=25)
#plt.xlabel(r'$\sigma_{T_s}^2$',fontsize=25)
plt.ylabel(r'$\overline{DKE}/\overline{MKE}$',fontsize=25)
#plt.ylabel(r'$\overline{DKE}/\overline{DKE_{10 AM}}$',fontsize=25)
plt.xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s} \,(m)$',fontsize=25)
plt.title('LES, Simulated Truth', fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax = plt.gca()
ax.text(0.78, 0.95, r'$\rho_{P} = %.2f$' % rho_pr , transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
ax.text(0.78, 0.85, r'$\rho_{S} = %.2f$' % rho_sr , transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
plt.show()

# %%
# Error bar shit

# Iterate through networks 
net_errs = {}
net_means = {}
net_num = list(np.arange(3, 11, 1)) + list(np.arange(15, 105, 5)) + [500,1000]

# Get DKE estimate from each network for one day (max-min error bars)
for m in net_num:
    day_error = []
    day_mean = []
    for nd in range(len(all_het)):
        day_est = []
        for i in range(100):
            day_est += [new_dict[m][i][nd]]
        daysem = (np.max(day_est) - np.min(day_est)) / 2
        day_mean += [np.mean(day_est)]
        day_error += [daysem] 
    net_errs[m] = day_error
    net_means[m] = day_mean
    
# calculate error --> plot all with error bars

# %%
# Testing --> Big FIgure?
num = 1000
plot_dke = net_means[num]
plot_err = net_errs[num]
rho_pr = sp.stats.pearsonr(het_vals,plot_dke)[0]
rho_sr = sp.stats.spearmanr(het_vals,plot_dke)[0]

plt.figure(figsize=(12,8))
sns.regplot(x=het_vals,y=plot_dke,scatter_kws={'s':100})
plt.ylabel(r'$\overline{DKE}/\overline{MKE}$',fontsize=25)
plt.xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s} \,(m)$',fontsize=25)
plt.title('LES')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax = plt.gca()
ax.text(0.78, 0.95, r'$\rho_{P} = %.2f$' % rho_pr , transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
ax.text(0.78, 0.85, r'$\rho_{S} = %.2f$' % rho_sr , transform=ax.transAxes, fontsize=25,
        verticalalignment='top')

plt.errorbar(het_vals, plot_dke, yerr=plot_err, fmt='o')

plt.show()

# %%
# Try and format the above into a big figure

# Change: plot DKE vs. variance only
# add best fit lines (weighted least squares?)

import matplotlib.gridspec as gridspec 

site_size = [0, 3, 5, 7, 10, 100]
# 0 for truth, placeholder

# Figure dimensions, prescribe
fig = plt.figure(figsize=(20,10))
G = gridspec.GridSpec(2,3)

# 6 = 2 by 3
for n in range(len(site_size)):
    # Simulation truth in top left
    col = n % 3
    row = n // 3
    plt.subplot(G[row,col])
    if (col == 0) & (row == 0):
        plt.scatter(het_vals, full_base, marker='.')
        #sns.regplot(x=het_vals, y=full_base,scatter_kws={'s':10},line_kws=dict(color='k'))
        plt.ylabel(r'$\overline{DKE}/\overline{MKE}$')
        plt.xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s} \,(m)$')
        plt.ylim(0,0.25)
        plt.text(10,0.23,'Simulated Truth')

    else:
        net = site_size[n]
        #sns.regplot(x=het_vals, y=net_means[net],ci=100,scatter_kws={'s':10},line_kws=dict(color='k'))
        plt.errorbar(het_vals, net_means[net], yerr=net_errs[net], fmt='.')
        # err
        plt.ylabel(r'$\overline{DKE}/\overline{MKE}$')
        plt.xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s} \,(m)$')
        plt.ylim(0,0.25)
        plt.text(10,0.23,'%i sites' % net)
        
        
plt.show()

# %%
# Comparison of two plots of the same network? this is kind of already being shown?
fig2 = plt.figure(figsize=(12,6))
G2 = gridspec.GridSpec(1,2)

# One Run - 5 Sites
plt.subplot(G2[0,0])
dke5_1 = new_dict[5][35]
rho_pr = sp.stats.pearsonr(het_vals,dke5_1)[0]
rho_sr = sp.stats.spearmanr(het_vals,dke5_1)[0]
#plt.scatter(het_vals, dke5_1)
sns.regplot(x=het_vals,y=dke5_1,scatter_kws={'s':50})
plt.title(r'$\rho_{P} = %.2f$, $\rho_{S} = %.2f$' % (rho_pr, rho_sr))
plt.xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s} \,(m)$')
plt.ylabel(r'$\overline{DKE}/\overline{MKE}$')
plt.ylim([0,0.14])

# Another Run - 5 Sites
ax2 = plt.subplot(G2[0,1])
dke5_2 = new_dict[5][57]
rho_pr = sp.stats.pearsonr(het_vals,dke5_2)[0]
rho_sr = sp.stats.spearmanr(het_vals,dke5_2)[0]
#plt.scatter(het_vals, dke5_2)
sns.regplot(x=het_vals,y=dke5_2,scatter_kws={'s':50})
plt.title(r'$\rho_{P} = %.2f$, $\rho_{S} = %.2f$' % (rho_pr, rho_sr))
plt.xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s} \,(m)$')
plt.ylim([0,0.14])
plt.show()

# Add correlations here too
# we have the main idea

# %%
# iterate
size = 5
pearsons = []
spearmanns = []
for i in range(100):
    rhopr = sp.stats.pearsonr(het_vals,new_dict[size][i])[0]
    rhosr = sp.stats.spearmanr(het_vals,new_dict[size][i])[0]
    pearsons += [rhopr]
    spearmanns += [rhosr]
pmin = np.where(pearsons == np.min(pearsons))[0][0]
pmax = np.where(pearsons == np.max(pearsons))[0][0]
print(pmin)
print(pmax)

# %%
# Ideal network?
# given number of sites --> enumerate all possible configurations


# %%
# TRIM
trim_path = '/stor/soteria/hydro/shared/LES_3D/clasp/trim/*_00.nc'
trim_files = glob.glob(trim_path)

test_trim = trim_files[2]
fpt = nc.Dataset(test_trim, 'r')
tke = fpt['tke']
t = fpt['time']

# %%
# 2 PM is 48th time step
tkes = {}
for file in trim_files:
    trimfp = nc.Dataset(file, 'r')
    tke_array = np.array(trimfp['tke'])
    daystr = file[-14:-6]
    tkes[daystr] = tke_array[47]

# %%
daykey = list(tvar.keys())
tke_plot = []
tvar_plot = []
for key in daykey:
    tke_plot += [tkes[key]]
    tvar_plot += [tvar[key]]

# %%
rho_pr = sp.stats.pearsonr(tvar_plot,tke_plot)[0]
rho_sr = sp.stats.spearmanr(tvar_plot,tke_plot)[0]

plt.figure(figsize=(12,8))
sns.regplot(x=tvar_plot,y=tke_plot,scatter_kws={'s':100})
plt.ylabel(r'$\overline{DKE}$',fontsize=25)
plt.xlabel(r'$\sigma_{T_s}^2$',fontsize=25)
plt.title('Trim TKE')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax = plt.gca()
ax.text(0.78, 0.95, r'$\rho_{P} = %.2f$' % rho_pr , transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
ax.text(0.78, 0.85, r'$\rho_{S} = %.2f$' % rho_sr , transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
plt.show()

# %%
