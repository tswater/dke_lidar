import numpy as np
import glob
import numba
from numba import jit
from herbie import Herbie
import pandas as pd
import metpy.calc as mpcalc # calculate vorticity ourselves
from metpy.units import units
from datetime import datetime,timedelta
import scipy
from scipy.optimize import curve_fit
from scipy.stats import circmean
import netCDF4 as nc
import rasterio
import os
import gstools as gs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#########################################################
############### FOLDERS ETC #############################
parmdir='/home/tswater/tyche/data/dke_peter/lidar_wind_profile/ARM/' # some of the ARM profiles
narmdir='/home/tswater/tyche/data/dke_peter/lidar_wind_profile/245518/' # the other ones
g17dir ='/home/tswater/tyche/data/dke_peter/GOES_holes/' # GOES data from 2017
g18dir ='/home/tswater/tyche/data/dke_peter/goes_lst_hourly/' # GOES data from other years
troot  ='/home/tswater/tyche/data/dke_peter/'
tempdir=troot+'aeri_temp/'
fnc    =nc.Dataset(troot+'lidar_lst_out.nc','r')
debug  = True

##########################################################
###################### FUNCTIONS #########################
def exponential(x, a, b):
    return a * np.exp(-x/b)

def shape2dims(data,dims):
    dimout=()
    for d in data.shape:
        dimout=dimout+(dims[d],)
    return dimout

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
    bins = np.linspace(0,40000,10)
    bin_center, gamma = gs.vario_estimate((x, y), field, bins)
    cov = np.max(gamma) - gamma
    opt, pcov = curve_fit(exponential, bin_center, cov,bounds=([0.9999*np.max(gamma),0],[np.max(gamma), 40000]))
    return opt[1]

lonlat={'E37':[-97.927376, 36.3109],
        'E41':[-97.08639, 36.879944],
        'E32':[-97.81987, 36.819656],
        'E39':[-97.06912, 36.373775],
        'C1':[-97.48658, 36.605293]}

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

def upscale(yin,window,isangle=False): # Upscaling Data?
    #print(np.unique(yin))
    i = 0
    count = 0
    yout = 0
    nt = int(yin.shape[0]/window)
    yout = np.ones((nt,yin.shape[1]))*float('nan')
    while i < yin.shape[0]:
        for j in range(yin.shape[1]):
            m1 = yin[i:i+window,j] != -9999
            if np.sum(m1) > 0:
                if isangle:
                    rad = np.deg2rad(yin[i:i+window,j][m1])
                    yout[count,j]=np.rad2deg(circmean(rad))
                else:
                    yout[count,j] = np.mean(yin[i:i+window,j][m1])


        i = i + window
        count += 1
    return yout

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    if (np.sum(np.isnan(v1))+np.sum(np.isnan(v2)))>0:
        return float('nan')
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi

##########################################################
################ GOING ##################################
# Generate a list of available days
flist={} # a dictionary of day:{'ARM':[list_of_profile_paths],'GOES':[list_of_goes_paths]} pairs

for file in os.listdir(g17dir):
    year=file[23:27]
    doy =file[27:29]
    hr  =file[29:31]
    if hr not in ['15','16']:
        continue
    date=datetime(int(year),1,1)+timedelta(days=int(doy)-1)
    dstr=date.strftime('%Y%m%d')
    if dstr not in flist.keys():
        flist[dstr]={'GOES':[],'ARM':[],'AERI':[]}
    flist[dstr]['GOES'].append(g17dir+file)

for file in os.listdir(g18dir):
    hr=file[8:10]
    if hr not in ['15','16']:
        continue
    date=datetime(int(file[0:4]),int(file[4:6]),int(file[6:8]))
    dstr=date.strftime('%Y%m%d')
    if dstr not in flist.keys():
        flist[dstr]={'GOES':[],'ARM':[],'AERI':[]}
    flist[dstr]['GOES'].append(g18dir+file)

for file in os.listdir(parmdir):
    if 'C1' in file:
        dstr=file[24:32]
    else:
        dstr=file[25:33]
    if dstr not in flist.keys():
        flist[dstr]={'GOES':[],'ARM':[],'AERI':[]}
    flist[dstr]['ARM'].append(parmdir+file)

for file in os.listdir(narmdir):
    if 'C1' in file:
        dstr=file[24:32]
    else:
        dstr=file[25:33]
    if dstr not in flist.keys():
        flist[dstr]={'GOES':[],'ARM':[],'AERI':[]}
    flist[dstr]['ARM'].append(narmdir+file)

for file in os.listdir(tempdir):
    if 'C1' in file:
        dstr=file[20:28]
    else:
        dstr=file[21:29]
    if dstr not in flist.keys():
        flist[dstr]={'GOES':[],'ARM':[],'AERI':[]}
    flist[dstr]['AERI'].append(tempdir+file)

klist0=fnc['date'][:]
klist=[]
for k in klist0:
    dstr=str(int(k))
    if len(flist[dstr]['AERI'])<3:
        continue
    else:
        # check that there are at least three of the same sites reporting
        # remove non-colocated instruments
        sites=['C1','E32','E37','E39']
        llist=[]
        alist=[]
        lfiles=[]
        afiles=[]
        for l in flist[dstr]['ARM']:
            for site in sites:
                if site in l:
                    llist.append(site)
                    lfiles.append(l)
        for a in flist[dstr]['AERI']:
            for site in sites:
                if site in a:
                    alist.append(site)
                    afiles.append(a)
        sitecount=0
        lfiles2=[]
        afiles2=[]
        for i in range(len(llist)):
            if llist[i] in alist:
                lfiles2.append(lfiles[i])
                sitecount=sitecount+1
        for i in range(len(alist)):
            if alist[i] in llist:
                afiles2.append(afiles[i])
        flist[dstr]['AERI']=afiles2
        flist[dstr]['ARM']=lfiles2

        if sitecount<3:
            continue
        else:
            klist.append(dstr)

klist.sort()
print(len(klist))
if debug:
    print()
    print(klist)
    print()

################################################################
# Initialize Output
out={}

out['sites']=['E37','E32','E39','C1']
out['height']=fnc['height'][:]
out['hour']=[8,9,10,11,12,13,14,15,16,17,18]
dout=[]
for k in klist:
    dout.append(float(k))
out['date']=dout

if debug:
    print(dout)

Na=36
Nh=11
Nd=len(klist)
Ns=4

out['lon']=[]
out['lat']=[]
for site in out['sites']:
    out['lon'].append(lonlat[site][0])
    out['lat'].append(lonlat[site][1])

out['lon']=np.array(out['lon'][:])
out['lat']=np.array(out['lat'][:])

copylist=['lst_std','lst_mean','lst_lhet','lst_a','weighting','vort']

out['aeri_std']=np.ones((Nd,Nh,Na))*float('nan')
out['lst_std']=np.ones((Nd,))*float('nan')
out['lst_mean']=np.ones((Nd,))*float('nan')
out['lst_lhet']=np.ones((Nd,))*float('nan')
out['lst_a']=np.ones((Nd))*float('nan')
out['weighting']=np.zeros((Nd,Nh,Na))*float('nan')
out['vort']=np.ones((Nd))*float('nan')

out['DKE_xy']=np.ones((Nd,Nh,Na))*float('nan')
out['DKE_z']=np.ones((Nd,Nh,Na))*float('nan')
out['MKE_xy']=np.ones((Nd,Nh,Na))*float('nan')
out['MKE_z']=np.ones((Nd,Nh,Na))*float('nan')
out['wind_speed']=np.ones((Nd,Nh,Na))*float('nan')
out['wind_a']=np.ones((Nd,Nh,Na))*float('nan')
out['lst_std_site']=np.ones((Nd,))*float('nan')

out['sites_repo']=np.zeros((Nd,Nh,Na,Ns))

print('SETUP and PREPROCESSING COMPLETE')
print('TOTAL days: '+str(len(klist)))
print('Beginning processing...',flush=True)

#################################################################
di=0
for k in klist:
    print('::::'+k+':::::')
    #if debug:
    #    print(flist[k])

    fncdi=np.where(fnc['date'][:]==float(k))[0][0]
    for cop in copylist:
        out[cop][di]=fnc[cop][fncdi]

    std_site=[]
    date=datetime(int(k[0:4]),int(k[4:6]),int(k[6:8]),1)
    print('   LST processing...',end='',flush=True)

    # iterate through (one or two) LST files
    for i in range(len(flist[k]['GOES'])):
        fp=rasterio.open(flist[k]['GOES'][i])
        data=fp.read(1)

        if np.sum((data<1) | (data!=data)) > 0.1*data.size:
            continue

        tmp = np.copy(data)
        tmp[tmp != tmp] = np.mean(tmp[tmp == tmp])
        tmp[tmp == 0] = np.mean(tmp[tmp != 0])
        lst_mesoscale = scipy.ndimage.gaussian_filter(tmp, sigma=3, mode='reflect') # smoother

        # get surface temperature at each site
        tsites=[]
        for ll in lonlat.keys():
            for f in flist[k]['ARM']:
                if ll in f:
                    loc=fp.index(*lonlat[ll])
                    dd=lst_mesoscale[loc[1],loc[0]]
                    tsites.append(dd)
        std_site.append(np.nanstd(tsites,ddof=1))
        fp.close()
    print('COMPLETE',flush=True)

    # Pull together data from the two time periods
    out['lst_std_site'][di]=np.nanmean(std_site)

    date=datetime(int(k[0:4]),int(k[4:6]),int(k[6:8]),1)
    print('   LST processing...',end='',flush=True)

    fp=nc.Dataset(tempdir+'sgpaerioe1turnE32.c1.20190904.000558.nc','r')
    h_aeri=fp['height'][:]
    fp.close()

    # now iterate through the temperature stuff
    temp_temp=np.ones((Nh,Na,Ns))*float('nan')
    for i in range(len(flist[k]['AERI'])):
        fp = nc.Dataset(flist[k]['AERI'][i],'r')
        try:
            time=fp['time'][:]
        except Exception as e:
            print(flist[k]['AERI'][i])
            print(fp.variables.keys())
            raise e
        if 'height' in fp.variables.keys():
            h_=fp['height'][:]*1000
        else:
            h_=h_aeri*1000
        hidx=0
        for hr in out['hour']:
            t0=(hr+5)*60*60
            tf=(hr+6)*60*60
            m=(time>=t0)&(time<tf)
            if np.sum(m)<1:
                hidx=hidx+1
                continue
            t_=np.nanmean(fp['temperature'][m,:],axis=0)
            temp_temp[hidx,:,i]=np.interp(out['height'][:],h_,t_)
            hidx=hidx+1
    for i in range(Nh):
        out['aeri_std'][di,i,:]=np.nanstd(temp_temp[i,:,:],axis=1,ddof=1)

    # now iterate through the profile files
    u=[]
    v=[]
    w=[]
    ug=[]
    ua=[]

    print('   LIDAR processing',end='',flush=True)
    for i in range(len(flist[k]['ARM'])):
        print('.',end='',flush=True)
        fp = nc.Dataset(flist[k]['ARM'][i],'r')
        dates=nc.num2date(fp['time'][:],units=fp['time'].units)
        m = (dates >= datetime(date.year,date.month,date.day,13,0))
        if fp['u'].shape[1] != 164:continue
        if dates.size == 144:
            tmp = fp['u'][m,:].data
            tmpu = upscale(tmp,6)
            tmp = fp['v'][m,:].data
            tmpv = upscale(tmp,6)
            tmp = fp['w'][m,:].data
            tmpw = upscale(tmp,6)
            tmpa = upscale(fp['wind_direction'][m,:].data,6,True)
        if dates.size == 96:
            tmp = fp['u'][m,:].data
            tmpu = upscale(tmp,4)
            tmp = fp['v'][m,:].data
            tmpv = upscale(tmp,4)
            tmp = fp['w'][m,:].data
            tmpw = upscale(tmp,4)
            tmpa = upscale(fp['wind_direction'][m,:].data,4,True)
        z = fp['height'][:]
        if di==0:
            out['height']=z[0:36]
        else:
            if np.abs(out['height'][25]-z[25])>1:
                print(out['height'][25]-z[25])
                print('ERROR! Height Incorrect ',flush=True)

        fp.close()
        rho = calculate_air_density(z,0,1000)
        ug.append((tmpu**2+tmpv**2)**0.5) # horizontal wind speed
        u.append(tmpu)
        v.append(tmpv)
        w.append(tmpw)
        ua.append(tmpa)

        for h in range(11):
            for a in range(Na):
                if (np.sum(np.isnan(tmpu[h,a]))+np.sum(tmpu[h,a]<-100))==0:
                    s=0
                    for site in out['sites']:
                        if site in flist[k]['ARM'][i]:
                            out['sites_repo'][di,h,a,s]=1
                        s=s+1

    print('COMPLETE')
    # make into arrays (Sites,Hours,full_height)
    u = np.array(u)
    v = np.array(v)
    w = np.array(w)
    ug = np.array(ug)
    ua = np.array(ua)

    u=u[:,:,0:Na]
    v=v[:,:,0:Na]
    w=w[:,:,0:Na]
    ug=ug[:,:,0:Na]
    ua=ua[:,:,0:Na]

    u[u==-9999]=float('nan')
    v[v==-9999]=float('nan')
    w[w==-9999]=float('nan')

    out['wind_a'][di,:,:]=np.rad2deg(circmean(np.deg2rad(ua),axis=0,nan_policy='omit'))
    out['wind_speed'][di,:,:]=np.nanmean(ug,axis=0)

    up2=np.nanvar(u,axis=0,ddof=1)
    vp2=np.nanvar(v,axis=0,ddof=1)
    wp2=np.nanvar(w,axis=0,ddof=1)

    u2=np.nanmean(u,axis=0)**2
    v2=np.nanmean(v,axis=0)**2
    w2=np.nanmean(w,axis=0)**2

    weight=rho[0:Na]*(z[1:Na+1]-z[0:Na])

    out['DKE_xy'][di,:,:]=weight*0.5*(up2+vp2)
    out['DKE_z'][di,:,:] =weight*0.5*(wp2)
    out['MKE_xy'][di,:,:]=weight*0.5*(u2+v2)
    out['MKE_z'][di,:,:]=weight*0.5*(w2)
    out['weighting'][di,:,:]=weight[:]

    di=di+1
    print('COMPLETE')

fpout=nc.Dataset(troot+'lidar_tprof.nc','w')
fpout.createDimension('date', size=Nd)
fpout.createDimension('hour',size=Nh)
fpout.createDimension('height',size=Na)
fpout.createDimension('sites',size=Ns)
dims={Nd:'date',Nh:'hour',Na:'height',Ns:'sites'}

for v in out.keys():
    dimout=shape2dims(np.array(out[v]),dims)
    if v=='sites':
        fpout.sites=out[v][:]
    else:
        fpout.createVariable(v,'f8',dimensions=dimout)
        fpout[v][:]=np.array(out[v][:])
if debug:
    print()
    print(dout)
    print()
    print(out['date'])
    print()
    print(fpout['date'][:])

fpout.close()

