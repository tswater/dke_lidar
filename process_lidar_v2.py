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
from scipy.ndimage import rotate
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
outfile=troot+'lidar_lst_out2.nc'
debug = False
newfile = True # if a new output file should be created
skip_herbie = False

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
# Generate a list of available days and a dictionary of files
flist={} # a dictionary of day:{'ARM':[list_of_profile_file_paths],'GOES':[list_of_goes_file_paths]} pairs

for file in os.listdir(g17dir):
    year=file[23:27]
    doy =file[27:29]
    hr  =file[29:31]
    if hr not in ['15','16']:
        continue
    date=datetime(int(year),1,1)+timedelta(days=int(doy)-1)
    dstr=date.strftime('%Y%m%d')
    if dstr not in flist.keys():
        flist[dstr]={'GOES':[],'ARM':[]}
    flist[dstr]['GOES'].append(g17dir+file)

for file in os.listdir(g18dir):
    hr=file[8:10]
    if hr not in ['15','16']:
        continue
    date=datetime(int(file[0:4]),int(file[4:6]),int(file[6:8]))
    dstr=date.strftime('%Y%m%d')
    if dstr not in flist.keys():
        flist[dstr]={'GOES':[],'ARM':[]}
    flist[dstr]['GOES'].append(g18dir+file)

for file in os.listdir(parmdir):
    if 'C1' in file:
        dstr=file[24:32]
    else:
        dstr=file[25:33]
    if dstr not in flist.keys():
        flist[dstr]={'GOES':[],'ARM':[]}
    flist[dstr]['ARM'].append(parmdir+file)

for file in os.listdir(narmdir):
    if 'C1' in file:
        dstr=file[24:32]
    else:
        dstr=file[25:33]
    if dstr not in flist.keys():
        flist[dstr]={'GOES':[],'ARM':[]}
    flist[dstr]['ARM'].append(narmdir+file)

# remove days without GOES or without ARM or error days from nate
klist=list(flist.keys())
for k in klist:
    if len(flist[k]['ARM'])<2:
        del flist[k]
    elif len(flist[k]['GOES'])<1:
        del flist[k]
    elif int(k[4:6])<5:
        del flist[k]
    elif int(k[4:6])>9:
        del flist[k]
    elif k=='20210730':
        del flist[k]
    elif k=='20210828':
        del flist[k]
    elif k=='20220617':
        del flist[k]
    elif k=='20220618':
        del flist[k]
    elif k=='20220625':
        del flist[k]
    elif k=='20220808':
        del flist[k]
    elif k=='20220809':
        del flist[k]
    elif k=='20220927':
        del flist[k]
    elif k=='20220901':
        del flist[k]

klist0=list(flist.keys())
klist0.sort()

# Clearsky check
di=0
klist=[]

for k in klist0:
    # iterate through (one or two) LST files
    isbad=True
    for i in range(len(flist[k]['GOES'])):
        fp=rasterio.open(flist[k]['GOES'][i])
        data=fp.read(1)

        # clear sky check basically
        if np.sum((data<1) | (data!=data)) > 0.1*data.size:
            pass
        else:
            isbad=False
        if debug:
            if k in ['20180704','20200916']:
                print(flist[k])
                print(isbad)
    if isbad:
        pass
    else:
        klist.append(k)

klist.sort()
print(len(klist))
if debug:
    print()
    print(klist)
    print()

################################################################
# Initialize Output
out={}

out['sites']=['E37','E41','E32','E39','C1']
out['height']=np.zeros((36,))
out['hour']=[8,9,10,11,12,13,14,15,16,17,18]
dout=[]
for k in klist:
    dout.append(float(k))
out['date']=dout

if debug:
    print(dout)

Na=36 # height (up to 1000km)
Nh=11 # hours
Nd=len(klist) # number of days
Ns=5 # number of sites

out['lon']=[]
out['lat']=[]
for site in out['sites']:
    out['lon'].append(lonlat[site][0])
    out['lat'].append(lonlat[site][1])

out['lon']=np.array(out['lon'][:])
out['lat']=np.array(out['lat'][:])

out['lst_std']=np.ones((Nd,))*float('nan')                # standard deviation LST
out['lst_std_parallel']=np.ones((Nd,))*float('nan')       # average standard deviation of LST parallel to mean wind
out['lst_std_perpendicular']=np.ones((Nd,))*float('nan')  # average standard deviation of LST perpendicular to mean wind
out['lst_mean']=np.ones((Nd,))*float('nan')               # mean LST
out['lst_std_site']=np.ones((Nd,))*float('nan')           # standard deviation of LST between the sites
out['lst_lhet']=np.ones((Nd,))*float('nan')               # lengthscale of heterogeneity lambda
out['lst_a']=np.ones((Nd))*float('nan')                   # direction of surface gradient
out['lst_a2']=np.ones((Nd))*float('nan')                  # direction of surface maximum variability (not recommended to use)
out['DKE_xy']=np.ones((Nd,Nh,Na))*float('nan')            # DKE (xy only; weighted by rho*dz)
out['DKE_z']=np.ones((Nd,Nh,Na))*float('nan')             # DKE (z only; weighted by rho*dz)
out['MKE_xy']=np.ones((Nd,Nh,Na))*float('nan')            # MKE (xy only; weighted by rho*dz)
out['MKE_z']=np.ones((Nd,Nh,Na))*float('nan')             # MKE (z only; weighted by rho*dz)
out['wind_speed']=np.ones((Nd,Nh,Na))*float('nan')        # wind speed lidar
out['wind_a']=np.ones((Nd,Nh,Na))*float('nan')            # wind direction lidar
out['wind_a_diff']=np.ones((Nd,Nh))*float('nan')          # difference between met winddir and lidar winddir (for qaqc)
out['wind_speed_diff']=np.ones((Nd,Nh))*float('nan')      # difference between met windspeed and lidar windspeed (for qaqc)
out['sites_repo']=np.zeros((Nd,Nh,Na,Ns))                 # sites reporting valid values
out['weighting']=np.zeros((Nd,Nh,Na))*float('nan')        # weighting
if not skip_herbie:
    out['vort']=np.ones((Nd))*float('nan')                # vorticity (HRRR)
    out['pgrad']=np.ones((Nd))*float('nan')               # magnitude of mean pressure gradient (HRRR)
out['precip']=np.ones((Nd,Nh))*float('nan')               # precipitation from MET data
print('SETUP and PREPROCESSING COMPLETE')
print('TOTAL days: '+str(len(klist)))
print('Beginning processing...',flush=True)

#################################################################
di=0
for k in klist:
    print('::::'+k+':::::')
    date=datetime(int(k[0:4]),int(k[4:6]),int(k[6:8]),1)

    # now iterate through the profile files
    u=[]
    v=[]
    w=[]
    ug=[]
    ua=[]
    precip=[]
    ma=[]
    mws=[]

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
            tmpws= upscale(fp['wind_speed'][m,:].data,6)
            tmpa = upscale(fp['wind_direction'][m,:].data,6,True)
            md   =fp['met_wdir'][m].data
            tmpma= upscale(md.reshape(len(md),1),6,True)
            md   =fp['met_wspd'][m].data
            tmpms= upscale(md.reshape(len(md),1),6)
            pp   = fp['met_spr'][m].data
            pp   = pp.reshape(len(pp),1)
            tmpp = upscale(pp,6)
        if dates.size == 96:
            tmp = fp['u'][m,:].data
            tmpu = upscale(tmp,4)
            tmp = fp['v'][m,:].data
            tmpv = upscale(tmp,4)
            tmp = fp['w'][m,:].data
            tmpw = upscale(tmp,4)
            tmpws= upscale(fp['wind_speed'][m,:].data,4)
            tmpa = upscale(fp['wind_direction'][m,:].data,4,True)
            md   =fp['met_wdir'][m].data
            tmpma= upscale(md.reshape(len(md),1),4,True)
            md   =fp['met_wspd'][m].data
            tmpms= upscale(md.reshape(len(md),1),4)
            pp   = fp['met_spr'][m].data
            pp   = pp.reshape(len(pp),1)
            tmpp = upscale(pp,4)
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
        precip.append(tmpp)
        ma.append(tmpma[:,0]-tmpa[:,9])
        mws.append(tmpms[:,0]-tmpws[:,9])

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
    precip= np.array(precip)
    ma = np.array(ma)
    mws= np.array(mws)

    u=u[:,:,0:Na]
    v=v[:,:,0:Na]
    w=w[:,:,0:Na]
    ug=ug[:,:,0:Na]
    ua=ua[:,:,0:Na]

    u[u==-9999]=float('nan')
    v[v==-9999]=float('nan')
    w[w==-9999]=float('nan')
    precip[precip==-9999]=float('nan')

    out['wind_speed_diff'][di,:]=np.nanmax(np.abs(mws),axis=0)
    out['wind_a_diff'][di,:]=np.nanmax(np.abs(ma),axis=0)
    out['wind_a'][di,:,:]=np.rad2deg(circmean(np.deg2rad(ua),axis=0,nan_policy='omit'))
    out['wind_speed'][di,:,:]=np.nanmean(ug,axis=0)
    out['precip'][di,:]=np.nanmean(precip,axis=(0,1))

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

    ########### LST PROCESSING ###############
    std=[]
    std_ll=[]
    std_perp=[]
    lhet=[]
    tmean=[]
    std_site=[]
    xgrad=[]
    ygrad=[]
    xgrad2=[]
    ygrad2=[]
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
        meso_lh = calculate_length_scale(lst_mesoscale,undef=0)

        std.append(np.nanstd(lst_mesoscale))
        lhet.append(meso_lh)
        tmean.append(np.nanmean(lst_mesoscale))
        grad=np.gradient(lst_mesoscale)
        xgrad.append(np.nanmean(grad[0]))
        ygrad.append(np.nanmean(grad[1]))
        xgrad2.append(np.nanmean(np.abs(grad[0])))
        ygrad2.append(np.nanmean(np.abs(grad[1])))

        # now handle the parallel and perpendicular std
        winda=np.rad2deg(circmean(np.deg2rad(out['wind_a'][di,2:8,0:5]),nan_policy='omit'))
        lst_rotated=rotate(lst_mesoscale,winda,cval=float('nan'))
        stds_0=np.nanstd(lst_rotated,axis=1)
        cnts_0=np.nansum(~np.isnan(lst_rotated),axis=1)
        stds_0=stds_0*cnts_0/np.nansum(cnts_0)
        std_perp.append(np.nansum(stds_0))

        stds_0=np.nanstd(lst_rotated,axis=0)
        cnts_0=np.nansum(~np.isnan(lst_rotated),axis=0)
        stds_0=stds_0*cnts_0/np.nansum(cnts_0)
        std_ll.append(np.nansum(stds_0))

        # get surface temperature at each site
        tsites=[]
        for ll in lonlat.keys():
            for f in flist[k]['ARM']:
                if ll in f:
                    loc=fp.index(*lonlat[ll])
                    dd=lst_mesoscale[loc[1],loc[0]]
                    tsites.append(dd)
        std_site.append(np.nanstd(tsites))
        fp.close()
    print('COMPLETE',flush=True)

    # Pull together data from the two time periods
    out['lst_lhet'][di]=np.nanmean(lhet)
    out['lst_std'][di]=np.nanmean(std)
    out['lst_mean'][di]=np.nanmean(tmean)
    out['lst_std_site'][di]=np.nanmean(std_site)
    out['lst_std_parallel'][di]=np.nanmean(std_ll)
    out['lst_std_perpendicular'][di]=np.nanmean(std_perp)
    out['lst_a'][di]=angle_between([-np.nanmean(xgrad),np.nanmean(ygrad)],[1,0])
    out['lst_a2'][di]=angle_between([np.nanmean(xgrad2),np.nanmean(ygrad2)],[1,0])

    if not skip_herbie:
        print('   HERBIE processing...',end='',flush=True)
        # Now herbie vorticity stuff
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
            ds2= H.xarray(':PRES',remove_grib=False)
            pres  = ds2[4]['sp'].values[591:630,887:916]
            grad  = np.gradient(pres)
            gx    = np.abs(np.nanmean(grad[0]))
            gy    = np.abs(np.nanmean(grad[1]))
            gout  = max(gx,gy)

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
        except Exception:
            vort = 0
            gout = -.0001

        vabs=np.abs(np.array(vort))
        out['vort'][di]=np.nanmax(vabs)
        out['pgrad'][di]=gout

    di=di+1
    print('COMPLETE')

if newfile:
    fpout=nc.Dataset(outfile,'w')
    fpout.createDimension('date', size=Nd)
    fpout.createDimension('hour',size=Nh)
    fpout.createDimension('height',size=Na)
    fpout.createDimension('sites',size=Ns)
else:
    fpout=nc.Dataset(outfile,'r+')
dims={Nd:'date',Nh:'hour',Na:'height',Ns:'sites'}

for v in out.keys():
    dimout=shape2dims(np.array(out[v]),dims)
    if v=='sites':
        try:
            fpout.sites=out[v][:]
        except:
            pass
    else:
        try:
            fpout.createVariable(v,'f8',dimensions=dimout)
        except:
            pass
        fpout[v][:]=np.array(out[v][:])
if debug:
    print()
    print(dout)
    print()
    print(out['date'])
    print()
    print(fpout['date'][:])

fpout.close()

