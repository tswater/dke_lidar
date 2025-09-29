import numpy as np
import glob
import numba
from datetime import datetime,timedelta
from scipy.stats import circmean
import os
import gstools as gs
#########################################################
############### FOLDERS ETC #############################
parmdir='/home/tswater/tyche/data/dke_peter/lidar_wind_profile/ARM/' # some of the ARM profiles
narmdir='/home/tswater/tyche/data/dke_peter/lidar_wind_profile/245518/' # the other ones
g17dir ='/home/tswater/tyche/data/dke_peter/GOES_holes/' # GOES data from 2017
g18dir ='/home/tswater/tyche/data/dke_peter/goes_lst_hourly/' # GOES data from other years
troot  ='/home/tswater/tyche/data/dke_peter/'

##########################################################
###################### FUNCTIONS #########################
def shape2dims(data,dims):
    dimout=()
    for d in data.shape:
        dimout=dimout+dims[d]
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

# remove days without one or the other
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

klist=list(flist.keys())
klist.sort()

################################################################
# Initialize Output
out={}

out['sites']=['E37','E41','E32','E39','C1']
out['height']=np.zeros((36,))
out['hour']=[8,9,10,11,12,13,14,15,16,17,18]
out['date']=klist

Na=36
Nh=11
Nd=len(klist)
Ns=5

out['lon']=[]
out['lat']=[]
for site in out['sites']:
    out['lon'].append(lonlat[site][0])
    out['lat'].append(lonlat[site][1])

out['lst_std']=np.ones((Nd,))*float('nan')
out['lst_mean']=np.ones((Nd,))*float('nan')
out['lst_std_site']=np.ones((Nd,))*float('nan')
out['lst_lhet']=np.ones((Nd,))*float('nan')
out['lst_a']=np.ones((Nd))*float('nan')
out['DKE_xy']=np.ones((Nd,Nh,Na))*float('nan')
out['DKE_z']=np.ones((Nd,Nh,Na))*float('nan')
out['MKE_xy']=np.ones((Nd,Nh,Na))*float('nan')
out['MKE_z']=np.ones((Nd,Nh,Na))*float('nan')
out['wind_speed']=np.ones((Nd,Nh,Na))*float('nan')
out['wind_a']=np.ones((Nd,Nh,Na))*float('nan')
out['sites_repo']=np.zeros((Nd,Nh,Ns))
out['weighting']=np.zeros((Nd,Nh,Na))*float('nan')
out['vort']=np.ones((Nd))*float('nan')


#################################################################
di=0
for k in klist:
    print('::::'+k+':::::')
    std=[]
    lhet=[]
    tmean=[]
    std_site=[]
    xgrad=[]
    ygrad=[]
    date=datetime(int(k[0:4]),int(k[4:6]),int(k[6:8]),1)

    # iterate through (one or two) LST files
    for i in range(len(flist[k]['GOES'])):
        fp=rasterio.open(flist[k]['GOES'][i])
        data=fp.read(1)

        # clear sky check basically
        if np.sum((data==0) | (data!=data)) > 0.1*data.size:
            std.append(float('nan'))
            lhet.append(float('nan'))
            tmean.append(float('nan'))
            xgrad.append(float('nan'))
            ygrad.append(float('nan'))
            std_site.append(float('nan'))
        else:
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

            # get surface temperature at each site
            tsites=[]
            for ll in lonlat.keys():
                for f in flist[k]['ARM']:
                    if ll in f:
                        if ll not in tmean_site.keys():
                            tmean_site[ll]=[]
                        loc=fp.index(*lonlat[ll])
                        dd=lst_mesoscale[loc[1],loc[0]]
                        tsites.append(dd)
            std_site.append(np.nanstd(tsites))
        fp.close()

    # Pull together data from the two time periods
    out['lst_lhet'][di]=np.nanmean(lhet)
    out['lst_std'][di]=np.nanmean(std)
    out['lst_mean'][di]=np.nanmean(tmean)
    out['lst_std_site'][di]=np.nanmean(std_site)
    out['lst_a'][di]=angle_between([np.nanmean(xgrad),np.nanmean(ygrad)],[1,0])

    # now iterate through the profile files
    u=[]
    v=[]
    w=[]
    ug=[]
    ua=[]

    for i in range(len(flist[k]['ARM'])):
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
                print('ERROR! Height Incorrect ',flush=True)

        fp.close()
        rho = calculate_air_density(z,0,1000)
        ug.append((tmpu**2+tmpv**2)**0.5) # horizontal wind speed
        u.append(tmpu)
        v.append(tmpv)
        w.append(tmpw)
        ua.append(tmpa)

        for h in range(11):
            if np.sum(np.isnan(tmpu[h])|(tempu[h]<-100))>0:
                pass
            else:
                s=0
                for site in out['sites']:
                    if site in flist[k]['ARM'][i]:
                        out['sites_repo'][di,h,s]=1
                    s=s+1

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
    v[u==-9999]=float('nan')
    w[u==-9999]=float('nan')

    # check if number of reporting sites is the same as recorded previously
    a=np.sum(np.isnan(u)|np.isnan(v)|np.isnan(w),axis=0)
    b=np.sum(out['sites_repo'][di,:,:],axis=1)
    numdiff=np.sum(a[:,0]!=b[:])
    if numdiff>0:
        print('ERROR: '+str(numdiff))

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

    vabs=np.abs(np.array(vort))
    out['vort'][di]=np.nanmax(vabs)

    di=di+1

fpout=nc.Dataset(troot+'lidar_lst_out.nc','w')
fpout.createDimension('date', size=Nd)
fpout.createDimension('hour',size=Nh)
fpout.createDimension('height',size=Na)
fpout.createDimension('sites',size=Ns)
dims={Nd:'date',Nh:'hour',Na:'height',Ns:'sites'}

for v in out.keys():
    fpout.createVariable(v,'f',dimensions=shape2dim(out[v],dims))
    fpout[v][:]=out[v][:]

fpout.close()

