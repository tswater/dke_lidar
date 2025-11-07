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
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc
import pickle
import numpy as np
import rasterio
import os
from scipy import stats
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import circmean,spearmanr
from sklearn.metrics import mean_squared_error
from matplotlib.gridspec import GridSpec
matplotlib.rcParams['figure.dpi'] = 500
plt.style.use('seaborn-v0_8-deep')
plt.rcParams.update({'figure.max_open_warning': 0})

# %%
troot='/home/tswater/tyche/data/dke_peter/'
proot='/home/tswater/projects/dke_lidar/'
g18dir ='/home/tswater/tyche/data/dke_peter/goes_lst_hourly/'
parmdir='/home/tswater/tyche/data/dke_peter/lidar_wind_profile/ARM/' # some of the ARM profiles
narmdir='/home/tswater/tyche/data/dke_peter/lidar_wind_profile/245518/' # the other ones
fnc=nc.Dataset(troot+'lidar_lst_out2.nc','r')
fnt=nc.Dataset(troot+'lidar_tprof.nc','r')
fles=pickle.load(open(proot+'pickle_tsw/dke_mke_tsw.pkl','rb'))
fles_corrs=pickle.load(open(proot+'pickle_tsw/corrs.pkl','rb'))
fles_hets=pickle.load(open(proot+'pickle_tsw/hets.pkl','rb'))

# %%

# %%
lonlat={'E37':[-97.927376, 36.3109],
        'E41':[-97.08639, 36.879944],
        'E32':[-97.81987, 36.819656],
        'E39':[-97.06912, 36.373775],
        'C1':[-97.48658, 36.605293]}

# %%
len('diag_d01_2016-06-26_02')

# %%
fnc['weighting'][5,5,5]

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # LES Example Profiles + DKE

# %% [markdown]
# #### Data Prep

# %%
fp=nc.Dataset(bdir+'fr2_'+day+'_00/diag_d01_2016-06-25_130000','r')

# %%

# %%
fp['AVV_Z'][:].shape
a=np.nanmean(fp['AVV_Z'][:],axis=(0,2,3))
np.where(a<1000)


# %%
def get_les_dke(fp):
    u=fp['AVV_U'][:]
    v=fp['AVV_V'][:]
    w=fp['AVV_W'][:]
    dke=np.zeros((226,))
    mke=np.zeros((226,))
    for k in range(226):
        mke[k]=(np.nanmean(u[0,k,:,:])**2+np.nanmean(v[0,k,:,:])**2+np.nanmean(w[0,k,:,:])**2)*0.5
        u[0,k,:,:]=u[0,k,:,:]-np.nanmean(u[0,k,:,:])
        v[0,k,:,:]=v[0,k,:,:]-np.nanmean(v[0,k,:,:])
        w[0,k,:,:]=w[0,k,:,:]-np.nanmean(w[0,k,:,:])
        dke[k]=0.5*(np.nanvar(u[0,k,:,:])+np.nanvar(v[0,k,:,:])+np.nanvar(w[0,k,:,:]))
    return dke,mke


# %%

# %%
wdir='/home/tswater/tyche/data/les/LES_wind/'
bdir='/run/media/tswater/Elements/LES/'
day='20160625'
hmg_td=np.zeros((15,100))
het_td=np.zeros((15,100))

hmg_tdm=np.zeros((15,100))
het_tdm=np.zeros((15,100))

hmg_su=np.ones((110,520))*float('nan')
het_su=np.ones((110,520))*float('nan')
hmg_tsf=np.ones((110,520))*float('nan')
het_tsf=np.ones((110,520))*float('nan')
hgdir='fr2_'+day+'_01/'
htdir='fr2_'+day+'_00/'

daylist=os.listdir(bdir+hgdir)
daylist.sort()
i=0
for file in daylist:
    if i==0:
        i=i+1
        continue
    fpg=nc.Dataset(wdir+hgdir+file[0:22]+'_wind.nc','r')
    fpt=nc.Dataset(wdir+htdir+file[0:22]+'_wind.nc','r')
    dke,mke=get_les_dke(fpg)
    hmg_td[i-1,:]=dke[:100]
    hmg_tdm[i-1,:]=dke[:100]/mke[:100]
    dke,mke=get_les_dke(fpt)
    het_td[i-1,:]=dke[:100]
    het_tdm[i-1,:]=dke[:100]/mke[:100]
    i=i+1

# %%
import scipy

hr=19
hr_='diag_d01_2016-06-25_'+str(hr)+'0000'
hr2='diag_d01_2016-06-25_'+str(hr)+'_wind.nc'

fpu=nc.Dataset(wdir+htdir+hr2,'r')
fpt=nc.Dataset(bdir+htdir+hr_,'r')
het_su[10:,:]=fpu['AVV_V'][0,0:100,:,350]
het_tsf[0,:]=fpt['AVS_TSK'][0,:,350]
het_norm=np.nanmean(fpu['AVV_V'][0,0:40,:,:])

fpu=nc.Dataset(wdir+hgdir+hr2,'r')
fpt=nc.Dataset(bdir+hgdir+hr_,'r')
hmg_su[10:,:]=fpu['AVV_V'][0,0:100,:,350]
hmg_tsf[0,:]=fpt['AVS_TSK'][0,:,350]
hmg_norm=np.nanmean(fpu['AVV_V'][0,0:40,:,:])

for i in range(10):
    het_tsf[i,:]=het_tsf[0,:]
    hmg_tsf[i,:]=hmg_tsf[0,:]

hmg_su[10:]=scipy.ndimage.gaussian_filter(hmg_su[10:], sigma=2, mode='reflect') # smoother
het_su[10:]=scipy.ndimage.gaussian_filter(het_su[10:], sigma=2, mode='reflect')
hmg_su[:]=hmg_su[:]-hmg_norm
het_su[:]=het_su[:]-hmg_norm

# %%
fpg.close()
fpt.close()

# %% [markdown]
# #### Plotting

# %%
fpt['AVP_Z'][0,0:33]

# %%
#plt.plot(np.linspace(0,32,33),fpt['AVP_Z'][0,0:33])
#plt.yticks([0,100,200,300,400,500,600,700,800,900,1000])
#plt.xticks(np.linspace(0,32,33))
#plt.grid(True)

# %%

# %%
fig,axs=plt.subplots(2,2,figsize=(5,3.5),width_ratios=[1,1.25])
cmapd='nipy_spectral'
cmapu='PuOr'
cmapt='coolwarm'
fntn=8
fnts=6
version=['dke','dke_mke'][1]

if version=='dke':
    dmin=0
    dmax=8
    dlabel=r'$DKE$ ($m^2s^{-2}$)'
    het_d=het_td
    hmg_d=hmg_td
elif version=='dke_mke':
    dmin=0
    dmax=.25
    dlabel=r'$DKE/MKE$'
    het_d=het_tdm
    hmg_d=hmg_tdm
    

axs[0,1].imshow(het_d.T[0:33,0:11],vmin=dmin,vmax=dmax,origin='lower',aspect=.2,interpolation='spline16',cmap=cmapd)
axs[0,0].imshow(hmg_d.T[0:33,0:11],vmin=dmin,vmax=dmax,origin='lower',aspect=.2,interpolation='spline16',cmap=cmapd)

aspt=7.8

axs[1,0].imshow(hmg_su[5:43,20:500],origin='lower',cmap=cmapu,aspect=aspt,vmin=-4,vmax=4)
axs[1,0].imshow(hmg_tsf[5:43,20:500],origin='lower',cmap=cmapt,aspect=aspt,vmin=304,vmax=316)
axs[1,1].imshow(het_su[5:43,20:500],origin='lower',cmap=cmapu,aspect=aspt,vmin=-4,vmax=4)
axs[1,1].imshow(het_tsf[5:43,20:500],origin='lower',cmap=cmapt,aspect=aspt,vmin=304,vmax=316)

xticks=np.linspace(0,10,11)
xticks_l=['','9:00','','','12:00','','','15:00','','','18:00']
yticks=np.array([-.6,2.6,5.8,9,12.2,15.4,18.6,21.8,25,28.2,31.4])
yticks_l=[0,'',200,'',400,'',600,'',800,'',1000]
xticks2=[0,50,100,150,200,250,300,350,400,450]
xticks2_l=[0,10,20,30,40,50,60,70,80,90]

ax=axs[0,0]
ax.set_xticks(xticks,xticks_l,fontsize=fnts)
ax.set_yticks(yticks,yticks_l,fontsize=fnts)
ax.set_xlabel('Local Hour',fontsize=fntn,labelpad=2)
ax.set_ylabel('Height (m)',fontsize=fntn)
ax.set_title('Homogeneous',fontsize=10)

ax=axs[0,1]
ax.set_xticks(xticks,xticks_l,fontsize=fnts)
ax.set_yticks(yticks,[],fontsize=fnts)
ax.set_xlabel('Local Hour',fontsize=fntn,labelpad=2)
ax.set_title('Heterogeneous',fontsize=10)

yticks=yticks+5
ax=axs[1,0]
ax.set_yticks(yticks,yticks_l,fontsize=fnts)
ax.set_xticks(xticks2,xticks2_l,fontsize=fnts)
ax.set_xlabel('Distance (km)',fontsize=fntn,labelpad=2)
ax.set_ylabel('Height (m)',fontsize=fntn)

ax=axs[1,1]
ax.set_yticks(yticks,[],fontsize=fnts)
ax.set_xticks(xticks2,xticks2_l,fontsize=fnts)
ax.set_xlabel('Distance (km)',fontsize=fntn,labelpad=2)

for i in range(2):
    for j in range(2):
        ax=axs[i,j]
        ax.grid(True,alpha=1,color='black',linewidth=.1)

plt.subplots_adjust(wspace=.05)

cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(dmin, dmax), cmap=cmapd),
             ax=axs[0,1], orientation='vertical',shrink=.84)
cb.ax.tick_params(labelsize=fnts)
cb.set_label(label=dlabel,size=fntn)

cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-4, 4), cmap=cmapu),
             ax=axs[1,1], orientation='vertical',shrink=.84)
cb.ax.tick_params(labelsize=fnts)
cb.set_label(label=r'$u$ ($ms^{-1}$)',size=fntn)

plt.savefig('../../plot_output/dke1/les_prof_'+version+'.png', bbox_inches = "tight")

# %%
fig,axs=plt.subplots(3,2,figsize=(5,5),width_ratios=[1,1.25])
cmapd='nipy_spectral'
cmapu='PuOr'
cmapt='coolwarm'
fntn=8
fnts=6

for i in range(2):
    version=['dke','dke_mke'][i]
    
    if version=='dke':
        dmin=0
        dmax=8
        dlabel=r'$DKE$ ($m^2s^{-2}$)'
        het_d=het_td
        hmg_d=hmg_td
    elif version=='dke_mke':
        dmin=0
        dmax=.25
        dlabel=r'$DKE/MKE$'
        het_d=het_tdm
        hmg_d=hmg_tdm
        
    
    axs[i,1].imshow(het_d.T[0:33,0:11],vmin=dmin,vmax=dmax,origin='lower',aspect=.2,interpolation='spline16',cmap=cmapd)
    axs[i,0].imshow(hmg_d.T[0:33,0:11],vmin=dmin,vmax=dmax,origin='lower',aspect=.2,interpolation='spline16',cmap=cmapd)
    
    
    xticks=np.linspace(0,10,11)
    xticks_l=['','9:00','','','12:00','','','15:00','','','18:00']
    yticks=np.array([-.6,2.6,5.8,9,12.2,15.4,18.6,21.8,25,28.2,31.4])
    yticks_l=[0,'',200,'',400,'',600,'',800,'',1000]
    xticks2=[0,50,100,150,200,250,300,350,400,450]
    xticks2_l=[0,10,20,30,40,50,60,70,80,90]
    
    ax=axs[i,0]
    ax.set_xticks(xticks,xticks_l,fontsize=fnts)
    ax.set_yticks(yticks,yticks_l,fontsize=fnts)
    ax.set_xlabel('Local Hour',fontsize=fntn,labelpad=2)
    ax.set_ylabel('Height (m)',fontsize=fntn)
    if i==0:
        ax.set_title('Homogeneous',fontsize=10)
    
    ax=axs[i,1]
    ax.set_xticks(xticks,xticks_l,fontsize=fnts)
    ax.set_yticks(yticks,[],fontsize=fnts)
    ax.set_xlabel('Local Hour',fontsize=fntn,labelpad=2)
    if i==0:
        ax.set_title('Heterogeneous',fontsize=10)

    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(dmin, dmax), cmap=cmapd),
             ax=axs[i,1], orientation='vertical',shrink=.9)
    cb.ax.tick_params(labelsize=fnts)
    cb.set_label(label=dlabel,size=fntn)

aspt=7.8

axs[2,0].imshow(hmg_su[5:43,20:500],origin='lower',cmap=cmapu,aspect=aspt,vmin=-4,vmax=4)
axs[2,0].imshow(hmg_tsf[5:43,20:500],origin='lower',cmap=cmapt,aspect=aspt,vmin=304,vmax=316)
axs[2,1].imshow(het_su[5:43,20:500],origin='lower',cmap=cmapu,aspect=aspt,vmin=-4,vmax=4)
axs[2,1].imshow(het_tsf[5:43,20:500],origin='lower',cmap=cmapt,aspect=aspt,vmin=304,vmax=316)

yticks=yticks+5
ax=axs[2,0]
ax.set_yticks(yticks,yticks_l,fontsize=fnts)
ax.set_xticks(xticks2,xticks2_l,fontsize=fnts)
ax.set_xlabel('Distance (km)',fontsize=fntn,labelpad=2)
ax.set_ylabel('Height (m)',fontsize=fntn)

ax=axs[2,1]
ax.set_yticks(yticks,[],fontsize=fnts)
ax.set_xticks(xticks2,xticks2_l,fontsize=fnts)
ax.set_xlabel('Distance (km)',fontsize=fntn,labelpad=2)

for i in range(2):
    for j in range(2):
        ax=axs[i,j]
        ax.grid(True,alpha=1,color='black',linewidth=.1)

plt.subplots_adjust(wspace=.05)

cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-4, 4), cmap=cmapu),
             ax=axs[2,1], orientation='vertical',shrink=.9)
cb.ax.tick_params(labelsize=fnts)
cb.set_label(label=r'$u$ ($ms^{-1}$)',size=fntn)

plt.savefig('../../plot_output/dke1/les_prof_cmb2.png', bbox_inches = "tight")

# %%

# %% [markdown]
# # LIDAR Example Profiles + Surface

# %%

# %%

# %%

# %% [markdown]
# #### Data Prep

# %%
dd=fnc['date'][:]
days=[]
i0=69
for i in range(i0,i0+3):
    days.append(str(int(dd[i])))

days=['20180531','20180628','20180629']
days=['20180521','20180719','20180629']

didx=[]
numvalid=[0,0,0]
profs=[{},{},{}]
dkep=[]
scatloc=[{},{},{}]
colors={'E37':'darkgoldenrod','E41':'teal','E32':'firebrick','E39':'forestgreen','C1':'darkorchid'}
goes=[]
lhet=[]
std=[]
wdir=[]
wdir2=[{},{},{}]

for d in days:
    didx.append(np.where(float(d)==fnc['date'][:].data)[0][0])
i=0
for day in days:
    if i==0:
        fgoes=rasterio.open(g18dir+day+'1600.tif','r')
    else:
        fgoes=rasterio.open(g18dir+day+'1500.tif','r')
    dd=fgoes.read(1)
    dd[dd<1]=float('nan')
    goes.append(dd)
    lhet.append(fnc['lst_lhet'][didx[i]])
    std.append(fnc['lst_std'][didx[i]])
    for file in os.listdir(narmdir):
        if day in file:
            numvalid[i]=numvalid[i]+1
            for s in lonlat.keys():
                if s in file:
                    kk=s
                    #print(kk)
            fprof=nc.Dataset(narmdir+file,'r')
            aa=fprof['u'].shape[0]
            if aa==96:
                dx=4
            elif aa==144:
                dx=6
            elif aa>120:
                dx=6
            u=fprof['u'][17*dx:18*dx,0:36].data
            u[u==-9999]=float('nan')
            u=np.nanmean(u,axis=0)
            v=fprof['v'][17*dx:18*dx,0:36].data
            v[v==-9999]=float('nan')
            v=np.nanmean(v,axis=0)
            z=fprof['height'][0:36]
            ug=np.sqrt(u**2+v**2)
            profs[i][kk]=ug
            scatloc[i][kk]=fgoes.index(*lonlat[kk])
    wdir.append(circmean(np.deg2rad(fnc['wind_a'][didx[i],4,0:15])))
    dkep.append(fnc['DKE_xy'][didx[i],4,0:35]/fnc['weighting'][didx[i],4,0:35])        
    i=i+1

# %%

# %% [markdown]
# #### Plotting

# %%
fig=plt.figure(figsize=(5.5,6))
subfigs = fig.subfigures(2, 1, hspace=0,wspace=0,frameon=False,height_ratios=[2,1.2])
ax=subfigs[0].subplots(1,3)
grid=ImageGrid(subfigs[1], 111,  # similar to subplot(111)
                nrows_ncols=(1, 3),
                axes_pad=0.1,
                cbar_mode='each',
                cbar_location='bottom',
                cbar_pad=.02,
                cbar_size="5%")
#vmin=[294,
#vrange=12.5
vmin=296
vmax=310
for i in range(3):
    for k in range(5):
        s=list(colors.keys())[k]
        ax[i].plot(profs[i][s],z,'-o',color=colors[s],markersize=2,linewidth=.5,label=s)
    if i==1:
        l=ax[i].legend(fontsize=8,loc='upper right',framealpha=.95)
        #l.set_labelsize(8)
    im=grid[i].imshow(goes[i],cmap='coolwarm',vmin=vmin,vmax=vmax)
    #im=grid[i].imshow(goes[i],cmap='coolwarm')
    xx=[]
    yy=[]
    clrs=[]
    for k in scatloc[0].keys():
        xx.append(scatloc[i][k][0])
        yy.append(scatloc[i][k][1])
        clrs.append(matplotlib.colors.to_rgb(colors[k]))
    xx=np.array(xx)
    yy=np.array(yy)
    grid[i].scatter(yy,xx,s=15,c=clrs,edgecolors='black',linewidth=.5,alpha=.95)

    if i==0:
        ax[i].set_yticks([200,400,600,800,1000],[200,400,600,800,1000])
        ax[i].set_ylabel(r'height $(m)$')
    else:
        ax[i].set_yticks([200,400,600,800,1000],[])
    ax[i].tick_params(axis='both',which='major',labelsize=8)
    ax[i].set_xlabel(r'$\overline{u}$  ($m\ s^{-1}$)')
    #ax[i].grid(True)

    grid[i].set_xticks([])
    grid[i].set_yticks([])

    ax[i].set_title(days[i][0:4]+'-'+days[i][4:6]+'-'+days[i][6:8],fontsize=10)

    tit=r'$\ell_{het}$:'+' '+str(lhet[i])[0:5]+'   '+r'$\sigma_{LST}$:'+' '+str(std[i])[0:3]
    grid[i].set_title(tit,fontsize=8)
    ax[i].set_ylim(0,1025)
    cb=grid.cbar_axes[i].colorbar(im,label=r'$T_{surf}$')
    grid.cbar_axes[i].tick_params(labelsize=10)
    cb.set_label(label=r'$T_{surf}$',size=10)

##### Add figure Labeling
ax[0].text(-.3,1075,'a)',fontsize=8)
grid[0].text(-17,-4,'b)',fontsize=8)


plt.savefig('../../plot_output/dke1/lidar_prof_sfc.png', bbox_inches = "tight")

# %%
fig=plt.figure(figsize=(5.5,7))
subfigs = fig.subfigures(3, 1, hspace=0,wspace=0,frameon=False,height_ratios=[1.25,1.25,1.1])
ax1=subfigs[1].subplots(1,3)
ax=subfigs[0].subplots(1,3)
grid=ImageGrid(subfigs[2], 111,  # similar to subplot(111)
                nrows_ncols=(1, 3),
                axes_pad=0.1,
                cbar_mode='each',
                cbar_location='bottom',
                cbar_pad=.02,
                cbar_size="5%")
vmax=np.nanpercentile(goes,95)
vmin=296
vmax=310
for i in range(3):
    for k in range(5):
        s=list(colors.keys())[k]
        ax[i].plot(profs[i][s],z,'-o',color=colors[s],markersize=2,linewidth=.5,label=s)
    ax1[i].plot(dkep[i][0:35],z[0:35],'-o',color='black',markersize=2)
    ax1[i].set_xlim(-.02,1.35)
    if i==2:
        l=ax[i].legend(fontsize=7,loc='lower right',framealpha=.95)
        #l.set_labelsize(8)
    im=grid[i].imshow(goes[i],cmap='coolwarm',vmin=vmin,vmax=vmax)
    #im=grid[i].imshow(goes[i],cmap='coolwarm')
    xx=[]
    yy=[]
    clrs=[]
    for k in scatloc[0].keys():
        xx.append(scatloc[i][k][0])
        yy.append(scatloc[i][k][1])
        clrs.append(matplotlib.colors.to_rgb(colors[k]))
    xx=np.array(xx)
    yy=np.array(yy)
    grid[i].scatter(yy,xx,s=15,c=clrs,edgecolors='black',linewidth=.5,alpha=.95,zorder=5)

    if i==0:
        ax[i].set_yticks([200,400,600,800,1000],[200,400,600,800,1000])
        ax[i].set_ylabel(r'height $(m)$')
        ax1[i].set_yticks([200,400,600,800,1000],[200,400,600,800,1000])
        ax1[i].set_ylabel(r'height $(m)$')
    else:
        ax[i].set_yticks([200,400,600,800,1000],[])
        ax1[i].set_yticks([200,400,600,800,1000],[])
    ax[i].tick_params(axis='both',which='major',labelsize=8)
    ax[i].set_xlabel(r'$\overline{u}$  ($m\ s^{-1}$)',fontsize=8)

    ax1[i].tick_params(axis='both',which='major',labelsize=8)
    ax1[i].set_xlabel(r'$DKE$  ($m^{2}\ s^{-2}$)',fontsize=8)
    #ax[i].grid(True)

    mag=np.nanmean(profs[i]['C1'][0:15])
    print(np.rad2deg(wdir[i]))
    grid[i].arrow(20, 20, mag*np.sin(wdir[i]), -mag*np.cos(wdir[i]), linewidth=2, head_width=0.2, head_length=0.1,color='black',alpha=.6)

    grid[i].set_xticks([])
    grid[i].set_yticks([])

    ax[i].set_title(days[i][0:4]+'-'+days[i][4:6]+'-'+days[i][6:8],fontsize=10)

    tit=r'$\lambda_{T_s}$:'+' '+str(lhet[i])[0:5]+'   '+r'$\sigma_{T_s}$:'+' '+str(std[i])[0:3]
    grid[i].set_title(tit,fontsize=8)
    ax[i].set_ylim(0,1025)
    ax1[i].set_ylim(0,1025)
    cb=grid.cbar_axes[i].colorbar(im,label=r'$T_{s}$')
    grid.cbar_axes[i].tick_params(labelsize=10)
    cb.set_label(label=r'$T_{s}$',size=10)

##### Add figure Labeling
ax[0].text(-.3,1075,'a)',fontsize=8)
ax1[0].text(-.42,1075,'b)',fontsize=8)
grid[0].text(-17,-4,'c)',fontsize=8)

plt.savefig('../../plot_output/dke1/lidar_prof_sfc_dke2.png', bbox_inches = "tight")

# %%

# %% [markdown]
# # LIDAR DKE through Time

# %% [markdown]
# #### Data Prep

# %%

# %%
dxylo=[]
dxyhi=[]
dzlo=[]
dzhi=[]

mxylo=[]
mxyhi=[]
mzlo=[]
mzhi=[]

for i in range(len(fnc['date'][:])):
    std=fnc['lst_std'][i]
    if std >=.75:
        dxyhi.append(fnc['DKE_xy'][i,:,:]/fnc['weighting'][i,:,:])
        dzhi.append(fnc['DKE_z'][i,:,:]/fnc['weighting'][i,:,:])
        mxyhi.append(fnc['MKE_xy'][i,:,:]/fnc['weighting'][i,:,:])
        mzhi.append(fnc['MKE_z'][i,:,:]/fnc['weighting'][i,:,:])
    else:
        dxylo.append(fnc['DKE_xy'][i,:,:]/fnc['weighting'][i,:,:])
        dzlo.append(fnc['DKE_z'][i,:,:]/fnc['weighting'][i,:,:])
        mxylo.append(fnc['MKE_xy'][i,:,:]/fnc['weighting'][i,:,:])
        mzlo.append(fnc['MKE_z'][i,:,:]/fnc['weighting'][i,:,:])

dxylo=np.array(dxylo)
dxyhi=np.array(dxyhi)
dzlo=np.array(dzlo)
dzhi=np.array(dzhi)

mxylo=np.array(mxylo)
mxyhi=np.array(mxyhi)
mzlo=np.array(mzlo)
mzhi=np.array(mzhi)

# %% [markdown]
# #### Plotting

# %%
##### SETUP
version=['dke','dke_mke'][1]
cmap='nipy_spectral'
mn=2
mx=98
fig=plt.figure(figsize=(5.5,3.5))
subfigs = fig.subfigures(2, 1, hspace=0,wspace=0,frameon=False,height_ratios=[1,1])
gxy=ImageGrid(subfigs[0], 111,  # similar to subplot(111)
                nrows_ncols=(1, 2),
                axes_pad=0.1,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=.04,
                cbar_size="5%")
gz=ImageGrid(subfigs[1], 111,  # similar to subplot(111)
                nrows_ncols=(1, 2),
                axes_pad=0.1,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=.04,
                cbar_size="5%")

height=fnc['height'][:]
time=fnc['hour'][:].astype(int)

##### Final Data Prep
if version=='dke':
    d_xyl=np.nanmean(dxylo,axis=0)
    d_xyh=np.nanmean(dxyhi,axis=0)
    d_zl=np.nanmean(dzlo,axis=0)
    d_zh=np.nanmean(dzhi,axis=0)
    clabelxy=r'$DKE_{xy}$ ($m^{2}\ s^{-2}$)'
    clabelz=r'$DKE_{z}$ ($m^{2}\ s^{-2}$)'
elif version=='dke_mke':
    d_xyl=np.nanmedian(dxylo/mxylo,axis=0)
    d_xyh=np.nanmedian(dxyhi/mxyhi,axis=0)
    d_zl=np.nanmedian(dzlo/mzlo,axis=0)
    d_zh=np.nanmedian(dzhi/mzhi,axis=0)
    clabelxy=r'$DKE_{xy}/MKE_{xy}$'
    clabelz=r'$DKE_{z}/MKE_{z}$'

vmn_xy=min(np.nanpercentile(d_xyl,mn),np.nanpercentile(d_xyh,mn))
vmx_xy=max(np.nanpercentile(d_xyl,mx),np.nanpercentile(d_xyh,mx))
vmn_z=min(np.nanpercentile(d_zl,mn),np.nanpercentile(d_zh,mn))
vmx_z=max(np.nanpercentile(d_zl,mx),np.nanpercentile(d_zh,mx))

xticks=np.linspace(0,len(time)-1,len(time))
xticks_l=['8:00','','10:00','','12:00','','14:00','','16:00','','18:00']
yticks=[0.4,4.2,8,11.8,15.6,19.6,23.4,27.25,31]
yticks_l=['','200','','400','','600','','800','']

##### Plotting Top
gxy[0].imshow(d_xyl.T[0:-1,:],cmap=cmap,origin='lower',aspect=.2,vmin=vmn_xy,vmax=vmx_xy,interpolation='spline16')
im=gxy[1].imshow(d_xyh.T[0:-1,:],cmap=cmap,origin='lower',aspect=.2,vmin=vmn_xy,vmax=vmx_xy,interpolation='spline16')

#gxy[1].set_xticks([0,3,6,9],[])
#gxy[0].set_xticks([0,3,6,9],[])
#gxy[0].set_yticks([4.2,11.8,19.6,27.25],[200,400,600,800])
#gxy[1].set_yticks([4.2,11.8,19.6,27.25])

gxy[0].set_xticks(xticks,[])
gxy[1].set_xticks(xticks,[])
gxy[0].set_yticks(yticks,yticks_l)
gxy[1].set_yticks(yticks,yticks_l)


gxy[0].set_title('Less Heterogeneous',fontsize=10)
gxy[1].set_title('Very Heterogeneous',fontsize=10)
gxy[0].tick_params(labelsize=8)
gxy[1].tick_params(labelsize=8)
gxy[0].set_ylabel(r'Height $(m)$',fontsize=8)
gxy[0].grid(True,alpha=1,color='black',linewidth=.1)
gxy[1].grid(True,alpha=1,color='black',linewidth=.1)

cb=gxy.cbar_axes[0].colorbar(im)
gxy.cbar_axes[0].tick_params(labelsize=8)
cb.set_label(label=clabelxy,size=8)


##### Plotting Bottom
gz[0].imshow(d_zl.T[0:-1,:],cmap=cmap,origin='lower',aspect=.2,vmin=vmn_z,vmax=vmx_z,interpolation='spline16')
im=gz[1].imshow(d_zh.T[0:-1,:],cmap=cmap,origin='lower',aspect=.2,vmin=vmn_z,vmax=vmx_z,interpolation='spline16')

#gz[1].set_xticks([0,3,6,9],time[::3])
#gz[0].set_xticks([0,3,6,9],time[::3])
#gz[0].set_yticks([4.2,11.8,19.6,27.25],[200,400,600,800])
#gz[1].set_yticks([4.2,11.8,19.6,27.25])

gz[0].set_xticks(xticks,xticks_l,rotation=45)
gz[1].set_xticks(xticks,xticks_l,rotation=45)
gz[0].set_yticks(yticks,yticks_l)
gz[1].set_yticks(yticks,yticks_l)

gz[0].tick_params(labelsize=8)
gz[1].tick_params(labelsize=8)
gz[0].set_ylabel(r'Height $(m)$',fontsize=8)
gz[0].set_xlabel('Local Hour',fontsize=8)
gz[1].set_xlabel('Local Hour',fontsize=8)
gz[0].grid(True,alpha=1,color='black',linewidth=.1)
gz[1].grid(True,alpha=1,color='black',linewidth=.1)

cb=gz.cbar_axes[0].colorbar(im)
gz.cbar_axes[0].tick_params(labelsize=8)
cb.set_label(label=clabelz,size=8)

#### Add Figure Labeling
gxy[0].text(-.5,37,'a)',fontsize=8)
gxy[1].text(-.5,37,'b)',fontsize=8)
gz[0].text(-.5,37,'c)',fontsize=8)
gz[1].text(-.5,37,'d)',fontsize=8)

plt.savefig('../../plot_output/dke1/lidar_prof_time_ratio.png', bbox_inches = "tight")

# %%
##### SETUP
version=['dke','dke_mke'][1]
cmap='nipy_spectral'
mn=2
mx=98
fig=plt.figure(figsize=(5.5,2))
subfigs = fig.subfigures(1, 1, hspace=0,wspace=0,frameon=False,height_ratios=[1])
gxy=ImageGrid(subfigs, 111,  # similar to subplot(111)
                nrows_ncols=(1, 2),
                axes_pad=0.1,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=.04,
                cbar_size="5%")

height=fnc['height'][:]
time=fnc['hour'][:].astype(int)

##### Final Data Prep
if version=='dke':
    d_xyl=np.nanmean(dxylo,axis=0)
    d_xyh=np.nanmean(dxyhi,axis=0)
    d_zl=np.nanmean(dzlo,axis=0)
    d_zh=np.nanmean(dzhi,axis=0)
    clabelxy=r'$DKE_{xy}$ ($m^{2}\ s^{-2}$)'
    clabelz=r'$DKE_{z}$ ($m^{2}\ s^{-2}$)'
elif version=='dke_mke':
    d_xyl=np.nanmedian((dxylo+dzlo)/(mxylo+mzlo),axis=0)
    d_xyh=np.nanmedian((dxyhi+dzhi)/(mxyhi+mzhi),axis=0)
    clabelxy=r'$DKE/MKE$'

vmn_xy=min(np.nanpercentile(d_xyl,mn),np.nanpercentile(d_xyh,mn))
vmx_xy=max(np.nanpercentile(d_xyl,mx),np.nanpercentile(d_xyh,mx))

xticks=np.linspace(0,len(time)-1,len(time))
xticks_l=['8:00','','10:00','','12:00','','14:00','','16:00','','18:00']
yticks=[0.4,4.2,8,11.8,15.6,19.6,23.4,27.25,31]
yticks_l=['','200','','400','','600','','800','']

##### Plotting Top
gxy[0].imshow(d_xyl.T[0:-1,:],cmap=cmap,origin='lower',aspect=.2,vmin=vmn_xy,vmax=vmx_xy,interpolation='spline16')
im=gxy[1].imshow(d_xyh.T[0:-1,:],cmap=cmap,origin='lower',aspect=.2,vmin=vmn_xy,vmax=vmx_xy,interpolation='spline16')

#gxy[1].set_xticks([0,3,6,9],[])
#gxy[0].set_xticks([0,3,6,9],[])
#gxy[0].set_yticks([4.2,11.8,19.6,27.25],[200,400,600,800])
#gxy[1].set_yticks([4.2,11.8,19.6,27.25])

gxy[0].set_xticks(xticks,xticks_l,rotation=45)
gxy[1].set_xticks(xticks,xticks_l,rotation=45)
gxy[0].set_yticks(yticks,yticks_l)
gxy[1].set_yticks(yticks,yticks_l)


gxy[0].set_title('Less Heterogeneous',fontsize=10)
gxy[1].set_title('Very Heterogeneous',fontsize=10)
gxy[0].tick_params(labelsize=8)
gxy[1].tick_params(labelsize=8)
gxy[0].set_ylabel(r'Height $(m)$',fontsize=8)
gxy[0].grid(True,alpha=1,color='black',linewidth=.1)
gxy[1].grid(True,alpha=1,color='black',linewidth=.1)
gxy[0].set_xlabel('Local Hour',fontsize=8)
gxy[1].set_xlabel('Local Hour',fontsize=8)

cb=gxy.cbar_axes[0].colorbar(im)
gxy.cbar_axes[0].tick_params(labelsize=8)
cb.set_label(label=clabelxy,size=8)


#### Add Figure Labeling
gxy[0].text(-.5,37,'a)',fontsize=8)
gxy[1].text(-.5,37,'b)',fontsize=8)

plt.savefig('../../plot_output/dke1/lidar_prof_time_ratio_full.png', bbox_inches = "tight")

# %% [markdown]
# # Data for Filtering Cells 
# (run before next few cells)

# %%
h=16

dke=fnc['DKE_z'][:]+fnc['DKE_xy'][:]
mke=fnc['MKE_z'][:]+fnc['MKE_xy'][:]
wfac=np.nanmean(np.sum((fnc['weighting'][:,:,:])[:,2:8,0:h],axis=2),axis=1)

_mke=1/(np.nanmean(np.sum((mke)[:,2:8,0:h],axis=2),axis=1)/wfac)
dke_1=np.nanmean(np.sum((dke/1)[:,2:8,0:h],axis=2),axis=1)/wfac

merror=np.abs(fnc['wind_a_diff'][:]).T
merror[merror>190]=360-merror[merror>190]
merror=np.nanmean(merror,axis=0)

dke_1[merror>60]=float('nan')
_mke[merror>60]=float('nan')
rat=_mke*dke_1


ws=np.mean(fnc['wind_speed'][:,2:8,0:5],axis=(1,2)) #first 200m
ws2=np.mean(fnc['wind_speed'][:,2:8,-2],axis=(1)) #1000m
winda=np.rad2deg(circmean(np.deg2rad(fnc['wind_a'][:,2:8,0:5]),axis=(1,2),nan_policy='omit'))
lsta=fnc['lst_a'][:]
lsta2=fnc['lst_a2'][:]
abet=angle_diff(winda,lsta)
abet2=angle_diff(lsta,lsta2)
windp=np.cos(abet)*ws
repo=np.sum(fnc['sites_repo'][:,2:8,0:h,:],axis=3)
repo=np.mean(repo,axis=(1,2))
vort=fnc['vort'][:]
lhet=fnc['lst_std'][:]/fnc['lst_mean'][:]*fnc['lst_lhet'][:]
lhet2=np.log10(lhet*9.81/(ws*np.cos(abet))**2)
cv=fnc['lst_std'][:]/fnc['lst_mean'][:]
cv_s=fnc['lst_std_site'][:]/fnc['lst_mean'][:]
cv_pll=fnc['lst_std_perpendicular'][:]/fnc['lst_mean'][:]
cv_ppd=fnc['lst_std_parallel'][:]/fnc['lst_mean'][:]
lhet_0=fnc['lst_lhet'][:]
rain=np.nanmean(fnc['precip'][:,2:8],axis=1)
rain[rain>0]=2
rain[np.isnan(rain)]=1


# %%

# %% [markdown]
# # Progressive Filtering

# %% [markdown]
# #### Data Prep

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
np.nanmean(fnc['weighting'][:])

# %%
corr=np.zeros((2,5))
corr1=np.zeros((2,5))
corr2=np.zeros((2,5))
xx=np.ones((2,5,273))*float('nan')
yy=np.ones((2,5,273))*float('nan')
yy1=np.ones((2,5,273))*float('nan')
yy2=np.ones((2,5,273))*float('nan')
count=np.zeros((2,5))

lims=[[3,60,20,1,16e-5],[3,70,15,0,25e-5]]

for i in range(2):
    m=np.ones((273,)).astype(bool)
    m=m&(rat<50)#&(ws>3)
    m1=m&(repo>=lims[i][0])
    m2=m1&(ws2<lims[i][2])
    m3=m2&(abet>lims[i][1])
    m4=m3&(rain<=lims[i][3])
    m5=m4&(vort<lims[i][4])
    ms=[m1,m2,m3,m4,m5]
    for j in range(5):
        xx[i,j,ms[j]]=lhet[ms[j]]
        yy[i,j,ms[j]]=rat[ms[j]]
        yy1[i,j,ms[j]]=_mke[ms[j]]
        yy2[i,j,ms[j]]=dke_1[ms[j]]
        corr[i,j]=spearmanr(xx[i,j,:],yy[i,j,:],nan_policy='omit')[0]
        corr1[i,j]=spearmanr(xx[i,j,:],yy1[i,j,:],nan_policy='omit')[0]
        corr2[i,j]=spearmanr(xx[i,j,:],yy2[i,j,:],nan_policy='omit')[0]
        count[i,j]=np.sum(ms[j])




# %%

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# #### Plotting Old

# %%
fig,axs=plt.subplots(4,2,figsize=(4,6))
for i in range(2):
    for j in range(4):
        ax=axs[j,i]
        ax.scatter(xx[i,j],yy[i,j],s=3,c=color)
        #ax.set_xlim(0,340)

        # regular
        # ax.text(150,33,r'$\rho_s=$'+str(np.round(corr[i,j],3)))
        # ax.set_ylim(-1,40)

        # logscale
        ax.semilogy()
        ax.set_title(r'$\rho_s=$'+str(np.round(corr[i,j],3)))
        #ax.text(20000,1133,r'$\rho_s=$'+str(np.round(corr[i,j],3)),zorder=5)
        ax.set_ylim(.05,50)
fig.subplots_adjust(hspace=.5)

# %%
axs.shape

# %% [markdown]
# # Check Correlations

# %%
fig,axs=plt.subplots(3,2,figsize=(4,6),width_ratios=[1,1.25])
idx=4
import matplotlib as mpl

vmax=15
vmin=0
clabel= r'$u_g$ ($ms^{-1}$)'
rt_ws=(ws-ws2)/ws
color=plt.get_cmap('Spectral')(ws2/vmax)

#vmax=.75
#vmin=0
#color=plt.get_cmap('Spectral')(np.abs(rt_ws)/.75)
    
axs[0,0].scatter(xx[0,0],yy[0,0],s=5,c=color)
axs[0,0].set_title('No Filter\n'+r'$\rho_s=$'+str(np.round(corr[0,0],3)))
axs[0,0].semilogy()
axs[0,0].set_ylabel(r'$DKE/MKE$')
#axs[0,0].set_ylim(.05,50)

axs[0,1].scatter(xx[0,0],yy[0,0],s=5,c='grey',alpha=.05)
axs[0,1].scatter(xx[1,idx],yy[1,idx],s=5,c=color)
axs[0,1].semilogy()
axs[0,1].set_title('Strong Filter\n'+r'$\rho_s=$'+str(np.round(corr[1,idx],3)))
#axs[0,1].set_ylim(.05,50)
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap='Spectral'),
             ax=axs[0,1], orientation='vertical', label=clabel)

axs[1,0].scatter(xx[0,0],yy1[0,0],s=5,c=color)
axs[1,0].set_title(r'$\rho_s=$'+str(np.round(corr1[0,0],3)))
axs[1,0].semilogy()
axs[1,0].set_ylabel(r'$1/MKE$ ($m^{-2}\ s^{2}$)')
#axs[1,0].set_ylim(.000005,.002)

axs[1,1].scatter(xx[0,0],yy1[0,0],s=5,c='grey',alpha=.05)
axs[1,1].scatter(xx[1,idx],yy1[1,idx],s=5,c=color)
axs[1,1].semilogy()
axs[1,1].set_title(r'$\rho_s=$'+str(np.round(corr1[1,idx],3)))
#axs[1,1].set_ylim(.000005,.002)
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap='Spectral'),
             ax=axs[1,1], orientation='vertical', label=clabel)

axs[2,0].scatter(xx[0,0],yy2[0,0],s=5,c=color)
axs[2,0].set_title(r'$\rho_s=$'+str(np.round(corr2[0,0],3)))
axs[2,0].semilogy()
axs[2,0].set_ylabel(r'$DKE$ ($m^{2}\ s^{-2}$)')
#axs[2,0].set_ylim(.2,70)

axs[2,1].scatter(xx[0,0],yy2[0,0],s=5,c='grey',alpha=.05)
axs[2,1].scatter(xx[1,idx],yy2[1,idx],s=5,c=color)
axs[2,1].semilogy()
axs[2,1].set_title(r'$\rho_s=$'+str(np.round(corr2[1,idx],3)))
#axs[2,1].set_ylim(.2,70)
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap='Spectral'),
             ax=axs[2,1], orientation='vertical', label=clabel)

axs[2,0].set_xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s}$  ($m$)')
axs[2,1].set_xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s}$  ($m$)')

print(lims[1][0:idx+1])

plt.subplots_adjust(hspace=.5,wspace=.33)
plt.savefig('../../plot_output/dke1/filter_select_tsw.png', bbox_inches = "tight")

# %%

# %% [markdown]
# # Check Correlations -- Sensitivity

# %%
Nr=4
Nw=20
Nv=20
Na=20
Ns=2


corr=np.zeros((3,Nr*Nw*Nv*Na*Ns))
count=np.zeros((3,Nr*Nw*Nv*Na*Ns))
al_=np.linspace(0,85,Na)
vl_=np.linspace(8,35,Nv-1)
vl_=np.append(vl_,50)
vl_=vl_*10**(-5)
wl_=np.linspace(5,15,Nw-2)
wl_=np.append(wl_,[17.5,20])
rl_=np.array([2,3,4,5])
sl_=[0,2]
for y in range(3):
    dat=[rat,_mke,dke_1][y]
    als=[]
    vls=[]
    wls=[]
    rls=[]
    sls=[]
    i=0
    for r in range(Nr):
        print('.',end='',flush=True)
        rl=rl_[r]
        for s in range(Ns):
            sl=sl_[s]
            for w in range(Nw):
                wl=wl_[w]
                for v in range(Nv):
                    vl=vl_[v]
                    for a in range(Na):
                        al=al_[a]
                        m=(repo>=rl)
                        m=m&(ws2<wl)
                        m=m&(abet>al)
                        m=m&(vort<vl)
                        m=m&(rain<=sl)
                        xx=lhet[m]
                        yy=dat[m]
                        count[y,i]=np.sum(m)
                        try:
                            corr[y,i]=spearmanr(xx,yy,nan_policy='omit')[0]
                        except:
                            corr[y,i]=float('nan')
                        als.append(al)
                        vls.append(vl)
                        wls.append(wl)
                        rls.append(rl)
                        sls.append(sl)
    
                        i=i+1

# %%
fig,axs=plt.subplots(3,5,figsize=(6,3))
for v in range(5):
    l_=[sl_,rl_,wl_,vl_,al_][v]
    ls=[sls,rls,wls,vls,als][v]
    ls=np.array(ls)
    flip=[True,False,True,True,False][v]
    xticks=[[0,2],[2,3,4,5],[20,15,10,5],[.0005,.0004,.0003,.0002,.0001],[0,20,40,60,80]][v]
    xticksl=[['0',r'$\infty$'],[2,3,4,5],[20,15,10,5],[r'$5e-4$','','','',r'$1e-4$'],[0,20,40,60,80]][v]
    yticks=[.1,.2,.3,.4,.5]
    yticksl=['0.1','','0.3','','0.5']
    xlabel=[r'MAX Precip.'+'\n'+r'($mm$)','MIN sites\nreporting',r'MAX $u_g$ ($ms^{-1}$)',r'MAX $|\zeta|$',r'MIN $\alpha$'][v]
    for y in range(3):
        ylabel=[r'$\rho_{DKE/MKE}$',r'$\rho_{1/MKE}$',r'$\rho_{DKE}$'][y]
        ax=axs[y,v]
        color=plt.get_cmap('nipy_spectral')(count[y,:]/273)
        ylo=[]
        yhi=[]
        ym=[]
        for j in range([Ns,Nr,Nw,Nv,Na][v]):
            m=(ls==l_[j])&(count[y,:]>10)
            if np.sum(m)<200:
                ym.append(float('nan'))
                ylo.append(float('nan'))
                yhi.append(float('nan'))
            else:
                ylo.append(np.nanpercentile(corr[y,m],25))
                yhi.append(np.nanpercentile(corr[y,m],75))
                ym.append(np.nanpercentile(corr[y,m],50))
        ym=np.array(ym)
        ylo=np.array(ylo)
        yhi=np.array(yhi)
        ax.plot(l_,ym,'-o',markersize=2,linewidth=.75,color='darkgreen')
        ax.fill_between(l_,ylo,yhi,alpha=.5,color='darkgreen')
        if flip:
            ax.invert_xaxis()
        ax.set_ylim(.01,.55)
        ax.grid(True,color='black',linewidth=.2)
        if y<2:
            ax.set_xticks(xticks,[])
        else:
            ax.set_xticks(xticks,xticksl,fontsize=8)
            ax.set_xlabel(xlabel,fontsize=10)
        if v==0:
            ax.set_ylabel(ylabel)
            ax.set_yticks(yticks,yticksl,fontsize=8)
        else:
            ax.set_yticks(yticks,[])
        #ax.scatter(vs[v],corr[y,:],s=5,color=color)
plt.savefig('../../plot_output/dke1/filter_sensitivity_precip3.png', bbox_inches = "tight")

# %%

# %%

# %%

# %%

# %% [markdown]
# # DKE vs Alpha

# %% [markdown]
# #### Data Prep

# %%

# %% [markdown]
# #### Plotting

# %%

# %%

# %%

# %% [markdown]
# # Heterogeneity Differences

# %% [markdown]
# #### Data Prep

# %%
m=(ws2<15)&(abet>70)&(repo>=3)&(vort<2.5*10**(-4))&(rain<=0)


color=plt.get_cmap('Spectral')(ws2/15)

corr=np.zeros((2,4))
#corr2=np.zeros((2,4))
xx=np.ones((2,4,273))*float('nan')
yy=np.ones((2,4,273))*float('nan')

for v in range(2):
    for j in range(4):
        xx[v,j,m]=[cv,cv_s,lhet_0,lhet][j][m]
        yy[v,j,m]=[rat,dke_1][v][m]
        corr[v,j]=spearmanr(xx[v,j],yy[v,j],nan_policy='omit')[0]


# %% [markdown]
# #### Plotting

# %%
fig,axs=plt.subplots(2,4,figsize=(6,3),width_ratios=[1,1,1,1.25])
for i in range(2):
    for j in range(4):
        ax=axs[i,j]
        ax.scatter(xx[i,j],yy[i,j],s=3,c=color)

        # logscale
        ax.semilogy()
        ax.grid(True,color='black',linewidth=.2)
        ax.set_title(r'$\rho_s=$'+str(np.round(corr[i,j],3)))
        
        if j>0:
            ax.set_yticklabels([],minor=False)
        else:
            ax.set_ylabel([r'$DKE/MKE$',r'$DKE$ ($m^2\ s^{-2}$)'][i])
        if i==0:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(ax.get_xticks(),rotation=45)
            ax.set_xlabel([r'$CV_{full}$',r'$CV_{lidar}$',r'$\lambda_{T_s}$ ($m$)',r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s}$  ($m$)'][j])
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 15), cmap='Spectral'),
             ax=axs[0,3], orientation='vertical', label=r'$u_g$ ($ms^{-1}$)')
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 15), cmap='Spectral'),
             ax=axs[1,3], orientation='vertical', label=r'$u_g$ ($ms^{-1}$)')
fig.subplots_adjust(hspace=.3,wspace=.1)

plt.savefig('../../plot_output/dke1/het_type_sensitivity_.png', bbox_inches = "tight")

# %%

# %% [markdown]
# # Heterogeneity Differences Extended

# %%
m=(ws2<15)&(repo>=3)&(vort<2.5*10**(-4))&(rain<=0)&(abet>70)

color=plt.get_cmap('Spectral')(ws2/15)

corr=np.zeros((2,6))
#corr2=np.zeros((2,4))
xx=np.ones((2,6,273))*float('nan')
yy=np.ones((2,6,273))*float('nan')

for v in range(2):
    for j in range(6):
        xx[v,j,m]=[cv,cv_s,cv_ppd,cv_pll,lhet_0,lhet][j][m]
        yy[v,j,m]=[rat,dke_1][v][m]
        corr[v,j]=spearmanr(xx[v,j],yy[v,j],nan_policy='omit')[0]

# %%
fig,axs=plt.subplots(2,6,figsize=(6,2),width_ratios=[1,1,1,1,1,1.2])
for i in range(2):
    for j in range(6):
        ax=axs[i,j]
        ax.scatter(xx[i,j],yy[i,j],s=3,c=color)

        # logscale
        ax.semilogy()
        ax.grid(True,color='black',linewidth=.2)
        ax.set_title(r'$\rho_s=$'+str(np.round(corr[i,j],3)),fontsize=8)
        
        if j>0:
            ax.set_yticklabels([],minor=False)
        else:
            ax.set_ylabel([r'$DKE/MKE$',r'$DKE$ ($m^2\ s^{-2}$)'][i])
        if i==0:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(ax.get_xticks(),rotation=45,fontsize=8)
            ax.set_xlabel([r'$CV_{full}$',r'$CV_{lidar}$',r'$CV_{\perp}$',r'$CV_{||}$',r'$\lambda_{T_s}$ ($m$)',r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s}$  ($m$)'][j])
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 15), cmap='Spectral'),
            ax=axs[0,5], orientation='vertical', label=r'$u_g$ ($ms^{-1}$)')
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 15), cmap='Spectral'),
            ax=axs[1,5], orientation='vertical', label=r'$u_g$ ($ms^{-1}$)')
ax.tick_params(labelsize=8)
fig.subplots_adjust(hspace=.3,wspace=.1)

plt.savefig('../../plot_output/dke1/het_type_sensitivity_ext.png', bbox_inches = "tight")

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
fnc.close()

# %% [markdown]
# # Full std vs Site std

# %% [markdown]
# #### Data Prep

# %%

# %% [markdown]
# #### Plotting

# %%
fig=plt.figure(figsize=(2.75,5))
ax=plt.subplot(2,1,1)
nmrpt=np.nanmean(np.nansum(fnc['sites_repo'][:],axis=3),axis=(1,2))
color=plt.get_cmap('coolwarm')((nmrpt-2)/3)
plt.scatter(fnc['lst_std'][:],fnc['lst_std_site'][:],s=1,alpha=.4,c=color)
plt.plot([0,5],[0,5],'k--',alpha=.5,linewidth=.5)
plt.grid(True,color='black',linewidth=.2,alpha=.5)
plt.xlabel(r'$\sigma_{T_s,full}$')
plt.ylabel(r'$\sigma_{T_s,lidars}$')
plt.xticks([0,1,2,3,4,5])
plt.yticks([0,1,2,3,4,5])
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(2, 5), cmap='coolwarm'),
             ax=ax, orientation='vertical', label='Sites Reporting')

ax=plt.subplot(2,1,2)
color=plt.get_cmap('coolwarm')((nmrpt-2)/3)
plt.scatter(ws2,fnc['lst_std'][:],s=1,alpha=.5,c=color)
plt.plot([0,5],[0,5],'k--',alpha=0)
plt.grid(True,color='black',linewidth=.2,alpha=.5)
plt.ylabel(r'$\sigma_{T_s,full}$')
plt.xlabel(r'$u_g$ ($ms^{-1}$)')
plt.yticks([0,1,2,3,4,5])
plt.xlim(1,16)
plt.subplots_adjust(hspace=.3)
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(2, 5), cmap='coolwarm'),
             ax=ax, orientation='vertical', label='Sites Reporting')

plt.savefig('../../plot_output/dke1/std_full_lidar_ws.png', bbox_inches = "tight")

# %%

# %%

# %% [markdown]
# # LES Figure -- Rmse

# %%
klist=list(fles.keys())
klist.sort()
netlist=list(fles[klist[0]].keys())
Nn=len(netlist)-1
Nk=len(klist)
Nit=100

netrmse=np.zeros((Nn,Nk))
dke_les=np.zeros((Nk,))
het_=np.zeros((Nk,))
ell_=np.zeros((Nk,))
rt_0=np.zeros((Nk,))
rt_3=np.zeros((Nk,Nit))
rt_5=np.zeros((Nk,Nit))
rt_7=np.zeros((Nk,Nit))
rt_10=np.zeros((Nk,Nit))
rt_100=np.zeros((Nk,Nit))
for i in range(Nk):
    k=klist[i]
    rt_0[i]=fles[k][0]['dke']/fles[k][0]['mke']
    based=fles[k][0]['dke']
    dke_les[i]=based
    for j in range(Nn):
        rt=np.array(fles[k][netlist[j]]['dkes'])/np.array(fles[k][netlist[j]]['mkes'])
        #netrmse[j,i]=np.sqrt(mean_squared_error([based]*Nit,fles[k][netlist[j]]['dkes']))/based
        netrmse[j,i]=np.sqrt(mean_squared_error([rt_0[i]]*Nit,rt))/rt_0[i]
        if netlist[j]==3:
            rt_3[i,:]=rt
        if netlist[j]==5:
            rt_5[i,:]=rt
        if netlist[j]==7:
            rt_7[i,:]=rt
        if netlist[j]==10:
            rt_10[i,:]=rt
        if netlist[j]==100:
            rt_100[i,:]=rt
    het_[i]=fles_hets[k]


# %% [markdown]
# #### Plotting

# %%
np.nanmean(rt_3,axis=1).shape

# %% jupyter={"source_hidden": true, "outputs_hidden": true}
#plt.hist(rt_0,bins=np.linspace(0,2,20))
site=6
plt.hist(rt_3[site,:],bins=np.linspace(0,1.5,20),alpha=.5)
plt.hist(rt_10[site,:],bins=np.linspace(0,1.5,20),alpha=.5)
plt.hist(rt_100[site,:],bins=np.linspace(0,1.5,20),alpha=.5)
plt.plot([rt_0[site],rt_0[site]],[0,100])
plt.title('')

# %%
plt.figure(figsize=(6,3))
pos=np.array(netlist)
mcolor='navy'
flierprops={'markersize':2}
medianprops = dict(color=mcolor)
boxprops = dict(linestyle='-', linewidth=.5)
widths=pos[0:-1]/12#np.ones((len(pos),))
bplot=plt.boxplot(netrmse.T*100,positions=pos[0:-1],widths=widths,patch_artist=True,medianprops=medianprops,boxprops=boxprops,flierprops=flierprops)
for patch in bplot['boxes']:
        patch.set_facecolor('darkgreen')
        patch.set_alpha(.55)
plt.plot(pos[0:-1],np.nanmedian(netrmse.T,axis=0)*100,linewidth=.5,linestyle='--',color=mcolor,alpha=.75)
plt.xscale('log')
plt.xlabel('Number of Sites in Virtual Network')
plt.ylabel(r'nRMSE DKE/MKE (%)')
plt.savefig('../../plot_output/dke1/les_network_rmse.png', bbox_inches = "tight")

# %%

# %%

# %%

# %% [markdown]
# # LES Figure -- Scatters

# %% [markdown]
# #### Data Prep

# %%
rt.shape

# %% [markdown]
# #### Plotting

# %%
fig,axs=plt.subplots(2,3,figsize=(5.5,3))
axs=axs.flatten()
ymin=.02
ymax=10
xticks=[50,100,150]
yticks=[.1,1,10]
yticks_l=[1,'',3,'',5]
ylabel=r'$DKE_v/MKE_v$'

# handle first axis
ax=axs[0]
rho=spearmanr(het_,rt_0[:],nan_policy='omit')[0]
ax.scatter(het_,rt_0[:],color='darkgreen',s=2,alpha=.55)
ax.set_yscale('log')
#ax.set_yticks(yticks)
ax.set_ylabel(ylabel,fontsize=10)
ax.set_ylim(ymin,ymax)
ax.set_xticks(xticks,[])
ax.text(90,ymax*.4,r'$\rho=$'+str(rho)[0:5],fontsize=8,bbox=dict(facecolor='white',alpha=.7,linewidth=0))
ax.text(10,ymax*.4,r'$N=\text{full}$',fontsize=8,bbox=dict(facecolor='white',alpha=.7,linewidth=0))

for i in range(1,6):
    rt=np.nanmedian([0,rt_3,rt_5,rt_7,rt_10,rt_100][i],axis=1)
    #lo=np.nanpercentile([0,rt_3,rt_5,rt_7,rt_10,rt_100][i],25,axis=1)-rt
    err=np.nanstd([0,rt_3,rt_5,rt_7,rt_10,rt_100][i],axis=1)
    #hi=np.nanpercentile([0,rt_3,rt_5,rt_7,rt_10,rt_100][i],90,axis=1)-rt
    #print(np.nanmean((err)))
    #err=[hi*0,hi]
    ax=axs[i]
    rho=spearmanr(het_,rt,nan_policy='omit')[0]
    ax.scatter(het_,rt,color='darkgreen',alpha=.55,s=err*100)
    ax.set_yscale('log')
    ax.set_ylim(ymin,ymax)
    ax.tick_params(labelsize=8)
    if i<3:
        ax.set_xticks(xticks,[])
    else:
        ax.set_xlabel(r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s}$  ($m$)',fontsize=10)
    if i!=3:
        pass
        ax.set_yticks(yticks,[])
    else:
        #ax.set_yticks(yticks,yticks_l)
        ax.set_ylabel(ylabel,fontsize=10)
    ax.text(90,ymax*.4,r'$\rho=$'+str(rho)[0:5],fontsize=8,bbox=dict(facecolor='white',alpha=.7,linewidth=0))
    ax.text(10,ymax*.4,r'$N=$'+str([0,3,5,7,10,100][i]),fontsize=8,bbox=dict(facecolor='white',alpha=.7,linewidth=0))
plt.subplots_adjust(hspace=.1,wspace=.1)
plt.savefig('../../plot_output/dke1/les_network_scatter.png', bbox_inches = "tight")


# %%
np.nanmean(err)

# %%
np.nanmean(np.nanstd(rt_100,axis=1))

# %% [markdown]
# # Scratch

# %%
dd=fnc['date'][:]

mm=np.where(dke_1>9)[0]
dd[mm]

# %%
fnc['wind_a_diff'][mm]

# %%
dd=fnc['date'][:]
days=[]
i0=69
for i in range(i0,i0+3):
    days.append(str(int(dd[i])))

days=['20190807','20200828','20210611']
#days=['20180521','20180719','20180629']
h0=7

didx=[]
numvalid=[0,0,0]
profs=[{},{},{}]
dkep=[]
scatloc=[{},{},{}]
colors={'E37':'darkgoldenrod','E41':'teal','E32':'firebrick','E39':'forestgreen','C1':'darkorchid'}
goes=[]
lhet=[]
std=[]
wdir=[]
wdir2=[{},{},{}]

for d in days:
    didx.append(np.where(float(d)==fnc['date'][:].data)[0][0])
i=0
for day in days:
    try:
        fgoes=rasterio.open(g18dir+day+'1600.tif','r')
    except:
        fgoes=rasterio.open(g18dir+day+'1500.tif','r')
    
    dd=fgoes.read(1)
    dd[dd<1]=float('nan')
    goes.append(dd)
    lhet.append(fnc['lst_lhet'][didx[i]])
    std.append(fnc['lst_std'][didx[i]])
    for file in os.listdir(narmdir):
        if day in file:
            numvalid[i]=numvalid[i]+1
            for s in lonlat.keys():
                if s in file:
                    kk=s
                    #print(kk)
            fprof=nc.Dataset(narmdir+file,'r')
            aa=fprof['u'].shape[0]
            if aa==96:
                dx=4
            elif aa==144:
                dx=6
            elif aa>120:
                dx=6
            u=fprof['u'][h0*dx:(h0+1)*dx,0:36].data
            u[u==-9999]=float('nan')
            u=np.nanmean(u,axis=0)
            v=fprof['v'][h0*dx:(h0+1)*dx,0:36].data
            v[v==-9999]=float('nan')
            v=np.nanmean(v,axis=0)
            z=fprof['height'][0:36]
            ug=np.sqrt(u**2+v**2)
            profs[i][kk]=v
            scatloc[i][kk]=fgoes.index(*lonlat[kk])
    wdir.append(circmean(np.deg2rad(fnc['wind_a'][didx[i],4,0:15])))
    dkep.append(fnc['DKE_xy'][didx[i],4,0:35]/fnc['weighting'][didx[i],4,0:35])        
    i=i+1

# %%

# %%
fig=plt.figure(figsize=(5.5,7))
subfigs = fig.subfigures(3, 1, hspace=0,wspace=0,frameon=False,height_ratios=[1.25,1.25,1.1])
ax1=subfigs[1].subplots(1,3)
ax=subfigs[0].subplots(1,3)
grid=ImageGrid(subfigs[2], 111,  # similar to subplot(111)
                nrows_ncols=(1, 3),
                axes_pad=0.1,
                cbar_mode='each',
                cbar_location='bottom',
                cbar_pad=.02,
                cbar_size="5%")
vmax=np.nanpercentile(goes,95)
vmin=296
vmax=310
for i in range(3):
    for k in range(5):
        s=list(colors.keys())[k]
        try:
            ax[i].plot(profs[i][s],z,'-o',color=colors[s],markersize=2,linewidth=.5,label=s)
        except:
            pass
    ax1[i].plot(dkep[i][0:35],z[0:35],'-',color='black')
    if i==2:
        l=ax[i].legend(fontsize=7,loc='lower right',framealpha=.95)
        #l.set_labelsize(8)
    im=grid[i].imshow(goes[i],cmap='coolwarm',vmin=vmin,vmax=vmax)
    #im=grid[i].imshow(goes[i],cmap='coolwarm')
    xx=[]
    yy=[]
    clrs=[]
    for k in scatloc[0].keys():
        xx.append(scatloc[i][k][0])
        yy.append(scatloc[i][k][1])
        clrs.append(matplotlib.colors.to_rgb(colors[k]))
    xx=np.array(xx)
    yy=np.array(yy)
    grid[i].scatter(yy,xx,s=15,c=clrs,edgecolors='black',linewidth=.5,alpha=.95,zorder=5)

    if i==0:
        ax[i].set_yticks([200,400,600,800,1000],[200,400,600,800,1000])
        ax[i].set_ylabel(r'height $(m)$')
        ax1[i].set_yticks([200,400,600,800,1000],[200,400,600,800,1000])
        ax1[i].set_ylabel(r'height $(m)$')
    else:
        ax[i].set_yticks([200,400,600,800,1000],[])
        ax1[i].set_yticks([200,400,600,800,1000],[])
    ax[i].tick_params(axis='both',which='major',labelsize=8)
    ax[i].set_xlabel(r'$\overline{u}$  ($m\ s^{-1}$)',fontsize=8)

    ax1[i].tick_params(axis='both',which='major',labelsize=8)
    ax1[i].set_xlabel(r'$DKE$  ($m^{2}\ s^{-2}$)',fontsize=8)
    #ax[i].grid(True)

    mag=np.nanmean(profs[i]['C1'][0:15])
    print(np.rad2deg(wdir[i]))
    grid[i].arrow(20, 20, mag*np.sin(wdir[i]), -mag*np.cos(wdir[i]), linewidth=2, head_width=0.2, head_length=0.1,color='black',alpha=.6)

    grid[i].set_xticks([])
    grid[i].set_yticks([])

    ax[i].set_title(days[i][0:4]+'-'+days[i][4:6]+'-'+days[i][6:8],fontsize=10)

    tit=r'$\lambda_{T_s}$:'+' '+str(lhet[i])[0:5]+'   '+r'$\sigma_{T_s}$:'+' '+str(std[i])[0:3]
    grid[i].set_title(tit,fontsize=8)
    ax[i].set_ylim(0,1025)
    ax1[i].set_ylim(0,1025)
    cb=grid.cbar_axes[i].colorbar(im,label=r'$T_{s}$')
    grid.cbar_axes[i].tick_params(labelsize=10)
    cb.set_label(label=r'$T_{s}$',size=10)

##### Add figure Labeling
ax[0].text(-.3,1075,'a)',fontsize=8)
ax1[0].text(-.42,1075,'b)',fontsize=8)
grid[0].text(-17,-4,'c)',fontsize=8)

# %%
fnc.close()

# %%

# %%
fnc.variables.keys()

# %%
data=np.abs(fnc['wind_speed_diff'][:]).T
#data[data>190]=360-data[data>190]
#data[:,np.mean(data,axis=0)>75]=float('nan')
plt.imshow(data,aspect=10,origin='lower',cmap='terrain',vmax=30)
plt.colorbar()

# %%
data=np.abs(fnc['wind_a_diff'][:]).T
data[data>190]=360-data[data>190]
#data=np.abs(fnc['wind_a_diff'][:]).T
color=plt.get_cmap('Spectral')(dke_1/5)
plt.scatter(np.linspace(0,272,273),np.nanmean(data,axis=0),color=color)

# %%

# %%

# %%
wdir='/home/tswater/tyche/data/les/LES_wind/'
bdir='/run/media/tswater/Elements/LES/'
day='20160625'
hr=19
hr_='diag_d01_2016-06-25_'+str(hr)+'0000'
hr2='diag_d01_2016-06-25_'+str(hr)+'_wind.nc'
htdir='fr2_'+day+'_00/'
fpu=nc.Dataset(wdir+htdir+hr2,'r')
fpt=nc.Dataset(bdir+htdir+hr_,'r')

# %%
plt.imshow(fpt['AVS_TSK'][0,:,:],cmap='coolwarm')

# %%
plt.imshow(fpu['AVV_V'][0,5,:,:],cmap='coolwarm')

# %%
print(fnc['date'][:])

# %%
plt.scatter(fnt['lst_std_site'][:],fnt['aeri_std'][:,4,2])
plt.xlim(0,2)
plt.ylim(0,2)

# %% [markdown]
# ### Testing Windtype, Precip, Pgrad
#

# %%
fnc.close()
fnc=nc.Dataset(troot+'lidar_lst_out2.nc','r')

dke=fnc['DKE_z'][:]+fnc['DKE_xy'][:]
mke=fnc['MKE_z'][:]+fnc['MKE_xy'][:]
wfac=np.nanmean(np.sum((fnc['weighting'][:,:,:])[:,2:8,0:h],axis=2),axis=1)

_mke=1/(np.nanmean(np.sum((mke)[:,2:8,0:h],axis=2),axis=1)/wfac)
dke_1=np.nanmean(np.sum((dke/1)[:,2:8,0:h],axis=2),axis=1)/wfac
rat=_mke*dke_1

ws=np.mean(fnc['wind_speed'][:,2:8,0:5],axis=(1,2)) #first 200m
ws2=np.mean(fnc['wind_speed'][:,2:8,-2],axis=(1)) #1000m
winda=np.rad2deg(circmean(np.deg2rad(fnc['wind_a'][:,2:8,0:5]),axis=(1,2),nan_policy='omit'))
lsta=fnc['lst_a'][:]
abet=angle_diff(winda,lsta)
windp=np.cos(abet)*ws
repo=np.sum(fnc['sites_repo'][:,2:8,0:h,:],axis=3)
repo=np.mean(repo,axis=(1,2))
vort=fnc['vort'][:]
lhet=fnc['lst_std'][:]/fnc['lst_mean'][:]*fnc['lst_lhet'][:]
rain=np.nanmean(fnc['precip'][:,2:8],axis=1)
rain[rain>0]=2
rain[np.isnan(rain)]=1
pgrad=fnc['pgrad'][:]

fnc=nc.Dataset(troot+'lidar_lst_out.nc','r')

# %%
Nr=4
Nw=20
Nv=20
Na=20
Ns=2


corr=np.zeros((3,Nr*Nw*Nv*Na*Ns))
count=np.zeros((3,Nr*Nw*Nv*Na*Ns))
al_=np.linspace(0,85,Na)
vl_=np.linspace(8,35,Nv-1)
vl_=np.append(vl_,50)
vl_=vl_*10**(-5)
wl_=np.linspace(5,15,Nw-2)
wl_=np.append(wl_,[17.5,20])
rl_=np.array([2,3,4,5])
sl_=[0,2]
for y in range(3):
    dat=[rat,_mke,dke_1][y]
    als=[]
    vls=[]
    wls=[]
    rls=[]
    sls=[]
    i=0
    for r in range(Nr):
        print('.',end='',flush=True)
        rl=rl_[r]
        for s in range(Ns):
            sl=sl_[s]
            for w in range(Nw):
                wl=wl_[w]
                for v in range(Nv):
                    vl=vl_[v]
                    for a in range(Na):
                        al=al_[a]
                        m=(repo>=rl)
                        m=m&(ws2<wl)
                        m=m&(abet>al)
                        m=m&(vort<vl)
                        m=m&(rain<=sl)
                        xx=lhet[m]
                        yy=dat[m]
                        count[y,i]=np.sum(m)
                        corr[y,i]=spearmanr(xx,yy,nan_policy='omit')[0]
                        als.append(al)
                        vls.append(vl)
                        wls.append(wl)
                        rls.append(rl)
                        sls.append(sl)
    
                        i=i+1

# %%
20*20*20*4

# %%
10*10*10*4*10*3

# %%
vs=[np.array(rls),np.array(wls),np.array(vls),np.array(als)]
fig,axs=plt.subplots(3,5,figsize=(6,3))
for v in range(5):
    l_=[sl_,rl_,wl_,vl_,al_][v]
    ls=[sls,rls,wls,vls,als][v]
    ls=np.array(ls)
    flip=[True,False,True,True,False][v]
    xticks=[[0,2],[2,3,4,5],[20,15,10,5],[.0005,.0004,.0003,.0002,.0001],[0,20,40,60,80]][v]
    xticksl=[['0.0','>0.0'],[2,3,4,5],[20,15,10,5],[r'$5e-4$','','','',r'$1e-4$'],[0,20,40,60,80]][v]
    yticks=[.1,.2,.3,.4,.5]
    yticksl=['0.1','','0.3','','0.5']
    xlabel=[r'Precip. ($mm$)','MIN sites\nreporting',r'MAX $u_g$ ($ms^{-1}$)',r'MAX $|\zeta|$',r'MIN $\alpha$'][v]
    for y in range(3):
        ylabel=[r'$\rho_{DKE/MKE}$',r'$\rho_{1/MKE}$',r'$\rho_{DKE}$'][y]
        ax=axs[y,v]
        color=plt.get_cmap('nipy_spectral')(count[y,:]/273)
        ylo=[]
        yhi=[]
        ym=[]
        for j in range([Ns,Nr,Nw,Nv,Na][v]):
            m=(ls==l_[j])&(count[y,:]>10)
            if np.sum(m)<200:
                ym.append(float('nan'))
                ylo.append(float('nan'))
                yhi.append(float('nan'))
            else:
                ylo.append(np.nanpercentile(corr[y,m],25))
                yhi.append(np.nanpercentile(corr[y,m],75))
                ym.append(np.nanpercentile(corr[y,m],50))
        ym=np.array(ym)
        ylo=np.array(ylo)
        yhi=np.array(yhi)
        ax.plot(l_,ym,'-o',markersize=2,linewidth=.75,color='darkgreen')
        ax.fill_between(l_,ylo,yhi,alpha=.5,color='darkgreen')
        if flip:
            ax.invert_xaxis()
        ax.set_ylim(.075,.6)
        ax.grid(True,color='black',linewidth=.2)
        if y<2:
            ax.set_xticks(xticks,[])
        else:
            ax.set_xticks(xticks,xticksl,fontsize=8)
            ax.set_xlabel(xlabel,fontsize=10)
        if v==0:
            ax.set_ylabel(ylabel)
            ax.set_yticks(yticks,yticksl,fontsize=8)
        else:
            ax.set_yticks(yticks,[])
        #ax.scatter(vs[v],corr[y,:],s=5,color=color)
plt.savefig('../../plot_output/dke1/filter_sensitivity_precip.png', bbox_inches = "tight")

# %%
(10**4)**(1/3)

# %% [markdown]
# ### More Scratch

# %%
fnt['aeri_std'][:].shape

# %%
fnc['height'][0:10]

# %%
h=16
dke=fnt['DKE_z'][:]+fnt['DKE_xy'][:]
mke=fnt['MKE_z'][:]+fnt['MKE_xy'][:]

wfac=np.nanmean(np.sum((fnt['weighting'][:,:,:])[:,2:8,0:h],axis=2),axis=1)

_mke=1/(np.nanmean(np.sum((mke)[:,2:8,0:h],axis=2),axis=1)/wfac)
dke_1=np.nanmean(np.sum((dke/1)[:,2:8,0:h],axis=2),axis=1)/wfac
rat=_mke*dke_1

ws=np.mean(fnt['wind_speed'][:,2:8,0:5],axis=(1,2)) #first 200m
winda=np.rad2deg(circmean(np.deg2rad(fnt['wind_a'][:,2:8,0:5]),axis=(1,2),nan_policy='omit'))
lsta=fnt['lst_a'][:]
abet=angle_diff(winda,lsta)
windp=np.cos(abet)*ws
repo=np.sum(fnt['sites_repo'][:,2:8,0:h,:],axis=3)
repo=np.mean(repo,axis=(1,2))
vort=fnt['vort'][:]
lhet=fnt['lst_lhet'][:]
cv=fnt['lst_std'][:]/fnt['lst_mean'][:]
cv_s=fnt['lst_std_site'][:]/fnt['lst_mean'][:]
astd=np.nanmean(fnt['aeri_std'][:,4,0:4],axis=1)
cv_a=astd/fnt['lst_mean'][:]
het=lhet*cv
m=(ws<15)&(abet>20)&(repo>=3)&(vort<4*10**(-4))&(astd<2)

color=plt.get_cmap('Spectral')(ws/15)

corr=np.zeros((2,5))
#corr2=np.zeros((2,4))
xx=np.ones((2,5,70))*float('nan')
yy=np.ones((2,5,70))*float('nan')

for v in range(2):
    for j in range(5):
        xx[v,j,m]=[cv,cv_s,cv_a,lhet,het][j][m]
        yy[v,j,m]=[rat,dke_1][v][m]
        corr[v,j]=spearmanr(xx[v,j],yy[v,j],nan_policy='omit')[0]

# %%
fig,axs=plt.subplots(2,5,figsize=(6,3),width_ratios=[1,1,1,1,1.25])
for i in range(2):
    for j in range(5):
        ax=axs[i,j]
        ax.scatter(xx[i,j],yy[i,j],s=3,c=color)

        # logscale
        ax.semilogy()
        ax.grid(True,color='black',linewidth=.2)
        ax.set_title(r'$\rho_s=$'+str(np.round(corr[i,j],3)))
        
        if j>0:
            ax.set_yticklabels([],minor=False)
        else:
            ax.set_ylabel([r'$DKE/MKE$',r'$DKE$ ($kg\ s^{-2}$)'][i])
        if i==0:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(ax.get_xticks(),rotation=45)
            ax.set_xlabel([r'$CV_{full}$',r'$CV_{site}$',r'$CV_{aeri}$',r'$\lambda_{T_s}$ ($m$)',r'$\lambda_{T_s}\sigma_{T_s}/\overline{T_s}$  ($m$)'][j])
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 15), cmap='Spectral'),
             ax=axs[0,4], orientation='vertical', label=r'$u_g$ ($ms^{-1}$)')
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 15), cmap='Spectral'),
             ax=axs[1,4], orientation='vertical', label=r'$u_g$ ($ms^{-1}$)')
fig.subplots_adjust(hspace=.3,wspace=.1)

plt.savefig('../../plot_output/dke1/het_type_sensitivity_aeri.png', bbox_inches = "tight")

# %%

# %%
