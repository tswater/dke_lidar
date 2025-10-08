import pickle
import numpy as np
import netCDF4 as nc
import os

lesdir ='/home/tsw35/xSot_shared/LES_3D/clasp/fr2/'
outfile='/home/tsw35/soteria/dke_lidar/pickle_tsw/dke_mke_tsw.pkl'
netfile='/home/tsw35/soteria/dke_lidar/pickle_tsw/nets.pkl'

daylist=[]
daylist.sort()

for day in os.listdir(lesdir):
    if '_01' in day:
        continue
    else:
        daylist.append(day)

out={}

def get_les_dke(u,v,w,z,rho):
    dke=np.zeros((226,))
    mke=np.zeros((226,))
    weight=np.zeros((226,))
    for k in range(226):
        mke[k]=(np.nanmean(u[0,k,:,:])**2+np.nanmean(v[0,k,:,:])**2+np.nanmean(w[0,k,:,:])**2)*0.5
        u2=u[0,k,:,:]-np.nanmean(u[0,k,:,:])
        v2=v[0,k,:,:]-np.nanmean(v[0,k,:,:])
        w2=w[0,k,:,:]-np.nanmean(w[0,k,:,:])
        dke[k]=0.5*(np.nanvar(u2)+np.nanvar(v2)+np.nanvar(w2))
    weight[1:]=z[1:]-z[0:-1]
    weight[0]=z[0]
    idx=int(np.where(z>500)[0][0])
    return dke,mke,weight,idx

out={}

nets=pickle.load(open(netfile,'rb'))
netlist=list(nets.keys())

for day in daylist:
    print('Asessing '+day,flush=True)
    out[day[4:12]]={}
    for j in range(len(netlist)):
        out[day[4:12]][netlist[j]]={'dke':float('nan'),'mke':float('nan'),\
                                    'dkes':np.zeros((100,)),'mkes':np.zeros((100,))}
    dke_=np.zeros((6,226))
    mke_=np.zeros((6,226))
    weight_=np.zeros((6,226))
    idx_=np.zeros((6,))
    yr=day[4:8]
    mn=day[8:10]
    dy=day[10:12]

    i=0
    for hour in [15,16,17,18,19,20]:
        print('   '+str(hour)+':',end='',flush=True)
        fp=nc.Dataset(lesdir+day+'/diag_d01_'+yr+'-'+mn+'-'+dy+'_'+str(hour)+':00:00','r')
        u=fp['AVV_U'][:]
        v=fp['AVV_V'][:]
        w=fp['AVV_W'][:]
        z=fp['AVP_Z'][0,:]
        rho=fp['AVP_RHO'][0,:]
        dke_[i,:],mke_[i,:],weight_[i,:],idx_[i]=get_les_dke(u,v,w,z,rho)
        print('*',end='',flush=True)
        for j in range(len(netlist)):
            nsz=netlist[j]
            iters=nets[nsz]
            for it in range(len(iters)):
                tows=nets[nsz][it]
                uu=np.zeros((nsz,226))
                vv=np.zeros((nsz,226))
                ww=np.zeros((nsz,226))
                tt=0
                for tow in range(len(tows)):
                    uu[tt,:]=u[0,:,tows[tow][0],tows[tow][1]]
                    vv[tt,:]=v[0,:,tows[tow][0],tows[tow][1]]
                    ww[tt,:]=w[0,:,tows[tow][0],tows[tow][1]]
                    tt=tt+1
                mke=0.5*(np.mean(uu,axis=0)**2+np.mean(vv,axis=0)**2+np.mean(ww,axis=0)**2)
                dke=0.5*(np.var(uu,axis=0,ddof=1)+np.var(vv,axis=0,ddof=1)+np.var(ww,axis=0,ddof=1))

                mke=np.sum((weight_[i,:]*mke)[0:int(idx_[i])])/np.sum(weight_[i,:][0:int(idx_[i])])
                dke=np.sum((weight_[i,:]*dke)[0:int(idx_[i])])/np.sum(weight_[i,:][0:int(idx_[i])])

                # add 1/6th to get mean over the 6 hours
                out[day[4:12]][nsz]['dkes'][it]=out[day[4:12]][nsz]['dkes'][it]/6+dke/6
                out[day[4:12]][nsz]['mkes'][it]=out[day[4:12]][nsz]['mkes'][it]/6+mke/6

            print('.',end='',flush=True)
            out[day[4:12]][nsz]['dke']=np.nanmedian(out[day[4:12]][nsz]['dkes'])
            out[day[4:12]][nsz]['mke']=np.nanmedian(out[day[4:12]][nsz]['mkes'])

        fp.close()
        i=i+1
        print()
    dke_=(weight_*dke_)
    mke_=(weight_*mke_)
    dke=[]
    mke=[]
    idx=int(np.nanmean(idx_))
    for i in range(6):
        dke.append(np.sum(dke_[i,0:idx])/np.sum(weight_[i,0:idx]))
        mke.append(np.sum(mke_[i,0:idx])/np.sum(weight_[i,0:idx]))
    out[day[4:12]][0]={'dke':np.nanmean(dke),'mke':np.nanmean(mke)}

pickle.dump(out,open(outfile,'wb'))

