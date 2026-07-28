import netCDF4 as nc
import numpy as np
import os

#### USER INPUT ####
les_days = [20180709]
les_dir  = '/run/media/tswater/Elements/LES/'
out_dir  = '/home/tswater/tyche/data/les/test_compress/'
avv_vars = ['AVV_U','AVV_W','AVV_U','AVV_THV','AVV_QV','AVV_LWC','AVV_RHO']
debug    = False

#### SCRIPT ####
# iterate over each LES day
for d in les_days:
    # iterate over heterogeneous and homogeneous
    d=str(d)
    print(d,flush=True)
    for htg in [0,1]:
        print('   ',end='')
        folder_in = 'fr2_'+d+'_0'+str(htg)+'/'
        try:
            os.mkdir(out_dir+'/'+folder_in)
        except Exception as e:
            print(e)
        for file in os.listdir(les_dir+folder_in):
            print('.',end='',flush=True)
            # load input, initialize output
            fpi=nc.Dataset(les_dir+folder_in+file,'r')
            fpo=nc.Dataset(out_dir+folder_in+file[0:22]+'_compress.nc','w')

            # add dimensions
            fpo.createDimension('west_east',400)
            fpo.createDimension('south_north',400)
            fpo.createDimension('bottom_top',226)
            fpo.createDimension('Time',1)

            # add variables
            for k in fpi.variables.keys():
                if debug:
                    print(k)
                if ('AVV' in k) and (k not in avv_vars):
                    continue
                elif k=='Times':
                    continue
                else:
                    dims=fpi[k].dimensions
                    fpo.createVariable(k,'f4',dims)
                    if ('west_east' in dims) and ('bottom_top' in dims):
                        fpo[k][:]=fpi[k][:,:,60:-60,60:-60]
                    elif ('west_east' in dims):
                        fpo[k][:]=fpi[k][:,60:-60,60:-60]
                    else:
                        fpo[k][:]=fpi[k][:]
                    fpo[k].units=fpi[k].units
                    fpo[k].description=fpi[k].description
            fpo.close()
            fpi.close()
        print()





