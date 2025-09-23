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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc
import pickle
import numpy as np
matplotlib.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %% [markdown]
# # Trying to figure out pickle files 

# %%
with open('/home/tsw35/soteria/dke_lidar/pickle_network/20150606dke.pkl', 'rb') as file:
    p20150606dke = pickle.load(file)
with open('/home/tsw35/soteria/dke_lidar/pickle_network/20150606.pkl', 'rb') as file:
    p20150606 = pickle.load(file)

# %%
dkem=[]
dkes=[]
for k in p20150606dke.keys():
    dkem.append(np.nanmedian(p20150606dke[k]))
    dkes.append(np.nanstd(p20150606dke[k]))

# %%
with open('/home/tsw35/soteria/dke_lidar/pickle_network/base.pkl', 'rb') as file:
    base = pickle.load(file)
with open('/home/tsw35/soteria/dke_lidar/pickle_network/corrs.pkl', 'rb') as file:
    corrs = pickle.load(file)
with open('/home/tsw35/soteria/dke_lidar/pickle_network/dkebase.pkl', 'rb') as file:
    dkebase = pickle.load(file)
with open('/home/tsw35/soteria/dke_lidar/pickle_network/hets.pkl', 'rb') as file:
    hets = pickle.load(file)
with open('/home/tsw35/soteria/dke_lidar/pickle_network/nets.pkl', 'rb') as file:
    nets = pickle.load(file)

# %%
ht=[]
dk=[]
mk=[]
rt=[]
for k in hets.keys():
    ht.append(hets[k])
    dk.append(dkebase[k])
    a=dkebase[k]/base[k]
    mk.append(a)
    rt.append(base[k])
rt=np.array(rt)
dk=np.array(dk)
mk=np.array(mk)
ht=np.array(ht)

# %%
color=plt.get_cmap('terrain')((np.array(mk)-900)/50000)
rt2=np.sqrt(dk)/np.sqrt(mk)
plt.scatter(ht,rt2,c=color)
#plt.semilogy([10,.20],[.01,.02],alpha=0)
stats.spearmanr(ht,rt2)
#plt.ylim(0,0.04)

# %%

m=mk<5000

color=plt.get_cmap('terrain')((np.array(mk)-900)/50000)
color=color[m]
rt2=np.sqrt(dk)/np.sqrt(mk)
plt.scatter(ht[m],rt2[m],c=color)
#plt.semilogy([10,.20],[.01,.02],alpha=0)
print(stats.spearmanr(ht[m],rt2[m]))
#plt.ylim(0,0.04)
plt.xlim(0,120)
plt.ylim(0,.3)

# %%

m=(mk>5000)&(mk<12500)

color=plt.get_cmap('terrain')((np.array(mk)-900)/50000)
color=color[m]
rt2=np.sqrt(dk)/np.sqrt(mk)
plt.scatter(ht[m],rt2[m],c=color)
#plt.semilogy([10,.20],[.01,.02],alpha=0)
print(stats.spearmanr(ht[m],rt2[m]))
plt.xlim(0,120)
plt.ylim(0,.3)

# %%

m=(mk>12500)

color=plt.get_cmap('terrain')((np.array(mk)-900)/50000)
color=color[m]
rt2=np.sqrt(dk)/np.sqrt(mk)
plt.scatter(ht[m],rt2[m],c=color)
#plt.semilogy([10,.20],[.01,.02],alpha=0)
print(stats.spearmanr(ht[m],rt2[m]))
plt.xlim(0,120)
plt.ylim(0,.3)

# %%

# %%

# %%

# %%
from scipy import stats
stats.spearmanr(ht,dk)

# %%
np.max(mk)

# %%
plt.hist(mk)

# %%
plt.hist(dk)

# %%

# %%
