from subprocess import run
import os

flist=os.listdir('./')
flist.sort()
for file in flist:
    if '.ipynb' in file:
        # check if a checkpoint
        if 'checkpoint' in file:
            continue

        # check if jupytext analog exists, if so, sync
        if (file[0:-5]+'py') in flist:
            cmd='jupytext --sync '+file[0:-5]+'py'
        else:
            cmd='jupytext --set-formats ipynb,py:percent '+file
        run(cmd,shell=True)
