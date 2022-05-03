# script to calculate correlation matrices based on time series
# paths need to be set

import os
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob
import os.path
import scipy.io
from matplotlib import pyplot as plt
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import settings as s

outpath=s.HCP_derivatives+'correlation/' # change to s.Epi_derivatives when running for epilepsy data
inpath=s.HCP_derivatives+'/time_series/'

sub=os.listdir(inpath)
subjects=[]

for i in range(len(sub)):
    if os.path.isdir(inpath+sub[i]) and 'sub' in sub[i]:
        subjects.append(sub[i])
subjects=sorted(subjects)

for i in range(len(subjects)):
    print(subjects[i])
    outsub=outpath+subjects[i]+'/'
    insub=inpath+subjects[i]+'/'
    if os.path.exists(outsub)==False:
       os.mkdir(outsub)

    runs100=glob(insub+subjects[i]+'_*100_*.mat')
    for j in range(len(runs100)):
        filename=os.path.splitext(os.path.split(runs100[j])[1])[0]
        data=scipy.io.loadmat(runs100[j])
        corr=np.corrcoef(np.transpose(data['ts']))
        pos=np.unique(corr[corr>0])
        indthresh=int(len(pos)*0.5) # set threshold 0.5, 0.9, all positive (corr[corr<0]=0)
        thresh=pos[indthresh]
        corr[corr<thresh]=0
        Adj=corr
        Adj=Adj.astype(float)
        mydict={'Adj':Adj}
        scipy.io.savemat(outsub+filename+'_corr_full_top50.mat',mydict) # indicate chosen threshold in file name
        plt.matshow(Adj)
        plt.savefig(outsub+filename+'_corr_full_top50.png', vmin=0,vmax=1) # indicate chosen threshold in file name
        plt.close()

