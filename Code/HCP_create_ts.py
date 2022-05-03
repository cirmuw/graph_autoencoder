# file to extract parcel mean time series from HCP data,
# path to HCP data needs to be set
# input are time series already mapped to fsaverage4 surface of Freesurfer


import os
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob
import os.path
from nilearn.input_data import NiftiLabelsMasker #v0.6.2
from matplotlib import pyplot as plt
import scipy.io
import sklearn.preprocessing as sk
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import settings as s

infile ='/path/to/HCP_subject_info_restricted.xlsx' #file containing also restricted information like zygosity, twin status
subjectpath1='/root/folder/of/HCP/data/HCP_3T_RESTA_fmri/'#- path to HCP data
subjectpath2='/root/folder/of/HCP/data/HCP_3T_RESTB_fmri/'#/
atlas1l='/../Deliveries/lh.Schaefer2018_100Parcels_7Networks_order.annot'
atlas1r='/../Deliveries/rh.Schaefer2018_100Parcels_7Networks_order.annot'
target=s.HCP_derivatives+'time_series'
zerovertexlh=np.load('../Deliveries/0verticeslh.npy')
zerovertexrh=np.load('../Deliveries/0verticeslh.npy')

lhannot1=nib.freesurfer.io.read_annot(atlas1l)
lhlabels1=lhannot1[0]
rhannot1=nib.freesurfer.io.read_annot(atlas1r)
rhlabels1=rhannot1[0]
labelslh1=np.delete(lhlabels1,zerovertexlh,0)
labelsrh1=np.delete(rhlabels1,zerovertexrh,0)



#Get paths to mgh-files of available subjects, only unrelated subjects are used
xl=pd.ExcelFile(infile)
dataframe1=xl.parse('Sheet1')
isTwin=dataframe1['Twin_Stat']=='Twin'
isTwin=np.where(isTwin)[0]
dataframe2=dataframe1.drop(isTwin,0)
Subjects=dataframe2['Subject'].values

path1=[]
path2=[]
for i in range(Subjects.shape[0]):
    path1.append(subjectpath1+str(Subjects[i]))
    path2.append(subjectpath2+str(Subjects[i]))

fmri_LH_LR_R1=[]
fmri_RH_LR_R1=[]
fmri_LH_RL_R1=[]
fmri_RH_RL_R1=[]
fmri_LH_LR_R2=[]
fmri_RH_LR_R2=[]
fmri_LH_RL_R2=[]
fmri_RH_RL_R2=[]
truesubjects=[]

for i in range(Subjects.shape[0]):
    if os.path.isdir(path1[i])==True:
        fmri_LH_LR_R1.append(path1[i]+'/lh.rfMRI_REST1_LR_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        fmri_RH_LR_R1.append(path1[i]+'/rh.rfMRI_REST1_LR_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        fmri_LH_RL_R1.append(path1[i]+'/lh.rfMRI_REST1_RL_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        fmri_RH_RL_R1.append(path1[i]+'/rh.rfMRI_REST1_RL_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        #fmri_LH_LR_R2.append(path1[i]+'/lh.rfMRI_REST2_LR_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        #fmri_RH_LR_R2.append(path1[i]+'/rh.rfMRI_REST2_LR_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        #fmri_LH_RL_R2.append(path1[i]+'/lh.rfMRI_REST2_RL_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        #fmri_RH_RL_R2.append(path1[i]+'/rh.rfMRI_REST2_RL_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        truesubjects.append(Subjects[i])
    if os.path.isdir(path2[i])==True:
        fmri_LH_LR_R1.append(path2[i]+'/lh.rfMRI_REST1_LR_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        fmri_RH_LR_R1.append(path2[i]+'/rh.rfMRI_REST1_LR_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        fmri_LH_RL_R1.append(path2[i]+'/lh.rfMRI_REST1_RL_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        fmri_RH_RL_R1.append(path2[i]+'/rh.rfMRI_REST1_RL_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        #fmri_LH_LR_R2.append(path2[i]+'/lh.rfMRI_REST2_LR_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        #fmri_RH_LR_R2.append(path2[i]+'/rh.rfMRI_REST2_LR_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        #fmri_LH_RL_R2.append(path2[i]+'/lh.rfMRI_REST2_RL_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        #fmri_RH_RL_R2.append(path2[i]+'/rh.rfMRI_REST2_RL_Atlas_hp2000_clean_bpss_gsr_fs4.mgh')
        truesubjects.append(Subjects[i])

# extract timeseries
for i in range(len(truesubjects)):
       outsub=target+'sub-'+str(truesubjects[i])+'/'
       if os.path.exists(outsub)==False:
          os.mkdir(outsub)
      
       lh1=nib.load(fmri_LH_LR_R1[i]).get_data()
       lh2=nib.load(fmri_LH_RL_R1[i]).get_data()
       rh1=nib.load(fmri_RH_LR_R1[i]).get_data()
       rh2=nib.load(fmri_RH_RL_R1[i]).get_data()
       lh1.resize(lh1.shape[1],lh1.shape[3])
       lh2.resize(lh2.shape[1],lh2.shape[3])
       rh1.resize(rh1.shape[1],rh1.shape[3])
       rh2.resize(rh2.shape[1],rh2.shape[3])
       lh=np.concatenate((lh1,lh2),axis=1)
       rh=np.concatenate((rh1,rh2),axis=1)

       lhframe=pd.DataFrame(np.transpose(np.delete(lh, zerovertexlh,0)))
       lhframe=lhframe.groupby(labelslh1,axis=1).mean()
       lhframe=np.array(lhframe,dtype='float64')[:,1:]     
       rhframe=pd.DataFrame(np.transpose(np.delete(rh,zerovertexrh,0)))
       rhframe=rhframe.groupby(labelsrh1,axis=1).mean()
       rhframe=np.array(rhframe,dtype='float64')[:,1:]
       ts=np.concatenate([lhframe,rhframe],axis=1)
       ts=sk.StandardScaler().fit_transform(ts)
       #print(ts.shape)
       dict100={'ts':ts}
       scipy.io.savemat(outsub+'sub-'+str(truesubjects[i])+'_run-01_Schaefer100_7NetsOrder_LRRLconcat.mat',dict100)
     
