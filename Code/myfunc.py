from __future__ import division
from __future__ import print_function

import time
import os
import sys
from glob import glob
import pickle as pkl

#import tensorflow as tf
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import scipy.sparse as sp
import scipy.stats as stats
import scipy.io as io

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from sklearn import manifold
from scipy.special import expit

from optimizer import OptimizerSiemens
from input_data import *
from model import *
from preprocessing import *
from myfunc import *
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import settings as s

def load_myData(GraphType, ROInr, withGSR, withSurfaceMeasures, runID,corr_mode=None, keep_features=None):
    #root = '/home/cir/bburger/PHD/Epilepsy_SubjectData/Preprocessing/derivatives/connectome/'
    root=s.Epi_derivatives
    graph = root + GraphType + '/'
    Features = root + 'time_series/'
    surface_measures = root + 'SurfaceMeasures/'

    if withGSR == True:
        GSR_str = 'GSR_'
    else:
        GSR_str = '_'
    if corr_mode=='full':
        adj_type='corr_full'
    if corr_mode=='full_top90':
        adj_type='corr_full_top90'
    if corr_mode=='full_top50':
        adj_type='corr_full_top50'
    sub = os.listdir(graph)
    subjects = []
    for i in range(len(sub)):
        if os.path.isdir(graph + sub[i]) and 'sub' in sub[i]:
            subjects.append(sub[i])
    subjects = sorted(subjects)

    Adj = []
    features = []
    for i in range(len(subjects)):
        data = sorted(glob(graph + subjects[i] + '/' + subjects[i] + '_*' + ROInr + '_*aCompCor' + GSR_str +adj_type +'.mat'))

        if runID == 'run1':# or len(data) == 1:
            run = io.loadmat(data[0])  # choose run
        if runID == 'run2' and len(data) > 1:
            run = io.loadmat(data[1])  # choose run
        if runID == 'run2'and len(data) < 2:
            run=[]
        if run != []:
            Adj.append(sp.csr_matrix(run['Adj']))
        else:
            Adj.append(sp.csr_matrix(run,shape=(1,1)))
        data = sorted(
            glob(Features + subjects[i] + '/' + subjects[i] + '_*' + ROInr + '_*aCompCor' + GSR_str[:-1] + '.mat'))

        if runID == 'run1': #or len(data) == 1:
            run = io.loadmat(data[0])  # chose run
        if runID == 'run2' and len(data) > 1:
            run = io.loadmat(data[1])  # chose run
        if runID == 'run2'and len(data) < 2:
            run=[]
        if run !=[]:
            run = np.transpose(run['ts'])
        if run !=[] and run.shape[1] < 100:
            tmp = np.copy(run)
            run = np.zeros((tmp.shape[0], 100))
            run[:, 0:tmp.shape[1]] = tmp
            run[:, tmp.shape[1]:] = tmp[:, 0:(100 - tmp.shape[1])]
        if keep_features is not None and run !=[]:
            keep = np.ones((run.shape[1],), dtype=bool)
            keep[keep_features] = False
            run[keep, :] = 0
        if withSurfaceMeasures and ROInr != '1000' and run !=[]:
            measures = io.loadmat(surface_measures + subjects[i][0:6] + '1/surf_measures' + ROInr + '.mat')[
                'surf_measures']
            run = np.concatenate([run, measures], axis=1)

        features.append(sp.csr_matrix(run))

    A_orig = [sparse_to_tuple(A) for A in Adj]
    A = [preprocess_graph(A - sp.eye(A.shape[0])) for A in Adj]
    X = [sparse_to_tuple(X.tocoo()) for X in features]
    return A_orig, A, X




def load_myDataHCP(GraphType, ROInr, corr_mode=None, keep_features=None):
    #root = '/project/neuro/Twin_Study_Bianca/DataForEpiStudy/'
    root=s.HCP_derivatives
    graph = root + GraphType + '/'
    Features = root + 'time_series/'
    fileID = '*LRRLconcat*'
    if GraphType == 'correlation' and corr_mode =='full':
        fileID = '*LRRLconcat_corr_full'
    if GraphType == 'correlation' and corr_mode =='full_top90':
        fileID = '*LRRLconcat_corr_full_top90'
    if GraphType == 'correlation' and corr_mode =='full_top50':
        fileID = '*LRRLconcat_corr_full_top50'
    sub = os.listdir(graph)
    subjects = []
    for i in range(len(sub)):
        if os.path.isdir(graph + sub[i]) and 'sub' in sub[i]:
            subjects.append(sub[i])
    subjects = sorted(subjects)

    Adj = []
    features = []
    for i in range(len(subjects)):

        data = sorted(glob(graph + subjects[i] + '/' + subjects[i] + '_*' + ROInr + '_' + fileID + '.mat'))
        run = io.loadmat(data[0])
        Adj.append(sp.csr_matrix(run['Adj']))
        data = sorted(glob(Features + subjects[i] + '/' + subjects[i] + '_*' + ROInr + '_*LRRLconcat.mat'))
        run = io.loadmat(data[0])
        run = np.transpose(run['ts'])
        run = np.concatenate([run[:, 0:50], run[:, 1200:1250]], axis=1)  # set volumes used, orgigial 0:50 and 1200:1250
        if keep_features is not None:
            keep = np.ones((run.shape[1],), dtype=bool)
            keep[keep_features] = False
            run[keep, :] = 0
        features.append(sp.csr_matrix(run))

    A_orig = [sparse_to_tuple(A) for A in Adj]
    A = [preprocess_graph(A - sp.eye(A.shape[0])) for A in Adj]
    X = [sparse_to_tuple(X.tocoo()) for X in features]
    return A_orig, A, X



