#code to get individual embeddings after auto_encoder training
# input: params to set directory to trained model
        #choice HCP data or epilepsy data

from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import numpy as np

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import scipy.sparse as sp
import scipy.stats as stats

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from sklearn import manifold
from scipy.special import expit

from optimizer import OptimizerSiemens
from input_data import *
from model import *
from preprocessing import *
from myfunc import  load_myData, load_myDataHCP

# settings to load data,
data_mode= 'HCP' #'epi', 'HCP',
conn_density='full_top90'
seed=1
# Load data
if data_mode=='HCP':
   adj_post_orig, adj_pre, features_pre = load_myDataHCP('correlation', '100',conn_density) #set data loader params

if data_mode=='epi':
    adj_orig, adj, features = load_myData('correlation', '100', True, False, 'run1',conn_density)#set data loader params
    pre_ind = list(range(0, 56, 2))
    del pre_ind[16:18]
    post_ind = list(range(1, 56, 2))
    del post_ind[16:18]


    adj_pre_orig = [adj_orig[ind] for ind in pre_ind]
    adj_pre = [adj[ind] for ind in pre_ind]
    features_pre = [features[ind] for ind in pre_ind]
    adj_post_orig = [adj_orig[ind] for ind in post_ind]
    adj_post = [adj[ind] for ind in post_ind]
    features_post = [features[ind] for ind in post_ind]



# Addional Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
if data_mode=='HCP':
   flags.DEFINE_integer('epochs', int(np.floor(len(adj_pre)*0.8))*25, 'Number of epochs to train.')
if data_mode=='epi' or data_mode=='mixed':
    flags.DEFINE_integer('epochs', 26,'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 8, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 4.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('edge_dropout', 0., 'Dropout for individual edges in training graph')
flags.DEFINE_float('autoregressive_scalar', 0., 'Scalar for Graphites')
flags.DEFINE_integer('vae', 1, '1 for variational objective')

flags.DEFINE_integer('subsample', 0, 'Subsample in optimizer')
flags.DEFINE_float('subsample_frac', 1, 'Ratio of sampled non-edges to edges if using subsampling')

flags.DEFINE_integer('verbose', 1, 'verboseness')
flags.DEFINE_integer('test_count', 10, 'batch of tests')

flags.DEFINE_integer('gpu', -1, 'Which gpu to use')
flags.DEFINE_integer('seeded', 1, 'Set numpy random seed')
np.set_printoptions(suppress=True, precision=3)

if FLAGS.seeded:
    np.random.seed(1)

tf.compat.v1.disable_eager_execution()

# restore model
placeholders = {
    'features': tf.compat.v1.sparse_placeholder(tf.float32),
    'adj_pre': tf.compat.v1.sparse_placeholder(tf.float32),
    'adj_post_orig': tf.compat.v1.sparse_placeholder(tf.float32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
}
num_features = 100
keep_features_for_rois = None
features_nonzero = 10000
num_nodes = 100
model = myModel(placeholders, num_features, num_nodes, features_nonzero, keep_features_for_rois)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if FLAGS.gpu == -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.compat.v1.Session()
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)  # Or whichever device you would like to use
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

saver = tf.compat.v1.train.Saver()
latest_checkpoint = tf.compat.v1.train.latest_checkpoint('../TrainedModels/'+conn_density+'/checkpoint_pretrainOptimalepochs_shuffle_mse_corrregion_corr_seed'+str(seed))
saver.restore(sess, latest_checkpoint)

# get embeddings
embeddings = []
recon=[]
if os.path.exists('../Embeddings/')==False:
    os.mkdir('../Embeddings/')
if os.path.exists('../Embeddings/'+conn_density+'/')==False:
    os.mkdir('../Embeddings/'+conn_density+'/')
for index in range(len(adj_pre)):
    if adj_pre[index][2][0]>1:
        feed_dict = construct_feed_dict(adj_pre[index], adj_post_orig[index], features_pre[index], placeholders)########
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        out = sess.run([model.reconstructions_noiseless_full,model.get_z(random=False)], feed_dict)
        embeddings.append(out[1])
        recon.append(out[0])
    else:
        embeddings.append(np.empty((num_nodes,8,)))
        recon.append(np.empty((num_nodes,num_nodes,)))
    if index==179 or index==180:#True:
       plt.matshow(out[0], vmin=0, vmax=1)
       plt.colorbar()
       plt.savefig('../Embeddings/'+conn_density+'/'+data_mode+'mse_corrregion_corr_seed'+str(seed)+'_subject'+str(index)+'.png')



np.save('../Embeddings/'+conn_density+'/'+data_mode+'embeddings_mse_corrregion_corr_seed'+str(seed), embeddings)
np.save('../Embeddings/'+conn_density+'/'+data_mode+'recon_mse_corrregion_corr_seed'+str(seed), recon)




