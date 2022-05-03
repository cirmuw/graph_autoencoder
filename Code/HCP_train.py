# code to train variational autoencoder on HCP data
# connectivity matrix and parcel mean time series are required
# seed and connectome density can be set manually in the script



from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import tensorflow_probability as tfp
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
from myfunc import *

################ set parameters ##########################################################
seed=1 # set seed, in this project seed was set 1,2,3,4 and 5
conn_density='full_top90'# set density of connectome, full: all connections above 0, full_top90: connections above 90%ile, full_top50: connections above 50%ile
###########################################################################################

# Load data, use 80% for training the auto-encoder, 5% for validation, rest for test
adj_post_orig, adj_pre, features_pre = load_myDataHCP('correlation', '100',conn_density)
train_adj_post_orig=[adj_post_orig[ind] for ind in range(int(np.floor(len(adj_pre)*0.8)))]
train_adj_pre=[adj_pre[ind] for ind in range(int(np.floor(len(adj_pre)*0.8)))]
train_features_pre=[features_pre[ind] for ind in range(int(np.floor(len(adj_pre)*0.8)))]
valid_adj_post_orig=[adj_post_orig[ind] for ind in range(int(np.floor(len(adj_pre)*0.8)),int(np.floor(len(adj_pre)*0.85)))]
valid_adj_pre=[adj_pre[ind] for ind in range(int(np.floor(len(adj_pre)*0.8)),int(np.floor(len(adj_pre)*0.85)))]
valid_features_pre=[features_pre[ind] for ind in range(int(np.floor(len(adj_pre)*0.8)),int(np.floor(len(adj_pre)*0.85)))]

# Additional settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')  
flags.DEFINE_integer('epochs', int(np.floor(len(adj_pre)*0.8))*25, 'Number of epochs to train.')  
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
    np.random.seed(seed)         # set seed

tf.compat.v1.disable_eager_execution()

# create model
num_features = 100
keep_features_for_rois = None
features_nonzero = 10000
num_nodes = 100
placeholders = {
    'features': tf.compat.v1.sparse_placeholder(tf.float32),
    'adj_pre': tf.compat.v1.sparse_placeholder(tf.float32),
    'adj_post_orig': tf.compat.v1.sparse_placeholder(tf.float32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
}
model = myModel(placeholders, num_features, num_nodes, features_nonzero, keep_features_for_rois)

with tf.name_scope('optimizer'):
    opt = OptimizerSiemens(preds=model.reconstructions,
                           labels=tf.reshape(tf.compat.v1.sparse_tensor_to_dense(placeholders['adj_post_orig'],
                                                                                 validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=1,
                           norm=1,
                           conn_density=conn_density)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if FLAGS.gpu == -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.compat.v1.Session()
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)  # Or whichever device you would like to use
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

sess.run(tf.compat.v1.global_variables_initializer())

# training
if os.path.exists('../TrainedModels/'+conn_density+'/')==False:
    os.mkdir('../TrainedModels/'+conn_density+'/')
trueepoch=1
saver = tf.compat.v1.train.Saver()
shuffleind=list(range(len(train_adj_pre)))
for iter in range(FLAGS.epochs):
    index = iter % len(train_adj_pre)
    if index==0:
        valid_loss=[]
        for j in range(len(valid_adj_pre)):
            feed_dict = construct_feed_dict(valid_adj_pre[j], valid_adj_post_orig[j], valid_features_pre[j],placeholders)  ########
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            out = sess.run([model.reconstructions_noiseless_full],feed_dict=feed_dict)
            recon=out[0]
            truth=sp.coo_matrix((valid_adj_post_orig[j][1],(valid_adj_post_orig[j][0][:,0],valid_adj_post_orig[j][0][:,1])),(100,100))
            truth=np.array(truth.todense(),dtype='float64')
            valid_loss.append(np.mean(np.square(recon-truth)))

        Valid_loss=np.mean(valid_loss)
        saver.save(sess,'../TrainedModels/'+conn_density+'/checkpoint_pretrainOptimalepochs_shuffle_mse_corrregion_corr_seed'+str(seed)+'/model',
                   global_step=trueepoch)
        np.random.shuffle(shuffleind)
        train_adj_post_orig = [adj_post_orig[ind] for ind in shuffleind]
        train_adj_pre = [adj_pre[ind] for ind in shuffleind]
        train_features_pre = [features_pre[ind] for ind in shuffleind]

    feed_dict = construct_feed_dict(train_adj_pre[index], train_adj_post_orig[index], train_features_pre[index], placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    outs = sess.run([opt.accuracy, opt.cost, opt.opt_op, opt.MSE, opt.corr, opt.corr_region], feed_dict=feed_dict)
    avg_cost = outs[1]
    avg_accuracy = outs[0]
    mse = outs[3]
    corr = np.squeeze(outs[4])

    if FLAGS.verbose and index == 0:
        print("Epoch:", '%04d' % (trueepoch), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "corr_cost=", "{:.5f}".format(corr), "mse=",
          "{:.5f}".format(mse), "valid_loss=", "{:.5f}".format(Valid_loss))
        trueepoch=trueepoch+1
