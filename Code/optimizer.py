import tensorflow as tf
import tensorflow_probability as tfp
from layers import *

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        
        if FLAGS.subsample:
            #Subsample edges for scalable monte carlo estimate
            edge_count = tf.count_nonzero(labels_sub)
            edge_indices = tf.where(tf.not_equal(labels_sub, 0))
            no_edge_count = tf.count_nonzero(labels_sub, keepdims = True)
            no_edge_count = FLAGS.subsample_frac * tf.cast(no_edge_count, tf.float32)
            no_edge_count = tf.cast(no_edge_count, tf.int64)
            no_edge_indices = tf.random_uniform(no_edge_count, maxval = num_nodes*num_nodes, dtype=tf.int32)
            self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.gather(preds_sub, edge_indices), targets=tf.gather(labels_sub, edge_indices), pos_weight=1))
            self.cost += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.gather(preds_sub, no_edge_indices), targets=tf.gather(labels_sub, no_edge_indices), pos_weight=1))
        else:
            self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.log_lik = self.cost

        if FLAGS.vae:
            self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) - tf.square(tf.exp(model.z_log_std)), 1))
            self.cost -= self.kl


        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32), tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

class OptimizerSiemens(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm,conn_density):
        preds_sub = preds
        labels_sub = labels

        #neg_cost = tf.reduce_sum(tf.compat.v1.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=0))
        #pos_cost = tf.reduce_sum(tf.compat.v1.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=1)) - neg_cost
        #total = tf.reduce_sum(labels)

        #self.cost = pos_cost / total + neg_cost / (num_nodes * num_nodes - total)
        #self.cost = pos_cost*0.95  + neg_cost*0.05
        #self.cost = pos_cost*(num_nodes * num_nodes - total)/(num_nodes*num_nodes)  + neg_cost*total/(num_nodes*num_nodes)
        #self.nrconn=total
###################################################
        # loss for weighted graph

        self.MSE=tf.compat.v1.losses.mean_squared_error(labels=labels_sub,predictions=preds_sub)
        if conn_density=='full_top90':
            conn_mask=tf.where(tf.math.greater(labels_sub,tf.zeros_like(labels_sub)),tf.ones_like(labels_sub),tf.zeros_like(labels_sub))
            nonconn_mask=1-conn_mask
            nr_of_conns=tf.reduce_sum(conn_mask)
            conn_mse=tf.compat.v1.losses.mean_squared_error(labels=labels_sub,predictions=tf.math.multiply(conn_mask,preds_sub))
            nonconn_mse=tf.compat.v1.losses.mean_squared_error(labels=tf.math.multiply(nonconn_mask,labels_sub),predictions=tf.math.multiply(nonconn_mask,preds_sub))
            self.mse=conn_mse*(num_nodes * num_nodes - nr_of_conns)/(num_nodes*num_nodes)+nonconn_mse*nr_of_conns/(num_nodes*num_nodes)
        else:
             self.mse=tf.compat.v1.losses.mean_squared_error(labels=labels_sub,predictions=preds_sub)
        self.corr=1-tf.abs(tfp.stats.correlation(tf.expand_dims(preds_sub,1),tf.expand_dims(labels_sub,1)))
        tmp=1-tf.abs(tf.linalg.diag_part(tfp.stats.correlation(tf.reshape(preds_sub,[num_nodes,num_nodes]),tf.reshape(labels_sub, [num_nodes,num_nodes]))))
        self.corr_region=tf.reduce_mean(tf.where(tf.math.is_nan(tmp), tf.ones_like(tmp), tmp))
        self.cost=self.mse+tf.squeeze(self.corr)+self.corr_region

##################################################



        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.log_lik = self.cost

        if FLAGS.vae:
            self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) - tf.square(tf.exp(model.z_log_std)), 1))
            self.cost -= self.kl


        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.nn.sigmoid(preds_sub), 0.5), tf.int32), tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
