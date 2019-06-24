from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf


# Define the classification loss
def classification_loss(logits, labels):
    with tf.name_scope("classification_loss"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(labels, axis=3))
        loss = tf.reduce_mean(loss)

    return loss


# Define the jaccard loss
def jaccard_loss(logits, labels):
    with tf.name_scope("jaccard_loss"):
        gt_onehot = tf.one_hot(tf.squeeze(labels, axis=3), depth=2, dtype=tf.float32)
        smooth = 1e-9
        intersection = tf.reduce_sum(logits * gt_onehot)
        union = tf.reduce_sum(labels) + tf.reduce_sum(gt_onehot)
        jaccard_index = (intersection + smooth) / (union - intersection + smooth)
        jaccard_loss = (1 - jaccard_index)

    return jaccard_loss