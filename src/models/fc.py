from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def inference(images, keep_probability, phase_train=True, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return fc(images, is_training=phase_train, dropout_keep_prob=keep_probability, reuse=reuse)


def fc(inputs, is_training=True, num_classes=7, hidden_layer_size=[128, 128], dropout_keep_prob=0.8, reuse=None,
       scope='FC'):
    """Creates a fully connected neural net.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      hidden_layer_size: size of hidden layers, e.g. [x] or [x, y]
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}

    with tf.variable_scope(scope, 'FC', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            next_in = inputs
            for h_size in hidden_layer_size:
                net = slim.fully_connected(next_in, h_size, activation_fn=None, scope='Bottleneck', reuse=False)
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')
                next_in = net
            with tf.variable_scope('Logits'):
                net = slim.flatten(next_in)
                end_points['PreLogitsFlatten'] = net

            net = slim.fully_connected(net, num_classes, activation_fn=None, scope='Bottleneck', reuse=False)

    return net, end_points
