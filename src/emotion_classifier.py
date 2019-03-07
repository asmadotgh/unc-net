from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import h5py
import math
from tensorflow.python.ops import data_flow_ops
from data_loader import DataLoader
from datetime import datetime
from my_constants import Constants


class EmotionClassifier:
    def __init__(self, filename, model_name, layer_sizes=[128, 128], num_epochs=500, batch_size=90,
                 learning_rate=.01, dropout_prob=1.0, weight_penalty=0.0,
                 clip_gradients=True, checkpoint_dir='/mas/u/asma_gh/uncnet/logs/', seed=666):
        self.epsilon = 1e-20
        '''Initialize the class by loading the required datasets
        and building the graph.
        Args:
            filename: a file containing the data.
            model_name: name of the model being trained. Used in saving
                model checkpoints.
            layer_sizes: a list of sizes of the neural network layers.
            batch_size: number of training examples in each training batch.
            learning_rate: the initial learning rate used in stochastic
                gradient descent.
            dropout_prob: the probability that a node in the network will not
                be dropped out during training. Set to < 1.0 to apply dropout,
                1.0 to remove dropout.
            weight_penalty: the coefficient of the L2 weight regularization
                applied to the loss function. Set to > 0.0 to apply weight
                regularization, 0.0 to remove.
            clip_gradients: a bool indicating whether or not to clip gradients.
                This is effective in preventing very large gradients from skewing
                training, and preventing your loss from going to inf or nan.
            checkpoint_dir: the directly where the model will save checkpoints,
                saved files containing trained network weights.
            '''

        # Hyperparameters that should be tuned
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.weight_penalty = weight_penalty

        # Hyperparameters that could be tuned
        self.clip_gradients = clip_gradients
        self.activation_func = 'relu'
        self.optimizer = tf.train.AdamOptimizer

        # Extract the data from the filename
        self.seed = seed
        self.data_loader = DataLoader(filename, import_embedding=True, seed=self.seed)
        self.input_size = self.data_loader.get_embedding_size()
        self.output_size = self.data_loader.get_num_classes()
        self.metric_name = 'accuracy'

        # Set up tensorflow computation graph.
        self.graph = tf.Graph()
        self.build_graph()

        # Set up and initialize tensorflow session.
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.global_init)
        self.session.run(self.local_init)
        # self.session.run(tf.local_variables_initializer())

        # Logistics
        self.checkpoint_dir = checkpoint_dir
        self.filename = filename
        self.model_name = model_name
        self.output_every_nth = 10

        # Tensorboard
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.checkpoint_dir, 'train'), self.graph)
        self.valid_summary_writer = tf.summary.FileWriter(os.path.join(self.checkpoint_dir, 'validation'))

    def _plus_eps(self, inp):
        return tf.add(inp, self.epsilon)

    def initialize_network_weights(self):
        """Constructs Tensorflow variables for the weights and biases
        in each layer of the graph.
        """
        sizes = []
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) + 1):
            if i == 0:
                input_len = self.input_size  # X second dimension
            else:
                input_len = self.layer_sizes[i - 1]

            if i == len(self.layer_sizes):
                output_len = self.output_size
            else:
                output_len = self.layer_sizes[i]

            layer_weights = weight_variable([input_len, output_len], name='weights' + str(i))
            layer_biases = bias_variable([output_len], name='biases' + str(i))

            self.weights.append(layer_weights)
            self.biases.append(layer_biases)
            sizes.append((str(input_len) + "x" + str(output_len), str(output_len)))

        print(f"Making a fully connected net with the following structure: {sizes}")

    def build_graph(self):
        """Constructs the tensorflow computation graph containing all variables
        that will be trained."""
        print('Building computation graph...')

        with self.graph.as_default():
            self.tf_x = tf.placeholder(tf.float32, shape=(None, self.input_size), name="x")  # features
            self.tf_y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="y")  # labels
            self.tf_dropout_prob = tf.placeholder(tf.float32)  # Implements dropout

            # TODO [p1] add loading from previous checkpoint
            self.initialize_network_weights()

            # Defines the actual network computations using the weights.
            def run_network(input_x):
                hidden = input_x
                for i in range(len(self.weights)):
                    with tf.name_scope('layer' + str(i)) as scope:
                        hidden = tf.matmul(hidden, self.weights[i]) + self.biases[i]

                        if i < len(self.weights) - 1:
                            # Apply activation function
                            if self.activation_func == 'relu':
                                hidden = tf.nn.relu(hidden)

                            # Apply dropout
                            hidden = tf.nn.dropout(hidden, self.tf_dropout_prob)
                return hidden

            self.run_network = run_network

            # Compute the loss function
            self.logits = run_network(self.tf_x)

            # Apply a softmax function to get probabilities, train this dist against targets with
            # cross entropy loss.
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.tf_y))

            # Add weight decay regularization term to loss
            self.loss += self.weight_penalty * sum([tf.nn.l2_loss(w) for w in self.weights])

            self.class_probabilities = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.class_probabilities, axis=1)
            self.target = tf.argmax(self.tf_y, axis=1)

            self.kl = tf.reduce_sum(self._plus_eps(self.tf_y) * tf.log(
                self._plus_eps(self.tf_y) / self._plus_eps(self.class_probabilities)))
            self.log_loss = - tf.reduce_sum(
                self._plus_eps(self.tf_y) * tf.log(self._plus_eps(self.class_probabilities)) + (
                            tf.ones_like(self.tf_y) - self._plus_eps(self.tf_y)) * tf.log(
                    tf.ones_like(self.class_probabilities) - self._plus_eps(self.class_probabilities)))
            self.accuracy, _ = tf.metrics.accuracy(labels=self.target, predictions=self.predictions, name='metrics')
            self.mse, _ = tf.metrics.mean_squared_error(labels=self.tf_y, predictions=self.class_probabilities,
                                                        name='metrics')
            self.rmse, _ = tf.metrics.root_mean_squared_error(labels=self.tf_y, predictions=self.class_probabilities,
                                                        name='metrics')
            self.mean_per_class_accuracy, _ = tf.metrics.mean_per_class_accuracy(labels=self.target,
                                                                                 predictions=self.predictions,
                                                                                 num_classes=Constants.get_no_emotions(),
                                                                                 name='metrics')
            self.mean_cosine_distance, _ = tf.metrics.mean_cosine_distance(labels=self.tf_y,
                                                                           predictions=self.class_probabilities, dim=1,
                                                                           name='metrics')

            # Set up backpropagation computation
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.loss, self.params)
            if self.clip_gradients:
                self.gradients, _ = tf.clip_by_global_norm(self.gradients, 5)
            self.tf_optimizer = self.optimizer(self.learning_rate)
            self.opt_step = self.tf_optimizer.apply_gradients(zip(self.gradients, self.params),
                                                              self.global_step)

            [tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in self.gradients]
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('logits', self.logits)
            tf.summary.scalar('distance_metrics/KL', self.kl)
            tf.summary.scalar('distance_metrics/Log-loss', self.log_loss)
            tf.summary.scalar('distance_metrics/MSE', self.mse)
            tf.summary.scalar('distance_metrics/RMSE', self.rmse)
            tf.summary.scalar('distance_metrics/mean-per-class-accuracy', self.mean_per_class_accuracy)
            tf.summary.scalar('distance_metrics/mean-cosine-distance', self.mean_cosine_distance)
            self.summaries = tf.summary.merge_all()

            # Isolate the variables stored behind the scenes by the metric operation
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")

            # Define initializer to initialize/reset running variables
            self.local_init = tf.variables_initializer(var_list=running_vars)

            self.global_init = tf.global_variables_initializer()

    def train(self, output_every_nth=None):
        """Trains using stochastic gradient descent (SGD).

        Runs batches of training data through the model for a given
        number of steps.
        Note that if you set the class's batch size to the number
        of points in the training data, you would be doing gradient
        descent rather than SGD. SGD is preferred since it has a
        strong regularizing effect.
        """

        if output_every_nth is not None:
            self.output_every_nth = output_every_nth

        with self.graph.as_default():
            # Used to save model checkpoints.
            self.saver = tf.train.Saver()

            steps_per_epoch = int(self.data_loader.get_nrof_train_sampels()/self.batch_size)
            for num_epoch in range(self.num_epochs):
                for step in range(steps_per_epoch):
                    # Resetting metrics for each batch?
                    self.session.run(self.local_init)
                    # Grab a batch of data to feed into the placeholders in the graph.
                    _, labels, embeddings = self.data_loader.get_train_batch(batch_size=self.batch_size, idx=step)
                    feed_dict = {self.tf_x: embeddings,
                                 self.tf_y: labels,
                                 self.tf_dropout_prob: self.dropout_prob}

                    # Update parameters in the direction of the gradient computed by
                    # the optimizer.
                    _ = self.session.run([self.opt_step], feed_dict)

                    # Output/save the training and validation performance every few steps.
                if step % self.output_every_nth == 0:
                    # Grab a batch of validation data too.
                    _, valid_labels, valid_embeddings = self.data_loader.get_valid_batch(self.batch_size)
                    val_feed_dict = {self.tf_x: valid_embeddings,
                                     self.tf_y: valid_labels,
                                     self.tf_dropout_prob: 1.0}  # TODO [p1] add dropout for epistemic bayesian

                    train_summaries, train_score, train_loss = self.session.run(
                        [self.summaries, self.accuracy, self.loss], feed_dict)
                    valid_summaries, valid_score, valid_loss = self.session.run(
                        [self.summaries, self.accuracy, self.loss], val_feed_dict)

                    self.train_summary_writer.add_summary(train_summaries, global_step=num_epoch*steps_per_epoch+step)
                    self.valid_summary_writer.add_summary(valid_summaries, global_step=num_epoch*steps_per_epoch+step)

                    print(f"Epoch #{num_epoch}, Training iteration {step}")
                    print(f"\tTraining {self.metric_name} {train_score}, Loss: {train_loss}")
                    print(f"\tValidation {self.metric_name} {valid_score}, Loss: {valid_loss}")

                    # Save a checkpoint of the model
                    self.saver.save(self.session, self.checkpoint_dir + self.model_name + '.ckpt', global_step=step)

    def predict(self, x, get_probabilities=False):
        """Gets the network's predictions for some new data X

        Args:
            x: a matrix of data in the same format as the training
                data.
            get_probabilities: a boolean that if true, will cause
                the function to return the model's computed softmax
                probabilities in addition to its predictions. Only
                works for classification.
        Returns:
            integer class predictions if the model is doing
            classification, otherwise float predictions if the
            model is doing regression.
        """
        feed_dict = {self.tf_x: x,
                     self.tf_dropout_prob: 1.0}  # no dropout during evaluation

        probs, preds = self.session.run([self.class_probabilities, self.predictions],
                                        feed_dict)
        if get_probabilities:
            return preds, probs
        else:
            return preds

    def test_on_validation(self):
        """Returns performance on the model's validation set."""
        _, valid_labels, valid_embeddings = self.data_loader.get_valid_batch()
        score = self.get_performance_on_data(valid_embeddings,
                                             valid_labels)
        print(f"Final {self.metric_name} on validation data is: {score}")
        return score

    def test_on_test(self):
        """Returns performance on the model's test set."""
        _, test_labels, test_embeddings = self.data_loader.get_test_batch()
        score = self.get_performance_on_data(test_embeddings,
                                             test_labels)
        print(f"Final {self.metric_name} on test data is: {score}")
        return score

    def get_performance_on_data(self, x, y):
        """Returns the model's performance on input data X and targets Y."""
        feed_dict = {self.tf_x: x,
                     self.tf_y: y,
                     self.tf_dropout_prob: 1.0}  # no dropout during evaluation

        score = self.session.run(self.accuracy, feed_dict)

        return score


def weight_variable(shape, name):
    """Initializes a tensorflow weight variable with random
    values centered around 0.
    """
    initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(shape[0])), dtype=tf.float32)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """Initializes a tensorflow bias variable to a small constant value."""
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def main(args):
    emotion_classifier = EmotionClassifier(filename=args.file_path, model_name=args.model_name,
                                           checkpoint_dir=args.logs_base_dir+str(datetime.now().timestamp()), batch_size=args.batch_size,
                                           num_epochs=args.max_nrof_epochs, layer_sizes=args.hidden_layer_size,
                                           dropout_prob=args.keep_probability, learning_rate=args.learning_rate,
                                           weight_penalty=args.weight_decay, seed=args.seed)
    emotion_classifier.train(output_every_nth=100)
    emotion_classifier.test_on_validation()
    emotion_classifier.test_on_test()

    # TODO [p1] change to tf.slim or sth for more compact presentation?


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str,
                        default='/mas/u/asma_gh/uncnet/datasets/FER+/all.csv',
                        help='Path to the data file containing aligned faces/labels.')
    parser.add_argument('--model_name', type=str,
                        help='Model name.',
                        default='FC')
    parser.add_argument('--logs_base_dir', type=str,
                        default='/mas/u/asma_gh/uncnet/logs/',
                        help='Directory where to write event logs.')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=1000000)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--hidden_layer_size', type=list,
                        help='Dimensionality of the embedding.', default=[128, 128])
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)


    # TODO [p1] need these?
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',
                        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization',
                        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
                        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=0.0)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--prelogits_hist_max', type=float,
                        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
                        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--validate_every_n_epochs', type=int,
                        help='Number of epoch between validation', default=5)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))