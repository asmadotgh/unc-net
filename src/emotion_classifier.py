from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os.path
import sys
import tensorflow as tf
import argparse
import math
from data_loader import DataLoader
from my_constants import Constants
import numpy as np


class EmotionClassifier:
    def __init__(self, filename, model_name, embedding_model='VGGFace2_Inception_ResNet_v1', embedding_layer='Mixed_5a',
                 layer_sizes=[128, 128], num_epochs=500, batch_size=90, learning_rate=.001, dropout_prob=1.0,
                 weight_penalty=0.0, clip_gradients=True, checkpoint_dir='/mas/u/asma_gh/uncnet/logs/', seed=666):
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

        # Logistics
        self.checkpoint_dir = checkpoint_dir
        self.filename = filename
        self.model_name = model_name
        self.output_every_nth = 10
        self.embedding_model = embedding_model
        self.embedding_layer = embedding_layer

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
        # import pdb; pdb.set_trace()
        self.data_loader = DataLoader(filename, import_embedding=True, embedding_model=self.embedding_model,
                                      embedding_layer=self.embedding_layer, seed=self.seed)
        self.input_size = self.data_loader.get_embedding_size()
        self.output_size = self.data_loader.get_num_classes()
        self.metric_name = 'accuracy'

        # Set up tensorflow computation graph.
        self.graph = tf.Graph()
        self.build_graph()

        # Set up and initialize tensorflow session.
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)

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

            self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.target, self.predictions)))
            self.mse = tf.reduce_mean(tf.square(tf.subtract(self.tf_y, self.class_probabilities)))
            self.rmse = tf.sqrt(self.mse)

            # TODO: debug per class metrics: num_target_labels, num_predicted_labels, acc, precision, recall, F1
            # TODO: add AUC per class metric
            self.num_target_labels = {}
            self.num_predicted_labels = {}
            self.acc_per_class = {}
            self.precision_per_class = {}
            self.recall_per_class= {}
            self.f1_per_class = {}
            self.AUC_per_class = {}

            emotion_labels = Constants.get_emotion_cols()
            for idx, emotion_label in enumerate(emotion_labels):
                class_target = tf.to_float(tf.equal(self.target, idx))
                class_prediction = tf.to_float(tf.equal(self.predictions, idx))

                self.num_target_labels[idx] = tf.reduce_sum(class_target)
                self.num_predicted_labels[idx] = tf.reduce_sum(class_prediction)
                self.acc_per_class[idx] = tf.reduce_mean(tf.to_float(tf.equal(class_target, class_prediction)))
                self.precision_per_class[idx] = tf.reduce_sum(tf.to_float(tf.equal(class_target, class_prediction)))/tf.reduce_sum(class_prediction)
                self.recall_per_class[idx] = tf.reduce_sum(tf.to_float(tf.equal(class_target, class_prediction)))/tf.reduce_sum(class_target)
                self.f1_per_class[idx] = 2*self.precision_per_class[idx]*self.recall_per_class[idx]/(
                        self.precision_per_class[idx]+self.recall_per_class[idx])
                # self.AUC_per_class[idx] = tf.constant(np.nan, shape=[1])

                tf.summary.scalar(f'metrics_{emotion_label}/num_target_labels', self.num_target_labels[idx])
                tf.summary.scalar(f'metrics_{emotion_label}/num_predicted_labels', self.num_predicted_labels[idx])
                tf.summary.scalar(f'metrics_{emotion_label}/acc', self.acc_per_class[idx])
                tf.summary.scalar(f'metrics_{emotion_label}/precision', self.precision_per_class[idx])
                tf.summary.scalar(f'metrics_{emotion_label}/recall', self.recall_per_class[idx])
                tf.summary.scalar(f'metrics_{emotion_label}/F1', self.f1_per_class[idx])
                # tf.summary.scalar(f'metrics_{emotion_label}/AUC', self.AUC_per_class[idx])


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

            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('logits', self.logits)
            tf.summary.scalar('metrics_all/KL', self.kl)
            tf.summary.scalar('metrics_all/Log-loss', self.log_loss)

            tf.summary.scalar('metrics_all/acc', self.acc)
            tf.summary.scalar('metrics_all/MSE', self.mse)
            tf.summary.scalar('metrics_all/RMSE', self.rmse)
            self.summaries = tf.summary.merge_all()

            self.init = tf.global_variables_initializer()

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
                self.data_loader.reshuffle()
                for step in range(steps_per_epoch):
                    global_step = num_epoch * steps_per_epoch + step
                    # Grab a batch of data to feed into the placeholders in the graph.
                    labels, embeddings = self.data_loader.get_train_batch(batch_size=self.batch_size, idx=step)

                    # TODO [p2]: write a test for these instead of trying them here
                    # DEBUG - does it overfit to all neutral input? Yes, passed
                    # labels = np.repeat(np.array([[1.0, 0, 0, 0, 0, 0, 0, 0, 0]]), [self.batch_size], axis=0)
                    # DEBUG - does it overfit to a small training set? Yes, passed
                    # labels, embeddings = self.data_loader.get_train_batch(batch_size=self.batch_size, idx=0)
                    # DEBUG - What about a slightly larger dataset (almost 1/10 of data)? Yes, Pass
                    # labels, embeddings = self.data_loader.get_train_batch(batch_size=self.batch_size, idx=step % 30)

                    feed_dict = {self.tf_x: embeddings,
                                 self.tf_y: labels,
                                 self.tf_dropout_prob: self.dropout_prob}

                    # Update parameters in the direction of the gradient computed by
                    # the optimizer.
                    self.session.run([self.opt_step], feed_dict)

                    # Evaluate model every nth step
                    if global_step % self.output_every_nth == 0:
                        # Grab a random batch of train data.
                        train_labels, train_embeddings = self.data_loader.get_train_batch(batch_size=self.batch_size)
                        train_feed_dict = {self.tf_x: train_embeddings, self.tf_y: train_labels,
                                           self.tf_dropout_prob: 1.0}

                        # Grab all validation data.
                        valid_labels, valid_embeddings = self.data_loader.get_valid_batch()
                        val_feed_dict = {self.tf_x: valid_embeddings, self.tf_y: valid_labels,
                                         self.tf_dropout_prob: 1.0}
                        # TODO [p0] add dropout for epistemic bayesian
                        # TODO [p0] add dropout for aleatoric bayesian

                        # TODO [p2] reset metrics?
                        # stream_vars_valid = [v for v in tf.local_variables() if 'valid/' in v.name]
                        # sess.run(tf.variables_initializer(stream_vars_valid))

                        train_summaries, train_score, train_loss = self.session.run(
                            [self.summaries, self.acc, self.loss], train_feed_dict)
                        valid_summaries, valid_score, valid_loss = self.session.run(
                            [self.summaries, self.acc, self.loss], val_feed_dict)

                        self.train_summary_writer.add_summary(train_summaries, global_step=num_epoch)
                        self.valid_summary_writer.add_summary(valid_summaries, global_step=num_epoch)

                        print(f"Epoch #: {num_epoch}, training step: {step}, global step: {global_step}")
                        print(f"\tTraining {self.metric_name} {train_score}, Loss: {train_loss}")
                        print(f"\tValidation {self.metric_name} {valid_score}, Loss: {valid_loss}")

                        # Save a checkpoint of the model
                        self.saver.save(self.session, self.checkpoint_dir + self.model_name + '.ckpt',
                                        global_step=global_step)

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
        valid_labels, valid_embeddings = self.data_loader.get_valid_batch()
        score = self.get_performance_on_data(valid_embeddings,
                                             valid_labels)
        print(f"Final {self.metric_name} on validation data is: {score}")
        return score

    def test_on_test(self):
        """Returns performance on the model's test set."""
        test_labels, test_embeddings = self.data_loader.get_test_batch()
        score = self.get_performance_on_data(test_embeddings,
                                             test_labels)
        print(f"Final {self.metric_name} on test data is: {score}")
        return score

    def get_performance_on_data(self, x, y):
        """Returns the model's performance on input data X and targets Y."""
        feed_dict = {self.tf_x: x,
                     self.tf_y: y,
                     self.tf_dropout_prob: 1.0}  # no dropout during evaluation

        score = self.session.run(self.acc, feed_dict)

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
    log_dir = f'{args.logs_base_dir}/{args.embedding_model}/{args.embedding_layer}/{str(args.learning_rate)}/{str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    hparam_file = open(os.path.join(log_dir, 'hparams.txt'), 'w')
    for arg in dir(args):
        if arg.startswith('_'):
            continue
        curr_arg = eval(f'args.{arg}')
        hparam_file.write(f'{arg}={curr_arg}\n')
    hparam_file.close()

    emotion_classifier = EmotionClassifier(filename=args.file_path, model_name=args.model_name,
                                           embedding_model=args.embedding_model, embedding_layer=args.embedding_layer,
                                           checkpoint_dir=log_dir,
                                           batch_size=args.batch_size,
                                           num_epochs=args.max_nrof_epochs, layer_sizes=args.hidden_layer_size,
                                           dropout_prob=args.keep_probability, learning_rate=args.learning_rate,
                                           weight_penalty=args.weight_decay, seed=args.seed)
    emotion_classifier.train(output_every_nth=args.output_every_nth)
    emotion_classifier.test_on_validation()
    emotion_classifier.test_on_test()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str,
                        default='/mas/u/asma_gh/uncnet/datasets/FER+/all.csv',
                        help='Path to the data file containing aligned faces/labels.')
    parser.add_argument('--embedding_model', type=str,
                        default='VGGFace2_Inception_ResNet_v1',
                        help='The pre-trained model to use for exporting embedding. '
                             'Options: VGGFace2_Inception_ResNet_v1, CASIA_WebFace_Inception_ResNet_v1')
    parser.add_argument('--embedding_layer', type=str,
                        default='Mixed_5a',
                        help='Name of the embedding layer. Options: Mixed_8b, Mixed_8a, Mixed_7a, Mixed_6b, Mixed_6a, Mixed_5a.')
    parser.add_argument('--model_name', type=str,
                        help='Model name.',
                        default='FC')
    parser.add_argument('--logs_base_dir', type=str,
                        default='/mas/u/asma_gh/uncnet/logs/',
                        help='Directory where to write event logs.')
    parser.add_argument('--output_every_nth', type=int,
                        help='Write to tensorboard every n batches of training.', default=1000)
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=100000)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--hidden_layer_size', type=list,
                        help='Dimensionality of FC layers.', default=[128, 128])
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.001)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)


    # TODO [p2] need these?
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
