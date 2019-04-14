from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys
import math
from data_loader import DataLoader
import pickle
import os
from my_constants import Constants


def main(args):

    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            data_loader = DataLoader(args.data_dir, import_embedding=False)

            print('Number of images: %d' % len(data_loader.dataset))

            # Load the model
            print('Loading feature extraction model')
            data_loader.load_model(args.model)

            # print operations
            for op in tf.get_default_graph().get_operations():
                print(op.name)

            tf.summary.FileWriter(os.path.join(args.model, 'graph'), tf.get_default_graph())
            import pdb; pdb.set_trace()

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name(
                Constants.get_embedding_tensor_name(args.embedding_name))
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = data_loader.get_nrof_samples()
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                images, _, _ = data_loader.load_data(indices=np.arange(start_index, end_index))
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                print(f'\033[1A\033[KBatch {i} done!')

            print(f'embedding shape: {np.shape(emb_array)}')
            if not os.path.exists(args.embedding_dir):
                os.makedirs(args.embedding_dir)
            postfix = args.model[args.model.rfind('/')+1:]
            pickle.dump(emb_array,
                        open(os.path.join(args.embedding_dir, f'embeddings_{postfix}_{args.embedding_name}.pkl'), 'wb'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default='/mas/u/asma_gh/uncnet/datasets/FER+/all.csv',
                        help='Path to the data directory containing faces/labels.')
    parser.add_argument('--embedding_dir', type=str,
                        default='/mas/u/asma_gh/uncnet/datasets/FER+/embedding/',
                        help='Path to the embedding pkl file.')
    parser.add_argument('--embedding_name', type=str,
                        default='Mixed_5a',
                        help='Name of the embedding layer. Options: Mixed_8b, Mixed_8a, Mixed_7a, Mixed_6b, Mixed_5a, default.')
    parser.add_argument('--model', type=str,
                        default='/mas/u/asma_gh/uncnet/pretrained_models/VGGFace2_Inception_ResNet_v1', #CASIA_WebFace_Inception_ResNet_v1
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=48)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
