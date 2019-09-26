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
import importlib


def main(args):
    # Build the inference graph
    network = importlib.import_module('models.inception_resnet_v1', 'inference')
    phase_train_bool = False

    with tf.Session() as sess:

        np.random.seed(seed=args.seed)

        data_loader = DataLoader(args.data_dir, import_embedding=False)
        print('Number of images: %d' % len(data_loader.dataset))

        images_placeholder = tf.placeholder(np.float32,
                                            shape=(None, args.image_size, args.image_size, 3),
                                            name='input')

        # keep probability is 1.0
        # because we are not doing bayesian epistemic uncertainty on the embedding space
        _, endpoints = network.inference(images_placeholder, keep_probability=1.0, bottleneck_layer_size=512,
                                         phase_train=phase_train_bool)
        embeddings = endpoints[f'{args.embedding_name}_flatten']
        embedding_size = embeddings.get_shape()[1]

        # Load the model
        print('Loading pre-trained model...')
        data_loader.load_model(args.model)

        tf.summary.FileWriter(os.path.join(args.model, 'graph'), tf.get_default_graph())

        # Run forward pass to calculate embeddings
        print('Calculating features for images...')
        nrof_images = data_loader.get_nrof_samples()
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * args.batch_size
            end_index = min((i + 1) * args.batch_size, nrof_images)
            images, _, _ = data_loader.load_data(indices=np.arange(start_index, end_index))
            feed_dict = {images_placeholder: images}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            print(f'\033[1A\033[KBatch {i} done!')

        print(f'embedding shape: {np.shape(emb_array)}')
        if not os.path.exists(args.embedding_dir):
            os.makedirs(args.embedding_dir)
        postfix = args.model[args.model.rfind('/')+1:]
        subset = args.data_dir[args.data_dir.rfind('/') + 1:-4]
        pickle.dump(emb_array,
                    open(os.path.join(args.embedding_dir, f'{subset}/embeddings_{postfix}_{args.embedding_name}.pkl'),
                         'wb'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default=None,
                        help='Path to the data directory containing faces/labels.')
    parser.add_argument('--embedding_dir', type=str,
                        default=None,
                        help='Path to the embedding pkl file.')
    parser.add_argument('--embedding_name', type=str,
                        default='Mixed_7a',
                        help='Name of the embedding layer. Options: Mixed_8b, Mixed_8a, Mixed_7a, Mixed_6b, Mixed_6a, Mixed_5a.')
    parser.add_argument('--model', type=str,
                        default=None,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))