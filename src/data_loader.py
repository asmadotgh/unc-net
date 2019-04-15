import numpy as np
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import re
import pandas as pd
from PIL import Image
import argparse
from my_constants import Constants
import pickle


class DataLoader:
    def __init__(self, file_path, in_image_size=48, out_image_size=160, import_embedding=True,
                 embedding_model='VGGFace2_Inception_ResNet_v1', embedding_layer='Mixed_5a', seed=666):
        self.seed = seed
        np.random.seed(self.seed)
        self.root_dir = file_path[:file_path.rfind('/') + 1]
        self.subset = file_path[file_path.rfind('/') + 1:-4]
        self.import_embedding = import_embedding
        if self.import_embedding:
            embedding_dir = os.path.join(os.path.join(self.root_dir, 'embedding'), self.subset)
            embedding_file = os.path.join(embedding_dir, f'embeddings_{embedding_model}_{embedding_layer}.pkl')
            self.embeddings = pickle.load(open(embedding_file, 'rb'))
            self.embedding_size = np.shape(self.embeddings)[1]
        else:
            self.embedding = None
            self.embedding_size = None
        self.dataset = pd.read_csv(file_path)
        self.train = self.dataset[self.dataset['dataset'] == 'Training']
        self.valid = self.dataset[self.dataset['dataset'] == 'PrivateTest']
        self.test = self.dataset[self.dataset['dataset'] == 'PublicTest']
        self.in_image_size = in_image_size
        self.out_image_size = out_image_size
        self.nrof_samples = len(self.dataset)
        self.shuffled_train_indices = self.train.index.to_list()
        np.random.shuffle(self.shuffled_train_indices)
        return

    def get_nrof_train_sampels(self):
        return len(self.train)

    def get_nrof_samples(self):
        return self.nrof_samples

    def get_embedding_size(self):
        return self.embedding_size

    def get_image_size(self):
        return self.out_image_size

    @staticmethod
    def get_num_classes():
        return len(Constants.get_emotion_cols())

    @staticmethod
    def _to_rgb(img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    @staticmethod
    def _crop(image, random_crop, image_size):
        if image.shape[1] > image_size:
            sz1 = int(image.shape[1] // 2)
            sz2 = int(image_size // 2)
            if random_crop:
                diff = sz1 - sz2
                (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
            else:
                (h, v) = (0, 0)
            image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
        return image

    @staticmethod
    def _flip(image, random_flip):
        if random_flip and np.random.choice([True, False]):
            image = np.fliplr(image)
        return image

    @staticmethod
    def _get_model_filenames(model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files) > 1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file

        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups()) >= 2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

    def _save_image(self, inp_dataset, img_name, img):
        dataset = None
        if inp_dataset == 'Training':
            dataset = 'train'
        elif inp_dataset == 'PrivateTest':
            dataset = 'valid'
        elif inp_dataset == 'PublicTest':
            dataset = 'test'
        if not dataset:
            print('Wrong dataset. Only options are train/valid/test')
            return
        dataset_dir = os.path.join(os.path.join(self.root_dir, 'images'), dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        img.save(os.path.join(dataset_dir, img_name))

    def load_data(self, indices=None, do_random_crop=False, do_random_flip=False, save_images=False):
        if indices is None:
            indices = range(self.nrof_samples)
        nrof_samples = len(indices)
        images = np.zeros((nrof_samples, self.out_image_size, self.out_image_size, 3))
        labels = np.zeros((nrof_samples, len(Constants.get_emotion_cols())))
        for out_idx, sample_idx in enumerate(indices):
            # Getting images
            img_name = self.dataset.iloc[sample_idx]['img_name']
            inp_dataset = self.dataset.iloc[sample_idx]['dataset']
            img_arr = np.array([int(i) for i in self.dataset.iloc[sample_idx]['pixels'].split(' ')])
            img_arr = np.reshape(img_arr, (self.in_image_size, self.in_image_size))
            if img_arr.ndim == 2:
                img_arr = self._to_rgb(img_arr)
            img_arr = self._crop(img_arr, do_random_crop, self.in_image_size)
            img_arr = self._flip(img_arr, do_random_flip)
            img = Image.fromarray(img_arr, 'RGB')
            resized_img = img.resize((self.out_image_size, self.out_image_size))
            if save_images:
                self._save_image(inp_dataset, img_name, resized_img)
            images[out_idx, :, :, :] = resized_img

            # Getting labels
            inp_labels = self.dataset.iloc[sample_idx][Constants.get_emotion_cols()]
            n_annotations = sum(inp_labels)
            probs_labels = [float(i) for i in inp_labels] / n_annotations
            labels[out_idx, :] = probs_labels

        # Getting previously processed embeddings
        if self.import_embedding:
            embeddings = self.embeddings[indices]
        else:
            embeddings = None
        return images, labels, embeddings

    def load_model(self, model, input_map=None):
        # Check if the model is a model directory (containing a metagraph and a checkpoint file)
        #  or if it is a protobuf file with a frozen graph
        model_exp = os.path.expanduser(model)
        if os.path.isfile(model_exp):
            print('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, input_map=input_map, name='')
        else:
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = self._get_model_filenames(model_exp)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            saver = tf.train.Saver()
            saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    def get_train_batch(self, batch_size=None, idx=None):
        if idx is None:
            if batch_size is None:
                batch_size = len(self.train)
            indices = np.random.choice(self.train.index, size=batch_size)
            images, labels, embeddings = self.load_data(indices=indices)
        else:
            if ((idx + 1) * batch_size) > len(self.train):
                print('batch number exceeds train size.')
                return None, None, None
            indices = self.shuffled_train_indices[idx * batch_size:(idx + 1) * batch_size]
            images, labels, embeddings = self.load_data(indices=indices)
        return images, labels, embeddings

    def get_valid_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = len(self.valid)
        indices = np.random.choice(self.valid.index, size=batch_size)
        images, labels, embeddings = self.load_data(indices=indices)
        return images, labels, embeddings

    def get_test_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = len(self.test)
        indices = np.random.choice(self.test.index, size=batch_size)
        images, labels, embeddings = self.load_data(indices=indices)
        return images, labels, embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str,
                        default='/mas/u/asma_gh/uncnet/datasets/FER+/all.csv',
                        help='Path to the file containing faces/labels.')
    args = parser.parse_args()
    data_loader = DataLoader(args.file_path, in_image_size=48, out_image_size=160, import_embedding=False)
    data_loader.load_data(save_images=True)
