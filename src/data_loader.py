import numpy as np
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import re
import pandas as pd


class DataLoader:
    def __init__(self, data_dir, image_size=48):
        self.dataset = pd.read_csv(data_dir)
        self.train = self.dataset[self.dataset['dataset'] == 'Training'].reset_index(drop=True)
        self.valid = self.dataset[self.dataset['dataset'] == 'PrivateTest'].reset_index(drop=True)
        self.test = self.dataset[self.dataset['dataset'] == 'PublicTest'].reset_index(drop=True)
        self.image_size = image_size
        self.nrof_samples = len(self.dataset)
        return

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

    def load_data(self, do_random_crop, do_random_flip):
        images = np.zeros((self.nrof_samples, self.image_size, self.image_size, 3))
        for sample_idx in range(self.nrof_samples):
            img = np.array([int(i) for i in self.dataset.iloc[sample_idx]['pixels'].split(' ')])
            img = np.reshape(img, (self.image_size, self.image_size))
            if img.ndim == 2:
                img = self._to_rgb(img)
            img = self._crop(img, do_random_crop, self.image_size)
            img = self._flip(img, do_random_flip)
            images[sample_idx, :, :, :] = img
        return images

    def load_model(self, model, input_map=None):
        # Check if the model is a model directory (containing a metagraph and a checkpoint file)
        #  or if it is a protobuf file with a frozen graph
        model_exp = os.path.expanduser(model)
        if (os.path.isfile(model_exp)):
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

            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
            saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
