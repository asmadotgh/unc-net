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
import urllib.request
from PIL import Image


class BoundingBox:
    """
    Top-left column of the face bounding box in image1 normalized by width (float)
    Bottom-right column of the face bounding box in image1 normalized by width (float)
    Top-left row of the face bounding box in image1 normalized by height (float)
    Bottom-right row of the face bounding box in image1 normalized by height (float)
    """

    def __init__(self, top_left_column, bottom_right_column, top_left_row, bottom_right_row):
        self.top_left_column = top_left_column
        self.bottom_right_column = bottom_right_column
        self.top_left_row = top_left_row
        self.bottom_right_row = bottom_right_row

    def get_area(self, width, height):
        return (int(self.top_left_column * width), int(self.top_left_row * height),
                int(self.bottom_right_column * width), int(self.bottom_right_row * height))


class ImageDownloader:
    def __init__(self, file_path, seed=666):
        self.seed = seed
        np.random.seed(self.seed)
        self.file_path = file_path
        cols = []
        for i in range(3):
            cols += [f'img_{i}', f'top_left_column_{i}', f'bottom_right_column_{i}',
                     f'top_left_row_{i}', f'bottom_right_row_{i}']
        cols += ['triplet_type']
        for i in range(Constants.get_max_FEC_annotations()):
            cols += [f'id_{i}', f'label_{i}']
        self.dataset = pd.read_csv(file_path, header=None, names=cols, engine='python')
        self.root_dir = file_path[:file_path.rfind('/') + 1]
        if 'train' in file_path:
            self.images_dir = self.root_dir + 'images/train/'
        elif 'test' in file_path:
            self.images_dir = self.root_dir + 'images/test/'

    @staticmethod
    def download_image(url, bounding_box, raw_filename, processed_filename):
        # Downloading images
        try:
            request = urllib.request.urlopen(url, timeout=5)
            with open(raw_filename, 'wb') as f:
                f.write(request.read())
        except Exception as e:
            print(f'{str(e)}. Error downloading image {url}. Skipping downloading image.')
            return

        # Processing downloaded image
        try:
            img = Image.open(raw_filename)
            img_width, img_height = img.size
            cropped_img = img.crop(bounding_box.get_area(width=img_width, height=img_height))
            img_size = Constants.get_output_image_size()
            resized_img = cropped_img.resize((img_size, img_size))
            resized_img.save(processed_filename)
        except Exception as e:
            print(f'{str(e)}. Error reading image {raw_filename}. Skipping processing image.')

    def retrieve_images(self):
        if not os.path.exists(self.images_dir):
            print('Creating directories ...')
            os.makedirs(self.images_dir)
            os.makedirs(self.images_dir + 'raw/')
            os.makedirs(self.images_dir + 'processed/')

        for idx, row in self.dataset.iterrows():
            for i in range(3):
                url = row[i*5]
                if url.endswith('.gif') or url.endswith('.GIF') or url.endswith('.png') or url.endswith('.PNG'):
                    print(f'Unsupported format {url}. Skipping image.')
                    continue
                print(f'\033[1A\033[K idx: {idx}, i: {i}, url: {url}')
                bounding_box = BoundingBox(top_left_column=row[i*5+1],
                                           bottom_right_column=row[i*5+2],
                                           top_left_row=row[i*5+3],
                                           bottom_right_row=row[i*5+4])
                raw_filename = self.images_dir+f'raw/{idx}_{i}.jpg'
                processed_filename = self.images_dir + f'processed/{idx}_{i}.jpg'
                self.download_image(url, bounding_box, raw_filename, processed_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str,
                        default=None,
                        help='Path to the file containing image directories and labels.')
    args = parser.parse_args()
    image_downloader = ImageDownloader(args.file_path)
    image_downloader.retrieve_images()
