#!/usr/bin/env bash

python preprocess_FEC.py --file_path='/mas/u/asma_gh/uncnet/datasets/FEC_dataset/faceexp-comparison-data-train-public.csv'
python preprocess_FEC.py --file_path='/mas/u/asma_gh/uncnet/datasets/FEC_dataset/faceexp-comparison-data-test-public.csv'