#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python preprocess_FEC.py --file_path='/mas/u/asma_gh/uncnet/datasets/FEC_dataset/faceexp-comparison-data-train-public.csv'
CUDA_VISIBLE_DEVICES=4 python preprocess_FEC.py --file_path='/mas/u/asma_gh/uncnet/datasets/FEC_dataset/faceexp-comparison-data-test-public.csv'