#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python emotion_classifier.py --embedding_model="CASIA_WebFace_Inception_ResNet_v1" --embedding_layer="Mixed_7a" --learning_rate=0.0001 --uncertainty_type='aleatoric' --n_aleatoric=100 --logs_base_dir='/mas/u/asma_gh/uncnet/unc_logs/' &
CUDA_VISIBLE_DEVICES=2 python emotion_classifier.py --embedding_model="CASIA_WebFace_Inception_ResNet_v1" --embedding_layer="Mixed_7a" --learning_rate=0.0001 --uncertainty_type='epistemic' --n_epistemic=100 --logs_base_dir='/mas/u/asma_gh/uncnet/unc_logs/' &
CUDA_VISIBLE_DEVICES=3 python emotion_classifier.py --embedding_model="CASIA_WebFace_Inception_ResNet_v1" --embedding_layer="Mixed_7a" --learning_rate=0.0001 --uncertainty_type='both' --n_aleatoric=100 --n_epistemic=100 --logs_base_dir='/mas/u/asma_gh/uncnet/unc_logs/' &
CUDA_VISIBLE_DEVICES=4 python emotion_classifier.py --embedding_model="CASIA_WebFace_Inception_ResNet_v1" --embedding_layer="Mixed_7a" --learning_rate=0.0001 --uncertainty_type='none' --logs_base_dir='/mas/u/asma_gh/uncnet/unc_logs/'