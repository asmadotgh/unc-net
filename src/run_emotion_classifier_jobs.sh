#!/usr/bin/env bash

embedding_models='VGGFace2_Inception_ResNet_v1 CASIA_WebFace_Inception_ResNet_v1'
embedding_layers='Mixed_8b Mixed_8a Mixed_7a Mixed_6b Mixed_6a Mixed_5a'

embedding_model_list=( $embedding_models )
embedding_layer_list=( $embedding_layers )

num_models=${#embedding_model_list[@]}
num_layers=${#embedding_layer_list[@]}

gpu_id=0
for embedding_model in embedding_models
do
    for embedding_layer in embedding_layers
    do
        if (($gpu_id == $((num_models*num_layers-1)) ))
        then
            CUDA_VISIBLE_DEVICES=$(($gpu_id % 4)) python emotion_classifier.py --embedding_model="$embedding_model" --embedding_layer="$embedding_layer"
        else
            CUDA_VISIBLE_DEVICES=$(($gpu_id % 4)) python emotion_classifier.py --embedding_model="$embedding_model" --embedding_layer="$embedding_layer" &
        fi
        ((gpu_id++))
    done
done
