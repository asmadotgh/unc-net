#!/usr/bin/env bash

embedding_models='VGGFace2_Inception_ResNet_v1 CASIA_WebFace_Inception_ResNet_v1'
embedding_layers='Mixed_8b Mixed_8a Mixed_7a Mixed_6b Mixed_6a Mixed_5a'
available_gpu_list=(0 1 2 3 4 5 6 7)

# Done
# matlaber 11
#embedding_models='CASIA_WebFace_Inception_ResNet_v1'
#embedding_layers='Mixed_8b Mixed_8a Mixed_7a Mixed_6b'
#available_gpu_list=(0 1 2 3)

# matlaber 10
#embedding_models='CASIA_WebFace_Inception_ResNet_v1'
#embedding_layers='Mixed_6a Mixed_5a'
#available_gpu_list=(2 3)

#running
# matlaber 10
# CUDA_VISIBLE_DEVICES=2 python emotion_classifier.py --embedding_model='CASIA_WebFace_Inception_ResNet_v1' --embedding_layer='Mixed_7a' --single_label --uncertainty_type='epistemic'

# matlaber 6
# CUDA_VISIBLE_DEVICES=3 python emotion_classifier.py --embedding_model='CASIA_WebFace_Inception_ResNet_v1' --embedding_layer='Mixed_7a' --single_label --uncertainty_type='epistemic' --learning_rate=0.00001

embedding_model_list=( $embedding_models )
embedding_layer_list=( $embedding_layers )

num_models=${#embedding_model_list[@]}
num_layers=${#embedding_layer_list[@]}
num_available_gpus=${#available_gpu_list[@]}

gpu_id=0
for embedding_model in $embedding_models
do
    for embedding_layer in $embedding_layers
    do
        curr_gpu=${available_gpu_list[$gpu_id % $num_available_gpus]}
        if (($gpu_id == $((num_models*num_layers-1)) ))
        then
            echo "Running... CUDA_VISIBLE_DEVICES=$(($gpu_id % 4)) python emotion_classifier.py --embedding_model=$embedding_model --embedding_layer=$embedding_layer --single_label "
            CUDA_VISIBLE_DEVICES=$(($gpu_id % 4)) python emotion_classifier.py --embedding_model="$embedding_model" --embedding_layer="$embedding_layer" --single_label
        else
            echo "Running... CUDA_VISIBLE_DEVICES=$(($gpu_id % 4)) python emotion_classifier.py --embedding_model=$embedding_model --embedding_layer=$embedding_layer --single_label &"
            CUDA_VISIBLE_DEVICES=$(($gpu_id % 4)) python emotion_classifier.py --embedding_model="$embedding_model" --embedding_layer="$embedding_layer" --single_label &
        fi
        ((gpu_id++))
    done
done
