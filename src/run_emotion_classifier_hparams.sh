#!/usr/bin/env bash

#embedding_models='VGGFace2_Inception_ResNet_v1 CASIA_WebFace_Inception_ResNet_v1'
#embedding_layers='Mixed_8b Mixed_8a Mixed_7a Mixed_6b Mixed_6a Mixed_5a'
embedding_models='VGGFace2_Inception_ResNet_v1'
embedding_layers='Mixed_5a'

embedding_model_list=( $embedding_models )
embedding_layer_list=( $embedding_layers )

learning_rate_list=(0.0001 0.001 0.01 0.1 0.5)
available_gpu_list=(2 6 7)

num_models=${#embedding_model_list[@]}
num_layers=${#embedding_layer_list[@]}
num_learning_rates=${#learning_rate_list[@]}
num_available_gpus=${#available_gpu_list[@]}


gpu_id=0
for embedding_model in $embedding_models
do
    for embedding_layer in $embedding_layers
    do
        for learning_rate in "${learning_rate_list[@]}"
        do
            if (($gpu_id == $((num_models*num_layers*num_learning_rates-1)) ))
            then
                echo "Running... CUDA_VISIBLE_DEVICES=$((available_gpus[$gpu_id] % $num_available_gpus)) python emotion_classifier.py --embedding_model=$embedding_model --embedding_layer=$embedding_layer --learning_rate=$learning_rate"
                CUDA_VISIBLE_DEVICES=$(("$available_gpus[$gpu_id] % $num_available_gpus")) python emotion_classifier.py --embedding_model="$embedding_model" --embedding_layer="$embedding_layer" --learning_rate="$learning_rate"
            else
                echo "Running... CUDA_VISIBLE_DEVICES=$((available_gpus[$gpu_id] % $num_available_gpus)) python emotion_classifier.py --embedding_model=$embedding_model --embedding_layer=$embedding_layer --learning_rate=$learning_rate &"
                CUDA_VISIBLE_DEVICES=$(("$available_gpus[$gpu_id] % $num_available_gpus")) python emotion_classifier.py --embedding_model="$embedding_model" --embedding_layer="$embedding_layer" --learning_rate="$learning_rate" &
            fi
            ((gpu_id++))
        done
    done
done