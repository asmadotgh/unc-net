#!/usr/bin/env bash

learning_rate_list=(0.001 0.0001 0.00005 0.00001)
mc_n_list=(50 100 1000)
#mc_n_list=(50 100)
#mc_n_list=(1000)
available_gpu_list=(0 1 2 3 4 5 6 7)
#available_gpu_list=(4 5 6 7)

num_learning_rates=${#learning_rate_list[@]}
num_mc_ns=${#mc_n_list[@]}
num_available_gpus=${#available_gpu_list[@]}

gpu_id=0
for learning_rate in "${learning_rate_list[@]}"
do
    for mc_n in "${mc_n_list[@]}"
    do
        curr_gpu=${available_gpu_list[$gpu_id % $num_available_gpus]}
        if (($gpu_id == $((num_learning_rates*num_mc_ns-1)) ))
        then
            echo "Running... CUDA_VISIBLE_DEVICES=$curr_gpu python emotion_classifier.py --learning_rate=$learning_rate --n_epistemic=$mc_n --uncertainty_type=epistemic --embedding_model=CASIA_WebFace_Inception_ResNet_v1 --embedding_layer=Mixed_7a --logs_base_dir=/mas/u/asma_gh/uncnet/unc_hparam_logs/"
            CUDA_VISIBLE_DEVICES="$curr_gpu" python emotion_classifier.py --learning_rate="$learning_rate" --n_epistemic="$mc_n" --uncertainty_type="epistemic" --embedding_model="CASIA_WebFace_Inception_ResNet_v1" --embedding_layer="Mixed_7a" --logs_base_dir="/mas/u/asma_gh/uncnet/unc_hparam_logs/"
        else
            echo "Running... CUDA_VISIBLE_DEVICES=$curr_gpu python emotion_classifier.py --learning_rate=$learning_rate --n_epistemic=$mc_n --uncertainty_type=epistemic --embedding_model=CASIA_WebFace_Inception_ResNet_v1 --embedding_layer=Mixed_7a --logs_base_dir=/mas/u/asma_gh/uncnet/unc_hparam_logs/ & "
            CUDA_VISIBLE_DEVICES="$curr_gpu" python emotion_classifier.py --learning_rate="$learning_rate" --n_epistemic="$mc_n" --uncertainty_type="epistemic" --embedding_model="CASIA_WebFace_Inception_ResNet_v1" --embedding_layer="Mixed_7a" --logs_base_dir="/mas/u/asma_gh/uncnet/unc_hparam_logs/" &
        fi
        ((gpu_id++))
    done
done