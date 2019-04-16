#!/usr/bin/env bash

# note that tensorflow=1.7.0 has been previously used to run this
# now tensorflow and tensorflow-gpu 1.12.0 are installed

#subsets='all train valid test'
subsets='all'
embeddings='Mixed_8b Mixed_8a Mixed_7a Mixed_6b Mixed_6a Mixed_5a'

subset_list=( $subsets )
embedding_list=( $embeddings )

num_subsets=${#subset_list[@]}
num_embeddings=${#embedding_list[@]}

gpu_id=0
for subset in $subsets
do
    for embedding in $embeddings
    do
        if (($gpu_id == $((num_subsets*num_embeddings-1)) ))
        then
            CUDA_VISIBLE_DEVICES=$(($gpu_id % 4)) python export_embeddings.py --embedding_name="$embedding" --data_dir="/mas/u/asma_gh/uncnet/datasets/FER+/$subset.csv"
        else
            CUDA_VISIBLE_DEVICES=$(($gpu_id % 4)) python export_embeddings.py --embedding_name="$embedding" --data_dir="/mas/u/asma_gh/uncnet/datasets/FER+/$subset.csv" &
        fi
        ((gpu_id++))
    done
done
