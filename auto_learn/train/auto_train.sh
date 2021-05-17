#!/bin/bash
for i in {1..10}
do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 20001 train_no_freeze.py --version=$i
    echo "start find new samples"
    #fellow by annotations path and checkpoints path
    python ../find_new.py $[i+1] "/home/jovyan/data-vol-polefs-1/small_sample/dataset/annotations/no_freeze_200" "/home/jovyan/data-vol-polefs-1/small_sample/checkpoints/no_freeze_200"
done