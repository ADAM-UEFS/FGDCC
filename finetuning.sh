#!/bin/bash

#touch run/experiment.log

export IMAGENET_DIR='/home/rtcalumby/adam/luciano/LifeCLEFPlant2022/'
export PRETRAIN_CHKPT='../pretrained/mae_pretrain_vit_large.pth'
export name="IN1k_Clef2022"
export all_epoch=100
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#OMP_NUM_THREADS=1 python3 -m torch.distributed.run --nproc_per_node=6 main_finetune.py \
#    --accum_iter 2 \
#    --batch_size 128 \
#    --model vit_large_patch16  \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${all_epoch} \
#    --blr 7.5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --log_dir "checkpoint/${name}/log" \
#    --nb_classes 80000 \
#    --resume "checkpoint/${name}/checkpoint-${resume_epoch}.pth" --start_epoch ${resume_epoch} \
#    --dist_eval \
#    --output_dir checkpoint/${name} \
#    --num_workers 20  # default: 20

OMP_NUM_THREADS=1 python3 -m torch.distributed.run --nproc_per_node=8 main_finetune.py \
    --accum_iter 4 \
    --batch_size 128 \
    --model vit_large_patch16  \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs ${all_epoch} \
    --blr 7.5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --log_dir "checkpoint/${name}/log" \
    --nb_classes 80000 \
    --output_dir "checkpoint/${name}" \
    --num_workers 64  # default: 20 


