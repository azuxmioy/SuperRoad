#!/usr/bin/env bash
python train_deeplabv3+_exp.py \
            --experiment_name deeplabv3+_external_midlr_newsampling2_wclass_tv \
            --use_external True \
            --save_path ./experiments/ \
            --save_soft_masks True \
            --batch_size 6 \
            --f_lr 0.00001 \
            --d_lr 0.00001 \
            --weight_jac 0.2 \
            --weight_tv 0.00000005 \
            --loss wclass \
            --scale_factor 1.8 \
            --output_stride 8 \
            --feature xception \
            --tf_initial_checkpoint xception/xception71/model.ckpt
