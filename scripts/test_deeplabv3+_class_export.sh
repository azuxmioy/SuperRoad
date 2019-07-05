#!/usr/bin/env bash
python train_deeplabv3+_exp.py \
            --experiment_name deeplabv3+_class_export \
            --save_path ./experiments/ \
            --save_soft_masks True \
            --batch_size 6 \
            --f_lr 0.000005 \
            --d_lr 0.000005 \
            --weight_jac 0.2 \
            --weight_tv 0.0000001 \
            --loss wclass \
            --scale_factor 2.0 \
            --output_stride 8 \
            --feature xception \
            --finetune True \
            --encoder_model ./experiments/deeplabv3+_lowlr_newsampling2_class_final/model/encoder.ckpt-15000 \
            --decoder_model ./experiments/deeplabv3+_lowlr_newsampling2_class_final/model/decoder.ckpt-15000 \
            --cil_mode final \
            --mode test