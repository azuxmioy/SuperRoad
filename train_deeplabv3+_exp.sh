#!/usr/bin/env bash
python train_deeplabv3+_exp.py \
            --experiment_name deeplabv3+_lowlr_newsampling2_wclass_tv_final_debug \
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
            --encoder_model ./experiments/deeplabv3+_external_midlr_newsampling2_wclass_tv/model/encoder.ckpt-10000 \
            --decoder_model ./experiments/deeplabv3+_external_midlr_newsampling2_wclass_tv/model/decoder.ckpt-10000 \
            --cil_mode final
