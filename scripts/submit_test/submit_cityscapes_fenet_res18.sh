#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
python3 eval.py \
    --dataset cityscapes \
    --arch network.fenet_resnet_conv_2b_2d.DeepR18_FE_deeply_completely \
    --inference_mode whole \
    --scales 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
    --split test \
    --cv_split 0 \
    --dump_images \
    --ckpt_path ${2} \
    --snapshot ${1}