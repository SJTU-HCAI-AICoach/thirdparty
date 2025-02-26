
#configs/mmdet/detection/detection_tensorrt-fp16_static-640x640.py \

python tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-640x640.py \
    yolox/yolox_x_8xb8-300e_humanart.py \
    ../yolox_x_8xb8-300e_humanart-a39d44ed.pth \
    demo/test.jpg \
    --work-dir ./ \
    --device cuda:0 \
    --show \
    --dump-info  # dump sdk info
