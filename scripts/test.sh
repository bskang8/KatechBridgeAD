## stage1
bash ./tools/dist_test.sh \
    projects/configs/BridgeAD_small_stage1.py \
    /home/BridgeAD/ckpt/resnet50-19c8e357.pth \
    4 \
    --deterministic \
    --eval bbox

## stage2
bash ./tools/dist_test.sh \
    projects/configs/BridgeAD_small_stage2.py \
    /home/BridgeAD/ckpt/resnet50-19c8e357.pth \
    4 \
    --deterministic \
    --eval bbox
