## stage1
bash ./tools/dist_test.sh \
    projects/configs/BridgeAD_small_stage1.py \
    work_dirs/your_path.pth \
    8 \
    --deterministic \
    --eval bbox

## stage2
bash ./tools/dist_test.sh \
    projects/configs/BridgeAD_small_stage2.py \
    work_dirs/your_path.pth \
    8 \
    --deterministic \
    --eval bbox
