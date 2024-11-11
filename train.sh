python main_tree.py \
    --epochs 8 \
    --log_dir ./runs/dinov2_vitb_exp \
    --dataset CUB-200-2011 \
    --batch_size 128 \
    --lr 0.001 \
    --lr_block 0.001 \
    --lr_net 1e-4 \
    --num_features 256 \
    --depth 9 \
    --net dinov2_vitb_exp \
    --freeze_epochs 1 \
    --milestones 3,6

python main_tree.py \
    --epochs 8 \
    --log_dir ./runs/dinov2_vits_exp \
    --dataset CUB-200-2011 \
    --batch_size 128 \
    --lr 0.001 \
    --lr_block 0.001 \
    --lr_net 1e-4 \
    --num_features 256 \
    --depth 9 \
    --net dinov2_vits_exp \
    --freeze_epochs 1 \
    --milestones 3,6

python main_tree.py \
    --epochs 8 \
    --log_dir ./runs/dino_vitb16 \
    --dataset CUB-200-2011 \
    --batch_size 128 \
    --lr 0.001 \
    --lr_block 0.001 \
    --lr_net 1e-4 \
    --num_features 256 \
    --depth 9 \
    --net dino_vitb16 \
    --freeze_epochs 1 \
    --milestones 3,6
