# NoPoVggtSplat

先下载vggt的pretrained weights，然后存储到./pretrained_weights/model.pt

该版本仅在re10k下训练

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k_1x8 wandb.mode=offline wandb.name=re10k
