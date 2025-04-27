# Training
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k_1x8 wandb.mode=offline wandb.name=re10k

# Evaluation

# RealEstate10K
CUDA_VISIBLE_DEVICES=0 python -m src.main \
+experiment=re10k_1x8 \
mode=test \
wandb.name=re10k \
wandb.mode=offline \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k_test.json \
checkpointing.load=./pretrained_weights/epoch_2-step_300000.ckpt \
test.save_image=true

# ACID
CUDA_VISIBLE_DEVICES=2 python -m src.main \
+experiment=acid \
mode=test \
wandb.name=acid \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
checkpointing.load=./pretrained_weights/acid.ckpt \
test.save_image=true