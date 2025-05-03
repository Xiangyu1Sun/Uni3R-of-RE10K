# Training
CUDA_VISIBLE_DEVICES=3 python -m src.main \
+experiment=re10k \
wandb.mode=offline \
wandb.name=re10k

# Train on 2 views first and then train on 4 views
CUDA_VISIBLE_DEVICES=3 python -m src.main \
+experiment=re10k_4view \
wandb.mode=offline \
wandb.name=re10k 

# Train on 4 views first and then train on 8 views
CUDA_VISIBLE_DEVICES=3 python -m src.main \
+experiment=re10k_8view \
wandb.mode=offline \
wandb.name=re10k 

# Evaluation
# RealEstate10K
CUDA_VISIBLE_DEVICES=3 python -m src.main \
+experiment=re10k_4view \
mode=test \
wandb.name=re10k \
wandb.mode=offline \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k_4v.json \
checkpointing.load=./pretrained_weights/vggtsplat_4views.pth \
test.save_image=false

# ACID
CUDA_VISIBLE_DEVICES=2 python -m src.main \
+experiment=acid \
mode=test \
wandb.name=acid \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
checkpointing.load=./pretrained_weights/acid.ckpt \
test.save_image=true

# Scannet++
CUDA_VISIBLE_DEVICES=3 python -m src.main \
+experiment=re10k_4view \
mode=test \
wandb.name=scannetpp \
wandb.mode=offline \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_scannet_4v.json \
dataset.re10k.roots=[/workspace/hdd/xiangyu/datasets/scannetpp] \
dataset.re10k.skip_bad_shape=false \
checkpointing.load=./pretrained_weights/vggtsplat_4views.pth \
test.save_image=true