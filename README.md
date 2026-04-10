# Self Implementation of RE10K datasets training and cross-domain datasets evaluation of Uni3R

The model is trained only on RE10K datasets and then do in-domian and cross-domain datasets evaluation.

Downloading vggt model checkpoint from https://github.com/facebookresearch/vggt. 
Then put the pretrained vggt model under the category of ./pretrained_weights/model.pt

# Training details

Please follow the Vicasplat training, 2views -> 4views -> 8views. (https://github.com/WU-CVGL/VicaSplat)

# Training on re10k
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.main \
+experiment=re10k_2view \
wandb.mode=online \
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
