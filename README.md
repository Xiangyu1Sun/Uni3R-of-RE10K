# Self Implementation of Uni3R on RE10K datasets

Self Implementation of RE10K datasets training and cross-domain datasets evaluation of [Uni3R](https://github.com/HorizonRobotics/Uni3R).

This is the not the official code. The model is trained only on RE10K datasets and then do in-domian and cross-domain datasets evaluation.



## Dataset processing

For re10k dataset, please download from [Pixelsplat](https://github.com/dcharatan/pixelsplat).

And for scannet++ dataset, please follow [Vicasplat](https://github.com/WU-CVGL/VicaSplat).

## More training and evaluation details

First, download the vggt model [checkpoint](https://github.com/facebookresearch/vggt). 
Then put the pretrained vggt model under the category of ./pretrained_weights/model.pt.

please follow our command.sh file

````
# Training on re10k
python -m src.main \
+experiment=re10k_2view \
wandb.mode=online \
wandb.name=re10k

# Train on 2 views first and then train on 4 views
python -m src.main \
+experiment=re10k_4view \
wandb.mode=offline \
wandb.name=re10k 

# Train on 4 views first and then train on 8 views
python -m src.main \
+experiment=re10k_8view \
wandb.mode=offline \
wandb.name=re10k 
````

````
# Evaluation
# RealEstate10K 2views
CUDA_VISIBLE_DEVICES=3 python -m src.main \
+experiment=re10k \
mode=test \
wandb.name=re10k \
wandb.mode=offline \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
checkpointing.load=./pretrained_weights/vggtsplat_2views.pth \
test.save_image=false

# RealEstate10K 4views
CUDA_VISIBLE_DEVICES=3 python -m src.main \
+experiment=re10k_4view \
mode=test \
wandb.name=re10k \
wandb.mode=offline \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k_4v.json \
checkpointing.load=./pretrained_weights/vggtsplat_4views.pth \
test.save_image=false

# RealEstate10K 8views
CUDA_VISIBLE_DEVICES=3 python -m src.main \
+experiment=re10k_8view \
mode=test \
wandb.name=re10k \
wandb.mode=offline \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k_8v.json \
checkpointing.load=./pretrained_weights/vggtsplat_8views.pth \
test.save_image=false

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
````
