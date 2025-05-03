# NoPoVggtSplat

先下载vggt的pretrained weights，然后存储到./pretrained_weights/model.pt

该版本目前仅在re10k下训练也仅在re10k测试, 参考command.sh文件

推荐，先多用几组8gpus训练2views，然后测试2views的性能，挑一个性能好的进行4views和8views的训练，
该流程完全follow vicasplat的流程。
训练时候注意保持 total batchsize * iters 和 vicasplat 完全一致，公平对比。

我认为需要试的参数，optimizer中的lr和backbone_lr_multiplie
以及view_sampler的warm up steps，可能值小一点会更早开始训练大view range，效果会好点

后续，得到2，4，8views在re10k上训练的ckpts之后，可以在scannet上做一下zero shot generalization的测试，参考vicasplat的步骤（论文中的Tab.2）。