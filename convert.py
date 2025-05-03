import torch

checkpoint = torch.load("./epoch_1-step_40000.ckpt", map_location="cpu")

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

torch.save(state_dict, "vggtsplat_4views.pth")

print("state_dict 提取并保存成功。")
