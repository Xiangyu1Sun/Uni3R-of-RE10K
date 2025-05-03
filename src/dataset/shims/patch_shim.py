from ..types import BatchedExample, BatchedViews
from torch import Tensor
import torch

def apply_patch_shim_to_views(views: BatchedViews, patch_size: int) -> BatchedViews:
    _, _, _, h, w = views["image"].shape

    # Image size must be even so that naive center-cropping does not cause misalignment.
    assert h % 2 == 0 and w % 2 == 0

    h_new = (h // patch_size) * patch_size
    row = (h - h_new) // 2
    w_new = (w // patch_size) * patch_size
    col = (w - w_new) // 2

    # Center-crop the image.
    image = views["image"][:, :, :, row : row + h_new, col : col + w_new]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = views["intrinsics"].clone()
    intrinsics[:, :, 0, 0] *= w / w_new  # fx
    intrinsics[:, :, 1, 1] *= h / h_new  # fy

    return {
        **views,
        "image": image,
        "intrinsics": intrinsics,
    }

def normalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean) / std

def apply_patch_shim(
    batch: BatchedExample, 
    patch_size: int, 
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> BatchedExample:
    """Crop images in the batch so that their dimensions are cleanly divisible by the
    specified patch size.
    """
    batch["context"]["image"] = normalize_image(batch["context"]["image"], mean, std)
    return {
        **batch,
        "context": apply_patch_shim_to_views(batch["context"], patch_size),
        "target": apply_patch_shim_to_views(batch["target"], patch_size),
    }
