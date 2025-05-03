from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians

from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
# from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
# from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

# from ...global_cfg import get_cfg

# from .epipolar.epipolar_sampler import EpipolarSampler
# from ..encodings.positional_encoding import PositionalEncoding

from .vggt.aggregator import Aggregator
from .vggt.heads.dpt_head import DPTHead
from .vggt.heads.dpt_head_gs import DPTHeadGS
from .vggt.heads.camera_head import CameraHead
from .vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from .vggt.layers import PatchEmbed
from .vggt.ttt import TTT_Layer

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderVggtSplatCfg:
    name: Literal["vggt"]
    img_size: int
    patch_size: int
    embed_dim: int
    intrinsics_embed_type: Literal["pixelwise", "linear", "token"]
    downscale_factor: int
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    more_view_training: bool = False
    pose_free: bool = True

def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.LayerNorm)):
        nn.init.normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class EncoderVggtSplat(Encoder[EncoderVggtSplatCfg]):
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderVggtSplatCfg) -> None:
        super().__init__(cfg)
        
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        self.dim_end = 2*cfg.embed_dim
        self.intrinsics_embed_type = cfg.intrinsics_embed_type

        self.aggregator = Aggregator(img_size=cfg.img_size, patch_size=cfg.patch_size, embed_dim=cfg.embed_dim, intrinsics_embed_type=cfg.intrinsics_embed_type)
        # self.vggt_aggregator = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

        # inject images rgb to gaussian parameters head
        # TODO: figure out NoPosplat injection of RGB images and modified
        # self.input_merger = nn.Sequential(
        #     nn.Conv2d(3, 256, 7, 1, 3),
        #     # nn.ReLU(),
        #     # nn.Conv2d(256, 2048, 7, 1, 3),
        # )
        # self.input_merger = PatchEmbed(img_size=cfg.img_size, patch_size=cfg.patch_size, in_chans=3, embed_dim=self.dim_end)

        # # new DPT-Head for 3D Gaussians
        self.point_head = DPTHead(dim_in=2 * cfg.embed_dim, patch_size=cfg.patch_size, output_dim=4, activation="inv_log", conf_activation="expp1")
        # self.camera_head = CameraHead(dim_in=2 * cfg.embed_dim)

        # attributes sequence --> opacity, scale, rotation, sh
        self.token_decoder = DPTHeadGS(dim_in=2 * cfg.embed_dim, patch_size=cfg.patch_size, \
                                     output_dim=1 + 3 + 4 + (1+self.cfg.gaussian_adapter.sh_degree) ** 2 * 3, \
                                     feature_only=True)
        # self.token_decoder = nn.Sequential(
        #     nn.LayerNorm(self.dim_end, bias=False),
        #     nn.Linear(
        #         self.dim_end,
        #         (1 + 3 + 4 + (1+self.cfg.gaussian_adapter.sh_degree) ** 2 * 3) * cfg.patch_size ** 2,
        #         bias=False,
        #     )
        # )
        # self.token_decoder.apply(_init_weights)

        # # TTT module here TODO:
        # self.intermediate_layer_idx = [4, 11, 17, 23]
        # self.ttt_nums = len(self.intermediate_layer_idx)
        # self.ttt_layers = nn.ModuleList(
        #     [
        #         TTT_Layer(2*cfg.embed_dim, 2*cfg.embed_dim)
        #         for _ in range(self.ttt_nums)
        #     ]
        # )

        # gaussians convertor
        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        assert h // self.patch_size
        assert w // self.patch_size
        hh = h // self.patch_size
        ww = w // self.patch_size

        images = context["image"]
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        aggregated_tokens_list, patch_start_idx = self.aggregator.forward(images, context["intrinsics"])
        if self.intrinsics_embed_type == 'token':
            for i in range(len(aggregated_tokens_list)):
                aggregated_tokens_list[i] = aggregated_tokens_list[i][:,:,:-1,:]

        # tokens = aggregated_tokens_list[-1][:,:,patch_start_idx:,:]
        # inject_imgs = self.input_merger(images.view(b*v, 3, h, w)).view(b,v,-1,self.dim_end)
        # # inject_imgs = self.embed_merger(inject_imgs).view(b,v,-1,self.dim_end)
        # tokens += inject_imgs
        # # tokens += torch.nn.functional.relu(inject_imgs)
        # gaussians = self.token_decoder(tokens)

        # # TTT layer for different views token optimize
        # if aggregated_tokens_list[0].requires_grad:
        #     for i in range(self.ttt_nums):
        #         output_tokens = self.ttt_layers[i](aggregated_tokens_list[self.intermediate_layer_idx[i]])
        #         # print('the index', self.intermediate_layer_idx[i])
        #         aggregated_tokens_list[self.intermediate_layer_idx[i]] = output_tokens

        gaussians = self.token_decoder(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)

        # gaussians = rearrange(
        #     gaussians, "b v (hh ww) (ph pw) d -> b v (hh ph ww pw) d",
        #     v=v, hh=hh, ww=ww, ph=self.patch_size, pw=self.patch_size
        # )

        gaussians = rearrange(gaussians, "b v d h w -> b v (h w) d")

        # # Point DPT head to predict 3d points position and confidence
        if self.point_head is not None:
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )

        pts_all = rearrange(pts3d, "b v h w d -> b v (h w) d")
        pts_all = pts_all.unsqueeze(-2)

        depths = pts_all[..., -1].unsqueeze(-1)

        gaussians = gaussians.unsqueeze(-2)
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

        # # Camera DPT head to predict 3d points position and confidence
        # if self.camera_head is not None:
        #     pose_enc_list = self.camera_head(aggregated_tokens_list)

        # Convert the features and depths into Gaussians.
        if self.pose_free:
            gaussians = self.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                images,
            )
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            xy_ray = repeat(xy_ray, "s d xy -> b v s d xy", b=b, v=v)
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"].float(), "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"].float(), "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                densities,
                rearrange(
                    rotation,
                    "b v r c -> b v r () () c",
                ),
                rearrange(
                    scale,
                    "b v r c -> b v r () () c",
                ),
                rearrange(
                    feature,
                    "b v r c -> b v r () () c",
                ),
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1.0

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch
        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
