import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from ...evaluation.evaluation_index_generator import IndexEntry
from ...global_cfg import get_cfg
from ...misc.step_tracker import StepTracker
from ..types import Stage
from .three_view_hack import add_third_context_index
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

        self.cfg = cfg

        dacite_config = Config(cast=[tuple])
        with cfg.index_path.open("r") as f:
            self.index = {
                k: None if v is None else from_dict(IndexEntry, v, dacite_config)
                for k, v in json.load(f).items()
            }

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        def sample_support_indices(support_indices, max_samples=50):
            num = support_indices.shape[0]
            if num > max_samples:
                # 生成 max_samples 个均匀间隔的整数索引
                lin_indices = torch.linspace(0, num - 1, steps=max_samples).long()
                support_indices = support_indices[lin_indices]
            return support_indices

        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)

        overlap = entry.overlap if isinstance(entry.overlap, float) else 0.75 if entry.overlap == "large" else 0.25
        overlap = torch.tensor([overlap], dtype=torch.float32, device=device)

        # Handle 2-view index for 3 views.
        v = self.num_context_views
        if v > len(context_indices) and v == 3:
            context_indices = add_third_context_index(context_indices)

        # support_indices = torch.arange(context_indices[0] + 1, context_indices[1])
        # mask = ~torch.isin(support_indices, target_indices)
        # support_indices = support_indices[mask]
        # support_indices = sample_support_indices(support_indices, max_samples=50)

        # import pdb; pdb.set_trace()

        return context_indices, target_indices, overlap #, support_indices

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return 0
