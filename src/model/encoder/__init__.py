from typing import Optional

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .encoder_vggtsplat import EncoderVggtSplatCfg, EncoderVggtSplat

ENCODERS = {
    "vggt": (EncoderVggtSplat, None) ,
}

EncoderCfg = EncoderVggtSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
