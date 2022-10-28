from mmcv.utils import Registry
from mmcv.cnn import MODELS as MMCV_MODELS

MODELS = Registry('models', parent=MMCV_MODELS)

VIS_ENCODERS = MODELS
LAN_ENCODERS = MODELS
FUSIONS = MODELS
HEADS = MODELS


def build_vis_enc(cfg):
    """Build vis_enc."""
    return VIS_ENCODERS.build(cfg)


def build_lan_enc(cfg, default_args):
    """Build lan_enc."""
    return LAN_ENCODERS.build(cfg, default_args=default_args)


def build_fusion(cfg):
    """Build lad_conv_list."""
    return FUSIONS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_model(cfg, word_emb=None, num_token=-1):
    """Build model."""
    model = MODELS.build(cfg, default_args=dict(
        word_emb=word_emb, num_token=num_token))

    return model
