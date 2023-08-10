# by zwz
from detectron2.utils.registry import Registry

PROMPT_ENCODER_REGISTRY = Registry("PROMPT_ENCODER")
PROMPT_ENCODER_REGISTRY.__doc__ = """
Registry for modules that creates prompt encoding for prompts.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def build_prompt_encoder(cfg, input_shape=None):
    """
    Built an anchor generator from `cfg.MODEL.PROMPT_ENCODER.NAME`.
    """
    anchor_generator = cfg.MODEL.PROMPT_ENCODER.NAME
    return PROMPT_ENCODER_REGISTRY.get(anchor_generator)(cfg, input_shape)
