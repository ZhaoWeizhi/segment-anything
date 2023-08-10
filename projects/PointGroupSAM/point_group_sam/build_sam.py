# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

# from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

# 进行注册encoder单元测试使用的临时代码
from .modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from detectron2.modeling import build_backbone
from detectron2.modeling import build_prompt_encoder
from detectron2.modeling import build_sem_seg_head
from detectron2.config import CfgNode as CN


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = "D2ImageEncoder"
    cfg.MODEL.IMAGE_ENCODER = CN()
    cfg.MODEL.IMAGE_ENCODER.DEPTH = encoder_depth
    cfg.MODEL.IMAGE_ENCODER.EMBED_DIM = encoder_embed_dim
    cfg.MODEL.IMAGE_ENCODER.NUM_HEADS = encoder_num_heads
    cfg.MODEL.IMAGE_ENCODER.GLOBAL_ATTN_INDEXES = encoder_global_attn_indexes

    cfg.MODEL.PROMPT_ENCODER = CN()
    cfg.MODEL.PROMPT_ENCODER.NAME = "D2PromptEncoder"
    cfg.MODEL.PROMPT_ENCODER.IMAGE_EMBEDDING_SIZE = image_embedding_size

    cfg.MODEL.SEM_SEG_HEAD = CN()
    cfg.MODEL.SEM_SEG_HEAD.NAME = "D2MaskDecoder"
    
    sam = Sam(
        image_encoder=build_backbone(cfg),
        prompt_encoder=build_prompt_encoder(cfg),
        mask_decoder=build_sem_seg_head(cfg),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam
