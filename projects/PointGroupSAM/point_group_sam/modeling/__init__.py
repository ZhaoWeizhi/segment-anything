# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .sam import Sam
# 测试注册encoder临时代码
from .backbone.image_encoder import ImageEncoderViT
from .mask_decoder.mask_decoder import MaskDecoder
from .prompt_encoder.prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer
