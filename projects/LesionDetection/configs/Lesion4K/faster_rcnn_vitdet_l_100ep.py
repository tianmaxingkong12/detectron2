import os
import time
from functools import partial

from .faster_rcnn_vitdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model, 
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.output_dir = os.path.join("./logs", os.path.basename(__file__).split(".")[0], time.strftime("%Y%m%d-%H%M%S",time.localtime()))
train.amp.enabled = True
train.ddp.fp16_compression = True
dataloader.train.total_batch_size = 4
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_large.pth?matching_heuristics=True"
)

model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.drop_path_rate = 0.4
# 5, 11, 17, 23 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
)

optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24)
