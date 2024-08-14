import os
import time
from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.evaluation import COCOEvaluator

from ..common.data_loader_lsj import dataloader
from ..common.models.faster_rcnn_vitdet import model
from ..common.train import train

## TODO 修改模型
# model = model_zoo.get_config("../projects/LesionDetection/common/models/faster_rcnn_vitdet.py").model
model.roi_heads.num_classes = 16
# Initialization and trainer settings
# train = model_zoo.get_config("../projects/LesionDetection/common/train.py").train

train.output_dir = os.path.join("./logs", os.path.basename(__file__).split(".")[0], time.strftime("%Y%m%d-%H%M%S",time.localtime()))
train.amp.enabled = False
train.ddp.fp16_compression = False

dataloader.train.total_batch_size = 4
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir=train.output_dir,
    use_fast_impl=False,
)

## TODO 修改权重
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)


# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 184375

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
