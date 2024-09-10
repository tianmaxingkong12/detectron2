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

experiment_id = time.strftime("%Y%m%d-%H%M%S",time.localtime())
model.roi_heads.num_classes = 16

train.output_dir = os.path.join("./projects/LesionDetection/logs", os.path.basename(__file__).split(".")[0], experiment_id)
train.amp.enabled = False
train.ddp.fp16_compression = False
train.eval_period = 1000

dataloader.train.total_batch_size = 2
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir=train.output_dir,
    use_fast_impl=False,
)

train.checkpointer.max_to_keep=10

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 114700

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[101955, 110451],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
