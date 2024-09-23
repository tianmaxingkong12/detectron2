import os
import time

from detectron2.config import LazyCall as L
from detectron2.evaluation import COCOEvaluator
from detrex.config import get_config

from .detr_r50_100k import dataloader, lr_multiplier, optimizer, train
from ..common.models.detr_r50_dc5 import model

## model, dataloader, train, lr_multiplier, optimizer
experiment_id = time.strftime("%Y%m%d-%H%M%S",time.localtime())
output_dir = os.path.join("./projects/LesionDetection/logs", os.path.basename(__file__).split(".")[0], experiment_id)

# modify model config
model.num_classes = 16
model.criterion.num_classes = 16
model.num_queries=100

# modify dataloader config
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 4
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir=output_dir,
    use_fast_impl=False,
)

# modify training config
# using torchvision official checkpoint
# the urls can be found in: https://pytorch.org/vision/stable/models/resnet.html
train.output_dir = output_dir
train.init_checkpoint = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
train.amp.enabled = True
train.ddp.fp16_compression = True
train.eval_period = 1000
train.checkpointer.max_to_keep=10
train.max_iter = 100000
### wandb
train.wandb.enabled = True
train.wandb.params.dir = "projects/LesionDetection/wandb"
train.wandb.params.project = "detrex"
train.wandb.params.name = os.path.basename(__file__).split(".")[0]
train.wandb.params.notes = experiment_id


