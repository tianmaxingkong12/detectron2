import os
import time

from detectron2.config import LazyCall as L
from detectron2.evaluation import COCOEvaluator
from detrex.config import get_config

from ..common.models.detr_r50 import model
from ..common.data.Lesion4K_detr import dataloader

## model, dataloader, train, lr_multiplier, optimizer
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

experiment_id = time.strftime("%Y%m%d-%H%M%S",time.localtime())
output_dir = os.path.join("./projects/LesionDetection/logs", os.path.basename(__file__).split(".")[0], experiment_id)

### modify model config
model.num_classes = 16
model.criterion.num_classes = 16
model.num_queries=100

# modify dataloader config
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir=output_dir,
    use_fast_impl=False,
)

# modify training config
train.output_dir = output_dir
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
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


# modify lr_multiplier
lr_multiplier.scheduler.milestones = [80000, 100000]

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

