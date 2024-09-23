import os
import time

from detectron2.config import LazyCall as L
from detectron2.evaluation import COCOEvaluator
from detrex.config import get_config

from .detr_r50_dc5_100k import model, dataloader, train, lr_multiplier, optimizer

## model, dataloader, train, lr_multiplier, optimizer
experiment_id = time.strftime("%Y%m%d-%H%M%S",time.localtime())
output_dir = os.path.join("./projects/LesionDetection/logs", os.path.basename(__file__).split(".")[0], experiment_id)

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
train.init_checkpoint = "projects/LesionDetection/offical_ckpt/detr/converted_detr_r50_dc5.pth?matching_heuristics=True"
### wandb
train.wandb.params.name = os.path.basename(__file__).split(".")[0]
train.wandb.params.notes = experiment_id


