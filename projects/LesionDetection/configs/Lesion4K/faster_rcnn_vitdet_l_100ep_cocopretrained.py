import os
import time
from .faster_rcnn_vitdet_b import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

experiment_id = time.strftime("%Y%m%d-%H%M%S",time.localtime())
train.output_dir = os.path.join("./projects/LesionDetection/logs", os.path.basename(__file__).split(".")[0], experiment_id)
## 预训练权重
train.init_checkpoint = (
    "projects/LesionDetection/offical_ckpt/detectron2/model_final_6146ed.pkl"
)


# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 50000
lr_multiplier.scheduler.milestones = [35000, 45000]

# Optimizer
optimizer.lr = 0.00001
