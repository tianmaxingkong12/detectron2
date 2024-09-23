import os
import time

from functools import partial
from detectron2.config import LazyCall as L
from detectron2.evaluation import COCOEvaluator
from detrex.config import get_config
from detrex.modeling.backbone.eva import get_vit_lr_decay_rate

from ..common.models.dino_eva_01 import model
from ..common.data.Lesion4K_detr import dataloader

# get default config
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

experiment_id = time.strftime("%Y%m%d-%H%M%S",time.localtime())
output_dir = os.path.join("./projects/LesionDetection/logs", os.path.basename(__file__).split(".")[0], experiment_id)

# modify model config
model.num_classes = 16
model.backbone.net.beit_like_qkv_bias = True
model.backbone.net.beit_like_gamma = False
model.backbone.net.freeze_patch_embed = True
model.backbone.square_pad = 1280
model.backbone.net.img_size = 1280
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 1408
model.backbone.net.depth = 40
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 6144 / 1408
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.6  # 0.5 --> 0.6
# global attention for every 4 blocks
model.backbone.net.window_block_indexes = (
    list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) + list(range(12, 15)) + list(range(16, 19)) +
    list(range(20, 23)) + list(range(24, 27)) + list(range(28, 31)) + list(range(32, 35)) + list(range(36, 39))
)

# modify training config
train.output_dir = output_dir
train.init_checkpoint = "projects/LesionDetection/offical_ckpt/detr/dino_eva_01_o365_finetune_detr_like_augmentation_4scale_12ep.pth"
train.amp.enabled = True
train.ddp.fp16_compression = True
train.eval_period = 1000
train.log_period = 20
train.checkpointer.max_to_keep=10

# max training iterations
train.max_iter = 90000


# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

### wandb
train.wandb.enabled = True
train.wandb.params.dir = "projects/LesionDetection/wandb"
train.wandb.params.project = "detrex"
train.wandb.params.name = os.path.basename(__file__).split(".")[0]
train.wandb.params.notes = experiment_id

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.9, num_layers=40)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

# modify dataloader config
dataloader.train.num_workers = 4

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 4
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir=output_dir,
    use_fast_impl=False,
)
