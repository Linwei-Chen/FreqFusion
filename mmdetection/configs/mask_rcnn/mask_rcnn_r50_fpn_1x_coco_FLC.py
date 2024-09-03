_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)

model = dict(
    # type='FasterRCNN',
    backbone=dict(
        # type='ResNet',
        type='ResNetFLC',
        lateral_hf=True,
        depth=50,
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        # norm_cfg=dict(type='BN', requires_grad=True),
        # norm_eval=True,
        # style='pytorch',
        # with_cp=True,
    )
)