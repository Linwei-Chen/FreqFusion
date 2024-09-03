_base_ = [
'../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)

model = dict(
    # type='FasterRCNN',
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        # type='ResNet',
        # depth=50,
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        # norm_cfg=dict(type='BN', requires_grad=True),
        # norm_eval=True,
        # style='pytorch',
        with_cp=True,
        # frozen_stages=0,
        # dcn=dict( 
            #在最后三个block加入可变形卷积 
            # type='DCNv2',
            # type='DCN_AS',
            # radius=3,
            # type='DCN_AS_DFDC',
            # type='DFDC_Conv2D',
            # type='ODFDC_Conv2D',
            # deformable_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(True, True, True, True),
    ),
        neck=dict(
        # type='FreqFusionFPN',
        type='FreqFusionCARAFEFPN',
        use_high_pass=True,
        use_low_pass=True,
        lowpass_kernel=5,
        highpass_kernel=3,
        compress_ratio=8,
        feature_align=True,
        feature_align_group=8,
        semi_conv=True,
    #     type='FaPNOri',
    #     deform_groups=8,
    #     use_adaptive_sampling=False,
    #     radius=3,
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64), ###
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
