_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_panoptic.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='PanopticFPN',
    backbone=dict(
        # type='ResNet',
        # depth=50,
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        # norm_cfg=dict(type='BN', requires_grad=True),
        # norm_eval=True,
        # style='pytorch',
        # with_cp=True,
        dcn=dict( #在最后三个block加入可变形卷积 
			# modulated=False, 
            # type='DCNv2',
            # type='DCN_AS',
            # type='FreqDecomp_DCNv2',
            type='AdaDilatedConv',
            epsilon=1e-4,
            use_zero_dilation=False,
            offset_freq=None,
            # offset_freq='SLP_res',
            deformable_groups=1, 
            padding_mode='zero',
            kernel_decompose='both',
            # kernel_decompose=None,
            # pre_fs=False,
            pre_fs=True,
            # conv_type='multifreqband',
            conv_type='conv',
            # fs_cfg=None,
            fs_cfg={
                # 'k_list':[3,5,7,9],
                'k_list':[2,4,8],
                'fs_feat':'feat',
                'lowfreq_att':False,
                # 'lp_type':'freq_eca',
                # 'lp_type':'freq_channel_att',
                # 'lp_type':'freq',
                # 'lp_type':'avgpool',
                'lp_type':'laplacian',
                'act':'sigmoid',
                'spatial':'conv',
                'channel_res':True,
                'spatial_group':1,
            },
            sp_att=False,
            fallback_on_stride=False, 
            ),
        stage_with_dcn=(False, True, True, True),
    ),
    # neck=dict(
    #     # type='FreqFusionFPN',
    #     type='FreqFusionCARAFEFPN',
    #     use_high_pass=True,
    #     use_low_pass=True,
    #     lowpass_kernel=5,
    #     highpass_kernel=3,
    #     compress_ratio=8,
    #     comp_feat_upsample=True,
    #     feature_align=True,
    #     semi_conv=True,
    #     use_dyedgeconv=False,
    #     feature_align_group=4,
    # ),
    semantic_head=dict(
        type='PanopticFPNHead',
        num_things_classes=80,
        num_stuff_classes=53,
        in_channels=256,
        inner_channels=128,
        start_level=0,
        end_level=4,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=None,
        loss_seg=dict(
            type='CrossEntropyLoss', ignore_index=255, loss_weight=0.5)),
    panoptic_fusion_head=dict(
        type='HeuristicFusionHead',
        num_things_classes=80,
        num_stuff_classes=53),
    test_cfg=dict(
        panoptic=dict(
            score_thr=0.6,
            max_per_img=100,
            mask_thr_binary=0.5,
            mask_overlap=0.5,
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
            stuff_area_limit=4096)))

custom_hooks = []
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64), ### for carafe
    dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
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
            dict(type='Pad', size_divisor=64), ### for carafe
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))