_base_ = [
    '../../_base_/models/mscan.py',
    '../../_base_/datasets/ade20k_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        use_checkpoint=True,
        init_cfg=dict(type='Pretrained', checkpoint='/home/ubuntu/code/ResolutionDet/SegNeXt/pretrained/mscan_t.pth')),
    decode_head=dict(
        type='LightHamHeadFreqAware',
        use_checkpoint=False,
        compress_ratio=4,
        feature_resample_group=4,
        feature_resample=True,
        comp_feat_upsample=True,
        hamming_window=False,
        semi_conv=True,
        use_high_pass=True, 
        use_low_pass=True,
        lowpass_kernel=5,
        highpass_kernel=3,

        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        ham_kwargs=dict(MD_R=16),
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=16, workers_per_gpu=16)
checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=2)
evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU', pre_eval='True')
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 ######
                                                 'upsampler': dict(lr_mult=10.),
                                                 'comp_conv': dict(lr_mult=10.),
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
