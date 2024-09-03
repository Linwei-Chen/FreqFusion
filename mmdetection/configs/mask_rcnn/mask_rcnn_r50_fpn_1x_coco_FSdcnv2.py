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
        # depth=50,
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        # norm_cfg=dict(type='BN', requires_grad=True),
        # norm_eval=True,
        # style='pytorch',
        with_cp=True,
        dcn=dict( #在最后三个block加入可变形卷积 
			# modulated=False, 
            # type='DCNv2',
            # type='DCN_AS',
            type='FreqDecomp_DCNv2',
            # type='FreqDecomp_Conv',
            # type='FreqDecomp2_DCNv2',
            # radius=2, 
            # plus_center=False,
            deformable_groups=1, 
            fallback_on_stride=False, 
            # only_on_stride_conv1=True
            # fs_feat='offset_conv3',
            # fs_feat='offset_conv1',
            # att_type='simat',
            act='sigmoid',
            fs_feat='feat',
            # pre_att=True,
            # fs_feat='freq_feat_share',
            # k_list=[3, 5, 7, 9, 11],
            # lp_type='freq',
            lp_type='freq_channel_att',
            k_list=[8/1, 8/2, 8/3, 8/4, 8/5, 8/6, 8/7][::-1],
            # k_list=[2, 3, 4, 5, 6, 7],
            # k_list=[7/6, 3, 4, 5, 6, 7],
            # k_list=[3, 5, 7, 9, 11],
            channel_group=1,
            channel_bn=True,
            ),
        stage_with_dcn=(False, True, True, True),
    )
)