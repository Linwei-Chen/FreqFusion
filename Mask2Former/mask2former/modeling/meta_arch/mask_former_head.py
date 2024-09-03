# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder

import torch
from mmseg.ops import resize
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, ModulatedDeformConv2dPack
from mmcv.ops.carafe import CARAFEPack, xavier_init
from torch.utils.checkpoint import checkpoint

@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        # final feature
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions


class LHPFConv3(nn.Module):
    """
    Learnable High Pass Filter
    """
    def __init__(self, channels=3, stride=2, padding=1):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        kernel = [
            [1., 1., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            # [1/8., 1/8., 1/8.],
        ]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # in_ch, out_ch,
        kernel = kernel.repeat(self.channels, 1, 1, 1).contiguous()
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        identity_kernel = [
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ]
        identity_kernel = torch.FloatTensor(identity_kernel).unsqueeze(0).unsqueeze(0)
        # self.identity_kernel = identity_kernel
        self.identity_kernel = nn.Parameter(data=identity_kernel, requires_grad=False)
 
    def forward(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # self.device = x.device
        # self.identity_kernel = self.identity_kernel.to(x.device)
        x = F.conv2d(x, self.identity_kernel - self.weight.reshape(self.channels, 1, -1).softmax(-1).reshape(self.channels, 1, 3, 3), 
                    padding=self.padding, groups=self.channels, stride=self.stride)
        return x

class ImageGuideUpsample(CARAFEPack):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,          
                align_corners=False,
                upsample_mode='nearest',
                use_encoder2=False,
                use_spatial_suppression=False,
                use_spatial_residual=False,
                hr_residual=False,
                use_deformable=False,
                use_leanable_hpf=False,
                use_extra_conv3=False,
                **kwargs):
        super().__init__(**kwargs)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_spatial_residual = use_spatial_residual
        # self._do_nothing = nn.Conv2d(19, 1, 1)
        self.use_deformable = use_deformable
        self.use_leanable_hpf = use_leanable_hpf
        self.use_extra_conv3 = use_extra_conv3
        if use_deformable:
            del self.content_encoder
            self.content_encoder = ModulatedDeformConv2dPack(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                stride=1,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1,
                deform_groups=1)
        self.use_encoder2 = use_encoder2
        del self.channel_compressor
        assert upsample_mode == 'nearest'
        if self.use_encoder2:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        if self.use_spatial_suppression:
            self.ss = SpatialSuppression(kernel_size=7, align_corners=align_corners, upsample_mode=upsample_mode)
        if self.use_leanable_hpf:
            self.LPF = LHPFConv3(channels=3, stride=1, padding=1)
            if self.use_extra_conv3:
                self.extra_conv3 = nn.Conv2d(3, 1, kernel_size=3, padding=1, dilation=1, groups=1)
        self.init_weights()

    def _forward(self, feat, seg, img):
        if self.use_leanable_hpf:
            img_hf = self.LPF(img)
        else:
            lr_img = resize(
                input=img,
                size=feat.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            lr_img = resize(
                input=lr_img,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            img_hf = img - lr_img + img
        feat = resize(
            input=feat,
            size=img_hf.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        seg = resize(
            input=seg,
            size=img_hf.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # _ = self._do_nothing(seg)
        # feat_img = torch.cat([feat, img_hf], dim=1)
        # compressed_x = self.channel_compressor(feat)
        if self.use_extra_conv3:
            img_hf = self.extra_conv3(img_hf)
        mask = self.content_encoder(torch.cat([feat, img_hf], dim=1))
        mask = self.kernel_normalizer(mask)
        seg = self.feature_reassemble(seg, mask)
        feat = self.feature_reassemble(feat, mask)
        # print(seg.shape)
        return seg, feat

    def forward(self, feat, seg, img, use_checkpoint=True):
        if use_checkpoint:
            return checkpoint(self._forward, feat, seg, img)
        else:
            return self._forward(feat, seg, img)

class CascadeImageGuideUpsample(nn.Module):
    def __init__(self, n=2, **kargs):
        super().__init__()
        self.n = n
        self.align_corners = kargs.get('align_corners', False)
        self.up = nn.ModuleList()
        for _ in range(n):
            self.up.append(ImageGuideUpsample(**kargs))

    def forward(self, feat, seg, img):
        for i in range(self.n):
            resized_img = resize(
                input=img,
                size=(2 * feat.size(-2), 2 * feat.size(-1)),
                mode='bilinear',
                align_corners=self.align_corners)
            # print(resized_img.shape)
            seg, feat = self.up[i](feat, seg, resized_img)
            # print(seg.shape, feat.shape)
        return seg
    

@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHeadFreqAware(MaskFormerHead):
    @configurable
    def __init__(self, compress_channel=4, compress_k=3, **kwargs):
        super().__init__(**kwargs)
        self.IGU = CascadeImageGuideUpsample(  
            n=2,
            channels=compress_channel + 1,
            scale_factor=1,
            up_kernel=3, 
            compressed_channels=compress_channel + 1,

            align_corners=False,
            upsample_mode='nearest',
            use_encoder2=False,
            use_spatial_suppression=False,
            use_spatial_residual=False,
            hr_residual=False,
            use_deformable=False,
            use_leanable_hpf=True,
            use_extra_conv3=True
        )
        self.comp_conv = nn.Sequential(
                nn.Conv2d(256, compress_channel, kernel_size=compress_k, padding=compress_k // 2),
                # nn.SyncBatchNorm(compress_channel),
                # nn.ReLU(True),
                # nn.Conv2d(compress_channel, self.num_classes * self.patch_size ** 2, kernel_size=1),
            )
        xavier_init(self.comp_conv, distribution='uniform')

    def forward(self, features, mask=None, img=None):
        return self.layers(features, mask, img)
    
    def layers(self, features, mask=None, img=None):
        # final feature
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)

        '''
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        '''
        compressed_feat = checkpoint(self.comp_conv, mask_features)
        pred_masks = predictions['pred_masks']
        pred_masks = self.IGU(compressed_feat, pred_masks, img)
        predictions['pred_masks'] = pred_masks
        return predictions