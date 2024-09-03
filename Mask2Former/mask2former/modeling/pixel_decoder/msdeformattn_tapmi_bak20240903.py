# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn


# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides) # 8
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride)) # log8 - log4

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        return ret

    @autocast(enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            # print(idx, x.shape)
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        # print('self.in_features', self.in_features)
        # print('self.num_fpn_levels', self.num_fpn_levels)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            # print(idx, x.shape)
            # print('self.lateral_convs', self.lateral_convs)
            # print('self.output_convs', self.output_convs)
            lateral_conv = self.lateral_convs[idx]
            # print('lateral_conv', lateral_conv)
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features

from mmcv.ops.carafe import CARAFEPack, xavier_init, normal_init, carafe
from torch.utils.checkpoint import checkpoint
class DyHPFusion2(CARAFEPack):
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
                use_high_pass=True,
                use_low_pass=True,
                hr_residual=True,
                **kwargs):
        super().__init__(**kwargs)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass

        self.use_encoder2 = use_encoder2
        if self.use_encoder2:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group *
                self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        if self.use_spatial_suppression:
            self.spatial_suppression = nn.Sequential(
                nn.Conv2d(self.compressed_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
        self.init_weights()

    # def forward(self, x):
        # compressed_x = self.channel_compressor(x)
        # mask = self.content_encoder(compressed_x)
        # mask = self.kernel_normalizer(mask)
        # x = self.feature_reassemble(x, mask)
        # return x

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        lr_feat = F.interpolate(lr_feat, hr_feat.shape[2:],
            mode=self.upsample_mode,
            align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # if self.use_spatial_suppression:
            # _, hr_feat, lr_feat = self.ss(hr_feat, lr_feat, residual=self.use_spatial_residual)
        x = torch.cat([hr_feat, lr_feat], dim=1)
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        if self.use_spatial_suppression:
            hr_feat = self.spatial_suppression(compressed_x) * hr_feat
        if self.use_high_pass:
            if self.hr_residual:
                # print('using hr_residual')
                hr_feat = hr_feat - self.feature_reassemble(hr_feat, mask) + hr_feat
            else:
                hr_feat = hr_feat - self.feature_reassemble(hr_feat, mask)
        if self.use_low_pass:
            if self.use_encoder2:
                mask2 = self.content_encoder2(compressed_x)
                mask2 = self.kernel_normalizer(mask2)
            else:
                mask2 = mask
            lr_feat = self.feature_reassemble(lr_feat, mask2)
        return mask, hr_feat, lr_feat

class FreqFusion(nn.Module):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,
                hr_channels,
                lr_channels,
                scale_factor,
                up_kernel=3,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64,        
                align_corners=False,
                upsample_mode='nearest',
                use_encoder2=False,
                use_spatial_suppression=False,
                use_spatial_residual=False,
                use_high_pass=True,
                use_low_pass=True,
                hr_residual=True,
                semi_conv=False,
                **kwargs):
        super().__init__()
        # self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels,1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels,1)

        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            self.up_kernel * self.up_kernel * self.up_group *
            self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv

        self.use_encoder2 = use_encoder2
        if self.use_encoder2:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group *
                self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        if self.use_spatial_suppression:
            self.spatial_suppression = nn.Sequential(
                nn.Conv2d(self.compressed_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_encoder2:
            normal_init(self.content_encoder, std=0.001)

    def kernel_normalizer(self, mask):
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        # use float division explicitly,
        # to void inconsistency while exporting to onnx
        mask_channel = int(mask_c / float(self.up_kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

    def feature_reassemble(self, x, mask):
        x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def carafe_forward(self, x):
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        x = self.feature_reassemble(x, mask)
        return x

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        # compressed_lr_feat = resize(
        #     input=compressed_lr_feat,
        #     size=hr_feat.shape[2:],
        #     mode=self.upsample_mode,
        #     align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
        # if self.use_spatial_suppression:
            # _, hr_feat, lr_feat = self.ss(hr_feat, lr_feat, residual=self.use_spatial_residual)
        # x = torch.cat([hr_feat, lr_feat], dim=1)
        # compressed_x = self.channel_compressor(x)
        if self.semi_conv:
            # mask = self.content_encoder(compressed_hr_feat) + F.interpolate(self.content_encoder(compressed_lr_feat), scale_factor=2, mode='nearest')
            mask = self.content_encoder(compressed_hr_feat) + F.interpolate(self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[2:], mode='nearest')
        else:
            # compressed_x = F.interpolate(compressed_lr_feat, scale_factor=2, mode='nearest') + compressed_hr_feat
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[2:], mode='nearest') + compressed_hr_feat
            mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        # if self.use_spatial_suppression:
            # hr_feat = self.spatial_suppression(compressed_x) * hr_feat
        if self.use_high_pass:
            if self.hr_residual:
                # print('using hr_residual')
                hr_feat = hr_feat - self.feature_reassemble(hr_feat, mask) + hr_feat
            else:
                hr_feat = hr_feat - self.feature_reassemble(hr_feat, mask)
        if self.use_low_pass:
            if self.use_encoder2:
                # mask2 = self.content_encoder2(compressed_hr_feat) + F.interpolate(self.content_encoder2(compressed_lr_feat), scale_factor=2)
                mask2 = self.content_encoder2(compressed_x)
                mask2 = self.kernel_normalizer(mask2)
            else:
                mask2 = mask
            if self.semi_conv:
                lr_feat = carafe(lr_feat, mask, self.up_kernel, self.up_group, 2)
            else:
                lr_feat = F.interpolate(
                    input=lr_feat,
                    size=hr_feat.shape[2:],
                    mode=self.upsample_mode,
                    align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
                lr_feat = self.feature_reassemble(lr_feat, mask2)
        return mask, hr_feat, lr_feat

class SpatialGateGenerator(nn.Module):
    def __init__(self, in_channels):
        super(SpatialGateGenerator, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.weights_init_random()

    def forward(self, hr_feat, lr_feat):
        return torch.sigmoid(F.interpolate(self.conv(lr_feat), size=hr_feat.shape[2:], mode='nearest'))

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

############ Dysample
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
    
class DyUpSample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True, kernel_size=1):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset, scale=None):
        if scale is None: scale = self.scale
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(
            B, 2, -1, scale * H, scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, scale * H, scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)
    
    def forward(self, offset_feat, x):
        offset = self.get_offset(offset_feat)
        return self.sample(x, offset)
    
    def get_offset_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return offset

    def get_offset_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return offset

    def get_offset(self, x):
        if self.style == 'pl':
            return self.get_offset_pl(x)
        return self.get_offset_lp(x)
    
class FusionDyUpSample(DyUpSample):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True, kernel_size=1):
        super().__init__(in_channels=in_channels, scale=scale, style=style, groups=groups, dyscope=dyscope, kernel_size=kernel_size)
        assert scale==2
        assert style=='lp'
        out_channels = 2 * groups
        self.hr_offset = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.hr_scope = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            constant_init(self.scope, val=0.)

    def forward(self, hr_x, lr_x, feat2sample):
        offset = self.get_offset(hr_x, lr_x)
        return super().sample(feat2sample, offset)
    
    def get_offset_lp(self, hr_x, lr_x):
        if hasattr(self, 'scope'):
            offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * (self.scope(lr_x) + F.pixel_unshuffle(self.hr_scope(hr_x), self.scale)).sigmoid() * 0.5 + self.init_pos
        else:
            offset =  (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def get_offset(self, hr_x, lr_x):
        if self.style == 'pl':
            raise NotImplementedError
        return self.get_offset_lp(hr_x, lr_x)

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm ** 2)

class ConvSaliencySampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, align_corners=False, coord_mode='yx', out_mode='abs_offset', 
                 gaussian_radius=2, padding_size=2, padding_mode='replicate'):
        super().__init__()
        self.align_corners = align_corners
        self.dilation = dilation
        self.gaussian_radius = gaussian_radius
        self.padding_size = padding_size
        self.padding_mode = padding_mode
        self.saliency_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        # self.saliency_conv_hr = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        # self.saliency_conv_lr = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        self.eps = 1e-16
        self.coord_mode = coord_mode
        self.out_mode = out_mode # abs_offset for deformconv, rel_grid for grid_sample
        self.init_gaussian_filter()
        # for fusion
        assert self.coord_mode == 'xy'
        assert self.out_mode == 'rel_grid'

    def init_gaussian_filter(self):
        gaussian_weights = torch.Tensor(makeGaussian(2 * self.padding_size + 1, fwhm = self.gaussian_radius))
        self.gaussian_weights = nn.Parameter(gaussian_weights[None, None, :], requires_grad=False)
        # print("gaussian_weights", gaussian_weights)
        # self.filter = nn.Conv2d(1, 1, kernel_size = (2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        # self.filter.weight[0].data[:,:,:] = gaussian_weights
        # self.filter.weight.requires_grad = False
        # print(self.gaussian_weights)

    def Filter(self, x):
        # print(self.gaussian_weights)
        return F.conv2d(x, weight=self.gaussian_weights, bias=None, stride=1, padding=0, dilation=1, groups=1)

    def forward(self, feature, sample_with_grid=True):
        saliency = self.saliency_conv(feature)
        b, c, h, w = saliency.shape
        saliency = saliency.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        offsets = self.create_conv_offset(saliency, h, w) # b, h, w, 2
        return offsets
        # return feature, feature_sampled, reversed_feat, saliency, grid
    
    def gen_coord_grid(self, h, w, device=None):
        """
        return: [2, h, w]
        """
        if self.align_corners:
            # x = torch.Tensor(list(range(-self.padding_size, w + self.padding_size, 1))) / (w - 1.0)
            # y = torch.Tensor(list(range(-self.padding_size, h + self.padding_size, 1))) / (h - 1.0)
            x = torch.arange(-self.padding_size, w + self.padding_size, 1) / (w - 1.0)
            y = torch.arange(-self.padding_size, h + self.padding_size, 1) / (h - 1.0)
        else:
            x = torch.arange(-self.padding_size, w + self.padding_size, 1) / w + 0.5 / w
            y = torch.arange(-self.padding_size, h + self.padding_size, 1) / h + 0.5 / h
        coord_grid = torch.stack(torch.meshgrid(y, x)[::-1], dim=0)
        # print(coord_grid.shape)
        # coord_grid = torch.stack(torch.meshgrid([x, y]), dim=0).transpose(1, 2)
        # print(coord_grid.shape)
        if device is not None: coord_grid = coord_grid.to(device)
        return coord_grid
    
    def create_conv_offset(self, s, h, w):
        """
        x: saliency in [b, g, h, w]
        """
        s = F.pad(s, pad=[self.padding_size] * 4, mode=self.padding_mode) #避免等式2和3偏向于图像中心的采样偏好
        padded_h, padded_w = h + 2 * self.padding_size, w + 2 * self.padding_size
        P = self.gen_coord_grid(h, w, device=s.device).reshape(1, 2, padded_h, padded_w)
        P = P.expand(s.size(0), -1, padded_h, padded_w) # b, 2, ph, pw
        P = P.repeat(1, s.size(1), 1, 1) # b, 2 * g, ph, pw

        p_filter = self.Filter(s.view(-1, 1, padded_h, padded_w)) # b * g, 1, ph, pw
        p_filter = p_filter.view(s.size(0), -1, h, w) # b, g, ph, pw
        '''
        #得到的是论文中等式2和等式3的分子
        x_mul = torch.mul(P, x_cat).view(-1, 1, padded_h, padded_w) #[batch_size*2, 1, 91, 91]
        # print("x_mul size is : ", x_mul.size()) #torch.Size([10, 1, 91, 91])
        #filter()输入[batch_size*2, 1, 91, 91], 输出[batch_size*2, 1, 31, 31]
        #然后重置为[batch_size, 2, 31, 31]
        all_filter = self.filter(x_mul).view(x.size(0), -1, h, w)
        # print("all_filter size is : ", all_filter.size()) #torch.Size([5, 2, 31, 31])

        # x_filter是u(x,y)的分子,y_filter是v(x,y)的分子
        x_filter = all_filter[:,0::2,:,:].contiguous().view(x.size(0), -1, h, w) #[batch_size, 1, 31, 31]
        y_filter = all_filter[:,1::2,:,:].contiguous().view(x.size(0), -1, h, w) #[batch_size, 1, 31, 31]
        # print("y_filter size is : ", y_filter.size()) #torch.Size([5, 1, 31, 31])
        '''
        x_filter = torch.mul(P[:,0::2,:,:].view(s.size(0), -1, padded_h, padded_w), s).view(-1, 1, padded_h, padded_w)
        x_filter = self.Filter(x_filter).view(s.size(0), -1, h, w) # b, g, ph, pw
        y_filter = torch.mul(P[:,1::2,:,:].view(s.size(0), -1, padded_h, padded_w), s).view(-1, 1, padded_h, padded_w)
        y_filter = self.Filter(y_filter).view(s.size(0), -1, h, w) # b, g, ph, pw

        #值的范围是[0,1]
        # print(x.min(), x.max(), x.mean())
        # x_filter = x_filter / (p_filter + 1e-16) #u(x,y)
        # y_filter = y_filter / (p_filter + 1e-16) #v(x,y)
        x_filter = x_filter / (p_filter + self.eps) #u(x,y)
        y_filter = y_filter / (p_filter + self.eps) #v(x,y)
        if 'abs_offset' ==  self.out_mode:
            x_offsets = (x_filter - P[:,0::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]) * w
            y_offsets = (y_filter - P[:,1::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]) * h
        elif 'rel_grid' ==  self.out_mode:
            x_offsets = x_filter * 2.0 - 1.0
            y_offsets = y_filter * 2.0 - 1.0
        else:
            raise NotImplementedError
        if self.coord_mode == 'xy': # dcnv3, grid_sample
            grid = torch.stack((x_offsets, y_offsets), 2) #[batch_size, position, 2, h, w]
        elif self.coord_mode == 'yx': # dcnv2
            grid = torch.stack((y_offsets, x_offsets), 2) #[batch_size, position, 2, h, w]
        else:
            raise NotImplementedError
        grid = grid.reshape(s.size(0), -1, h, w) #[batch_size, position, 2, h, w]
        return grid
    
    def sample_with_grid(self, feature, grid):
        feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        return feature_sampled
    
class FusionSaliencySampler(ConvSaliencySampler):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, align_corners=False, 
                 gaussian_radius=2, padding_size=2):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, align_corners=align_corners, coord_mode='xy', out_mode='rel_grid', gaussian_radius=gaussian_radius, padding_size=padding_size)
        del self.saliency_conv
        self.saliency_conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        self.saliency_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)

        self.conv_att_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        self.conv_att_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        normal_init(self.saliency_conv_1, std=0.001)
        normal_init(self.saliency_conv_2, std=0.001)
        constant_init(self.conv_att_1, val=0.)
        constant_init(self.conv_att_2, val=0.)

    def compressed_feat_forward(self, hr_x, lr_x, mask_lr, up_kernel, up_lr_x):
        """
        hr_x, lr_x, compressed feature
        """
        saliency = carafe(self.saliency_conv_1(lr_x), mask_lr, up_kernel, 1, 2) + self.saliency_conv_1(hr_x)
        saliency2 = carafe(self.saliency_conv_2(lr_x), mask_lr, up_kernel, 1, 2) + self.saliency_conv_2(hr_x)
        # att1 = carafe(self.conv_att_1(lr_x), mask_lr, up_kernel, 1, 2) + self.conv_att_1(hr_x)
        # att2 = carafe(self.conv_att_2(lr_x), mask_lr, up_kernel, 1, 2) + self.conv_att_2(hr_x)
        # att1 = att1.sigmoid()
        # att2 = att2.sigmoid()
        # saliency = F.pixel_shuffle(self.saliency_conv_lr(lr_x), upscale_factor=2) + self.saliency_conv_hr(hr_x)
        # saliency2 = F.pixel_shuffle(self.saliency_conv_lr_2(lr_x), upscale_factor=2) + self.saliency_conv_hr_2(hr_x)
        b, c, h, w = saliency.shape
        saliency = saliency.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        saliency2 = saliency2.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        # saliency2 = saliency
        grid = self.create_conv_offset(saliency, saliency2, 1, 1, h, w)
        grid = grid.reshape(-1, 2, h, w)
        group = c
        feature_sampled = F.grid_sample(up_lr_x.reshape(b * group, -1, h, w), grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        return feature_sampled.reshape(b, -1, h, w)

    def create_conv_offset(self, s, s2, att1, att2, h, w):
        """
        x: saliency in [b, g, h, w]
        """
        s = F.pad(s, pad=[self.padding_size] * 4, mode=self.padding_mode) #避免等式2和3偏向于图像中心的采样偏好
        s2 = F.pad(s2, pad=[self.padding_size] * 4, mode=self.padding_mode) #避免等式2和3偏向于图像中心的采样偏好
        padded_h, padded_w = h + 2 * self.padding_size, w + 2 * self.padding_size

        P = self.gen_coord_grid(h, w, device=s.device).reshape(1, 2, padded_h, padded_w)
        P = P.expand(s.size(0), -1, padded_h, padded_w) # b, 2, ph, pw
        P = P.repeat(1, s.size(1), 1, 1) # b, 2 * g, ph, pw

        p_filter = self.Filter(s.view(-1, 1, padded_h, padded_w)) # b * g, 1, ph, pw
        p_filter = p_filter.view(s.size(0), -1, h, w) # b, g, ph, pw
        
        p_filter2 = self.Filter(s2.view(-1, 1, padded_h, padded_w)) # b * g, 1, ph, pw
        p_filter2 = p_filter2.view(s2.size(0), -1, h, w) # b, g, ph, pw
        '''
        #得到的是论文中等式2和等式3的分子
        x_mul = torch.mul(P, x_cat).view(-1, 1, padded_h, padded_w) #[batch_size*2, 1, 91, 91]
        # print("x_mul size is : ", x_mul.size()) #torch.Size([10, 1, 91, 91])
        #filter()输入[batch_size*2, 1, 91, 91], 输出[batch_size*2, 1, 31, 31]
        #然后重置为[batch_size, 2, 31, 31]
        all_filter = self.filter(x_mul).view(x.size(0), -1, h, w)
        # print("all_filter size is : ", all_filter.size()) #torch.Size([5, 2, 31, 31])

        # x_filter是u(x,y)的分子,y_filter是v(x,y)的分子
        x_filter = all_filter[:,0::2,:,:].contiguous().view(x.size(0), -1, h, w) #[batch_size, 1, 31, 31]
        y_filter = all_filter[:,1::2,:,:].contiguous().view(x.size(0), -1, h, w) #[batch_size, 1, 31, 31]
        # print("y_filter size is : ", y_filter.size()) #torch.Size([5, 1, 31, 31])
        '''
        x_filter = torch.mul(P[:,0::2,:,:].view(s.size(0), -1, padded_h, padded_w), s).view(-1, 1, padded_h, padded_w)
        x_filter = self.Filter(x_filter).view(s.size(0), -1, h, w) # b, g, ph, pw
        y_filter = torch.mul(P[:,1::2,:,:].view(s2.size(0), -1, padded_h, padded_w), s2).view(-1, 1, padded_h, padded_w)
        y_filter = self.Filter(y_filter).view(s2.size(0), -1, h, w) # b, g, ph, pw

        #值的范围是[0,1]
        # print(x.min(), x.max(), x.mean())
        # x_filter = x_filter / (p_filter + 1e-16) #u(x,y)
        # y_filter = y_filter / (p_filter + 1e-16) #v(x,y)
        x_filter = x_filter / (p_filter + self.eps) #u(x,y)
        y_filter = y_filter / (p_filter2 + self.eps) #v(x,y)
        if 'abs_offset' ==  self.out_mode:
            x_offsets = (x_filter - P[:,0::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]) * w
            y_offsets = (y_filter - P[:,1::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]) * h
        elif 'rel_grid' ==  self.out_mode:
            grid_x = P[:,0::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
            grid_y = P[:,1::2, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
            x_offsets = (x_filter - grid_x) * att1 + grid_x
            y_offsets = (y_filter - grid_y) * att2 + grid_y
            x_offsets = x_offsets * 2.0 - 1.0
            y_offsets = y_offsets * 2.0 - 1.0
            # x_offsets = x_filter * 2.0 - 1.0
            # y_offsets = y_filter * 2.0 - 1.0
        else:
            raise NotImplementedError
        if self.coord_mode == 'xy': # dcnv3, grid_sample
            grid = torch.stack((x_offsets, y_offsets), 2) #[batch_size, position, 2, h, w]
        elif self.coord_mode == 'yx': # dcnv2
            grid = torch.stack((y_offsets, x_offsets), 2) #[batch_size, position, 2, h, w]
        else:
            raise NotImplementedError
        grid = grid.reshape(s.size(0), -1, h, w) #[batch_size, position, 2, h, w]
        return grid

    def align_forward(self, hr_x, lr_x, mask_lr, up_kernel):
        saliency = carafe(self.saliency_conv_lr(lr_x), mask_lr, up_kernel, 1, 2) + self.saliency_conv_hr(hr_x)
        b, c, h, w = saliency.shape
        saliency = saliency.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        grid = self.create_conv_offset(saliency, h, w)
        feature_sampled = F.grid_sample(feature, grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        return feature_sampled

def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    """
    计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

    参数：
    - input_tensor: 输入张量，形状为[B, C, H, W]
    - k: 范围大小，表示周围KxK范围内的点

    返回：
    - 输出张量，形状为[B, KxK-1, H, W]
    """
    B, C, H, W = input_tensor.shape
    # 使用零填充来处理边界情况
    # padded_input = F.pad(input_tensor, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

    # 展平输入张量中每个点及其周围KxK范围内的点
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation) # B, CxKxK, HW
    # print(unfold_tensor.shape)
    unfold_tensor = unfold_tensor.reshape(B, C, k**2, H, W)

    # 计算余弦相似度
    if sim == 'cos':
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    # 移除中心点的余弦相似度，得到[KxK-1]的结果
    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

    # 将结果重塑回[B, KxK-1, H, W]的形状
    similarity = similarity.view(B, k * k - 1, H, W)

    return similarity

class LocalSimGuidedSampler(DyUpSample):
    # class FusionDyUpSample(DyUpSample):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True, kernel_size=1, local_window=3, sim_type='cos', norm=False):
        super().__init__(in_channels=in_channels, scale=scale, style=style, groups=groups, dyscope=dyscope, kernel_size=kernel_size)
        assert scale==2
        assert style=='lp'

        self.local_window = local_window
        self.sim_type = sim_type

        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2
        # self.offset = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.offset = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        normal_init(self.offset, std=0.001)
        if dyscope:
            # self.scope = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            self.scope = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            constant_init(self.scope, val=0.)

        out_channels = 2 * groups
        # self.hr_offset = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.hr_offset = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        normal_init(self.offset, std=0.001)
        
        if dyscope:
            # self.hr_scope = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            self.hr_scope = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            constant_init(self.scope, val=0.)

        self.norm = norm
        if self.norm:
            self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
            self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)

    def forward(self, hr_x, lr_x, feat2sample):
        # hr_x = self.norm_hr(hr_x)
        # lr_x = self.norm_lr(lr_x)
        hr_x = torch.cat([hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')], dim=1)
        lr_x = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')], dim=1)
        # hr_sim = compute_similarity(hr_x, self.local_window)
        # lr_sim = compute_similarity(lr_x, self.local_window)
        offset = self.get_offset(hr_x, lr_x)
        return super().sample(feat2sample, offset)
    
    def get_offset_lp(self, hr_x, lr_x):
    # def get_offset_lp(self, hr_x, lr_x, hr_sim, lr_sim):
        if hasattr(self, 'scope'):
            offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * (self.scope(lr_x) + F.pixel_unshuffle(self.hr_scope(hr_x), self.scale)).sigmoid() + self.init_pos
        else:
            offset =  (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def get_offset(self, hr_x, lr_x):
        if self.style == 'pl':
            raise NotImplementedError
        return self.get_offset_lp(hr_x, lr_x)
    
class SaliencyDySampler(ConvSaliencySampler):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, align_corners=False, 
                 gaussian_radius=2, padding_size=2):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, align_corners=align_corners, coord_mode='xy', out_mode='rel_grid', gaussian_radius=gaussian_radius, padding_size=padding_size)
        del self.saliency_conv
        # self.saliency_conv_hr = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
        self.saliency_conv_lr = nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, padding=self.dilation * (kernel_size // 2), dilation=self.dilation, stride=1)
    
    def forward(self, x):
        saliency = self.saliency_conv_lr(x)
        saliency = F.pixel_shuffle(saliency, upscale_factor=2)
        # print(saliency.shape)
        b, c, h, w = saliency.shape
        saliency = saliency.view(b, c, h * w).softmax(dim=-1).view(b, c, h, w) * h * w
        grid = self.create_conv_offset(saliency, h, w)
        grid = grid.reshape(-1, 2, h, w)
        group = c
        feature_sampled = F.grid_sample(x.reshape(b * group, -1, x.size(-2), x.size(-1)), grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=self.align_corners, padding_mode="border") #得到重采样的图像
        return feature_sampled.reshape(b, -1, h, w)
    
##########
class GobalContextLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """
    def __init__(self, in_channels: int,  c_n: int, c_m: int = 0, reconstruct = False):
        """

        Parameters
        ----------
        in_channels
        c_m
        c_n
        reconstruct: `bool` whether to re-construct output to have shape (B, in_channels, L, R)
        """
        super().__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        # self.convA = nn.Conv2d(in_channels, c_m, kernel_size = 1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)

    def forward(self, compressed_x: torch.Tensor, x: torch.Tensor):
        """

        Parameters
        ----------
        x: `torch.Tensor` of shape (B, C, H, W)

        Returns
        -------

        """
        batch_size, c, h, w = x.size()
        # assert c == self.in_channels, 'input channel not equal!'
        # A = self.convA(compressed_x)  # (B, c_m, h, w) because kernel size is 1
        B = self.convB(compressed_x)  # (B, c_n, h, w)
        V = self.convV(compressed_x)  # (B, c_n, h, w)
        tmpA = x.view(batch_size, c, h * w)
        attention_maps = B.view(batch_size, self.c_n, h * w)
        attention_vectors = V.view(batch_size, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim = -1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = 1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        # tmpZ = tmpZ.view(batch_size, self.c_m, h, w)
        tmpZ = tmpZ.view(batch_size, c, h, w)
        if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ

def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M：窗口的行数
    - N：窗口的列数

    返回：
    - 二维Hamming窗
    """
    # 生成水平和垂直方向上的Hamming窗
    # hamming_x = np.blackman(M)
    # hamming_x = np.kaiser(M)
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # 通过外积生成二维Hamming窗
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d

class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        # return x * g
        return g
    
class EdgeConvAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=1, kernel_num=4, min_channel=16):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.c_att = SRMLayer(in_planes)
        # self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        # self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        # self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        # self.func_channel = self.get_channel_attention

        # if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
        #     self.func_filter = self.skip
        # else:
        #     self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
        #     self.func_filter = self.get_filter_attention

        # if kernel_size == 1:  # point-wise convolution
        #     self.func_spatial = self.skip
        # else:
        #     self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
        #     self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    # def get_channel_attention(self, x):
    #     channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
    #     return channel_attention

    # def get_filter_attention(self, x):
    #     filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
    #     return filter_attention

    # def get_spatial_attention(self, x):
    #     spatial_attention = self.spatial_fc(x).reshape(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
    #     # spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
    #     spatial_attention = torch.sigmoid(spatial_attention)
    #     return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).reshape(x.size(0), -1, 1, 1, 1, 1)
        # kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        kernel_attention = F.softmax(kernel_attention, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x) * self.c_att(x)
        # x = self.avgpool(x)
        # x = self.fc(x)
        # x = self.bn(x)
        x = self.relu(x)
        # return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
        return 1, 1, 1, self.func_kernel(x)

class DyEdgeConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 reduction=1, kernel_num=5):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = EdgeConvAttention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=5)
        # self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        self.weight = nn.Parameter(torch.randn(1, 1, out_planes, in_planes//groups, kernel_size, kernel_size) * 0.001, requires_grad=True) # 1, 1 for batch and kernel num
        self.bias = nn.Parameter(torch.zeros(out_planes), requires_grad=True)
        self.weight_lr_diag = nn.Parameter(torch.randn(1, 1, out_planes, in_planes//groups, kernel_size) * 0.001, requires_grad=True)
        self.weight_rl_diag = nn.Parameter(torch.randn(1, 1, out_planes, in_planes//groups, kernel_size) * 0.001, requires_grad=True)
        self.weight_h = nn.Parameter(torch.randn(1, 1, out_planes, in_planes//groups, 1, kernel_size) * 0.001, requires_grad=True)
        self.weight_v = nn.Parameter(torch.randn(1, 1, out_planes, in_planes//groups, kernel_size, 1) * 0.001, requires_grad=True)
        # normal_init(self.content_encoder2, std=0.001)
        # self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        # print('kernel_attention', kernel_attention.shape)
        # print('spatial_attention', spatial_attention.shape)
        batch_size, in_planes, height, width = x.size()
        # x = x * channel_attention
        x = x.reshape(1, -1, height, width) # 1, b*c , h, w 
        # spatial_attention (b, 1, 1, 1, self.kernel_size, self.kernel_size)
        # kernel_attention (b, -1, 1, 1, 1, 1)
        # aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        weight = kernel_attention[:, 0:1] * self.weight
        weight_lr_diag = kernel_attention[:, 1:2, ..., 0] * self.weight_lr_diag
        weight_rl_diag = kernel_attention[:, 2:3, ..., 0] * self.weight_rl_diag
        weight_h = kernel_attention[:, 3:4] * self.weight_h
        weight_v = kernel_attention[:, 4:5] * self.weight_v

        weight += torch.diag_embed(weight_lr_diag)
        # print(torch.diag_embed(weight_lr_diag))
        weight += torch.flip(torch.diag_embed(weight_rl_diag), dims=[-1])
        # print(torch.flip(torch.diag_embed(weight_rl_diag), dims=[-1]))
        weight[..., self.kernel_size//2:self.kernel_size//2+1, :] += weight_h
        weight[..., self.kernel_size//2:self.kernel_size//2+1] += weight_v
        # print('weight', weight.shape)
        
        # aggregate_weight = spatial_attention * weight
        aggregate_weight = weight
        # print('aggregate_weight', aggregate_weight.shape)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size)
        # print(aggregate_weight.shape) # batch * out_c, in_c, kh, kw
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1)) + self.bias.view(1, -1, 1, 1)
        # print(self.bias)
        # output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        # print('training', self.training)
        return self._forward_impl(x)
    
class FreqFusion2encoder(nn.Module):
    # def __init__(self,
    #              channels: int,
    #              scale_factor: int,
    #              up_kernel: int = 5,
    #              up_group: int = 1,
    #              encoder_kernel: int = 3,
    #              encoder_dilation: int = 1,
    #              compressed_channels: int = 64):
    def __init__(self,
                hr_channels,
                lr_channels,
                scale_factor=1,
                lowpass_kernel=5,
                highpass_kernel=3,
                lowpass_pad=0,
                highpass_pad=0,
                padding_mode='replicate',
                hamming_window=True,
                up_group=1,
                temp_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64,        
                align_corners=False,
                upsample_mode='nearest',
                feature_align=False,
                feature_align_group=4,
                comp_feat_upsample=True,
                # use_encoder2=False,
                use_spatial_suppression=False,
                use_high_pass=True,
                use_low_pass=True,
                hr_residual=True,
                hf_att=False,
                semi_conv=False,
                use_spatial_gate=False,
                use_global_context=False,
                use_channel_att=False,
                use_dyedgeconv=False,
                **kwargs):
        super().__init__()
        # self.channels = channels
        self.scale_factor = scale_factor
        # self.up_kernel = up_kernel

        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.lowpass_pad = lowpass_pad
        self.highpass_pad = highpass_pad
        self.padding_mode = padding_mode
        self.hamming_window = hamming_window
        self.use_channel_att = use_channel_att
        self.use_dyedgeconv = use_dyedgeconv
        if self.hamming_window:
            self.register_buffer('hamming_lowpass', torch.FloatTensor(hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
            self.register_buffer('hamming_highpass', torch.FloatTensor(hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
            # self.hamming_lowpass = 1
            # self.hamming_highpass = 1
        self.up_group = up_group
        self.temp_group = temp_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels,1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels,1)
        if self.use_channel_att:
            self.hr_channel_att = SRMLayer(hr_channels)
            self.lr_channel_att = SRMLayer(lr_channels)
        if self.use_dyedgeconv:
            self.content_encoder = DyEdgeConv2d(self.compressed_channels, 
                                                lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor + (self.temp_group if self.temp_group > 1 else 0), 
                                                kernel_size=self.encoder_kernel, stride=1, padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2), dilation=self.encoder_dilation, groups=1, reduction=1, kernel_num=5)
        else:
            self.content_encoder = nn.Conv2d(
                self.compressed_channels,
                lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor + (self.temp_group if self.temp_group > 1 else 0),
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        # self.lr_temp_encoder = None
        # if self.temp_group > 1:
        #     self.lr_temp_encoder = nn.Conv2d(
        #                 self.compressed_channels,
        #                 self.temp_group,
        #                 self.encoder_kernel,
        #                 padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
        #                 dilation=self.encoder_dilation,
        #                 groups=1)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.use_spatial_suppression = use_spatial_suppression
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.use_spatial_gate = use_spatial_gate
        self.semi_conv = semi_conv
        self.feature_align = feature_align
        self.comp_feat_upsample = comp_feat_upsample
        if self.feature_align:
            # self.lr_dysampler = AlignUpSample(in_channels=compressed_channels, scale=2, style='lp', groups=4, dyscope=True, kernel_size=encoder_kernel)
            # self.hr_dysampler = AlignUpSample(in_channels=compressed_channels, scale=1, style='lp', groups=4, dyscope=True, kernel_size=encoder_kernel)
            self.dysampler = LocalSimGuidedSampler(in_channels=compressed_channels, scale=2, style='lp', groups=feature_align_group, dyscope=True, kernel_size=encoder_kernel)
            # self.dysampler = FusionSaliencySampler(in_channels=compressed_channels, out_channels=feature_align_group, kernel_size=3, dilation=1, align_corners=False, gaussian_radius=2, padding_size=2)
        if self.use_high_pass:
            if self.use_dyedgeconv:
                self.content_encoder2 = DyEdgeConv2d(self.compressed_channels, 
                                                     highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor + (self.temp_group if self.temp_group > 1 else 0), 
                                                     kernel_size=self.encoder_kernel, stride=1, padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2), dilation=self.encoder_dilation, groups=1, reduction=1, kernel_num=5)
            else:
                self.content_encoder2 = nn.Conv2d(
                    self.compressed_channels,
                    highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor + (self.temp_group if self.temp_group > 1 else 0),
                    self.encoder_kernel,
                    padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                    dilation=self.encoder_dilation,
                    groups=1)
            if hf_att:
                self.hf_att_hr_conv = nn.Conv2d(
                    self.compressed_channels,
                    1, 
                    self.encoder_kernel, 
                    padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                    dilation=self.encoder_dilation,
                    groups=1)
                self.hf_att_lr_conv = nn.Conv2d(
                    self.compressed_channels,
                    1, 
                    self.encoder_kernel, 
                    padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                    dilation=self.encoder_dilation,
                    groups=1)
            # self.hr_temp_encoder = None
            # if self.temp_group > 1:
            #     self.hr_temp_encoder = nn.Conv2d(
            #             self.compressed_channels,
            #             self.temp_group,
            #             self.encoder_kernel,
            #             padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            #             dilation=self.encoder_dilation,
            #             groups=1)
        if use_global_context:
            self.global_context_layer = GobalContextLayer(in_channels=compressed_channels, c_n=compressed_channels)
        if self.use_spatial_suppression:
            self.spatial_suppression = nn.Sequential(
                nn.Conv2d(self.compressed_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
        if use_spatial_gate:
            self.sp_gate = SpatialGateGenerator(in_channels=lr_channels)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, padding=0, hamming=1, scale_factor=None):
        if self.temp_group > 1: 
            temp = mask[:, -self.temp_group:]
            mask = mask[:, :-self.temp_group]
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        # use float division explicitly,
        # to void inconsistency while exporting to onnx
        # mask_channel = int(mask_c / float(self.up_kernel**2))
        mask_channel = int(mask_c / float(kernel**2))
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()
        if self.temp_group > 1: 
            mask = mask.view(n, mask_channel, 1, -1, h, w)
            mask = mask.repeat(1, 1, self.temp_group, 1, 1, 1) * temp.reshape(n, 1, self.temp_group, 1, h, w)
            mask = F.softmax(mask, dim=-3, dtype=mask.dtype)
            mask = mask.view(n, -1, h, w).contiguous()
        else:
            if self.hamming_window:
                # mask = mask.view(n, mask_channel, kernel, kernel, h, w)
                # mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
                # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
                # mask = mask * hamming
                # # print(hamming)
                # # print(mask.shape)
                # mask = mask.view(n, mask_channel, h, w, -1).softmax(-1)
                # mask =  mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()

                mask = mask.view(n, mask_channel, -1, h, w)
                mask = F.softmax(mask, dim=2, dtype=mask.dtype)
                mask = mask.view(n, mask_channel, kernel, kernel, h, w)
                mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
                mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
                mask = mask * hamming
                mask /= mask.sum(dim=(-1, -2), keepdims=True)
                # print(hamming)
                # print(mask.shape)
                mask = mask.view(n, mask_channel, h, w, -1)
                mask =  mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
            else:
                mask = mask.view(n, mask_channel, -1, h, w)
                mask = F.softmax(mask, dim=2, dtype=mask.dtype)
                mask = mask.view(n, mask_c, h, w).contiguous()
        return mask

    # def feature_reassemble(self, x, mask):
    #     x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
    #     return x

    # def carafe_forward(self, x):
    #     compressed_x = self.channel_compressor(x)
    #     mask = self.content_encoder(compressed_x)
    #     mask = self.kernel_normalizer(mask)
    #     x = self.feature_reassemble(x, mask)
    #     return x

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        if self.use_channel_att:
            hr_feat *= self.hr_channel_att(hr_feat)
            lr_feat *= self.lr_channel_att(lr_feat)
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        if hasattr(self, 'global_context_layer'): # 'dyupsample'
            global_context_lr_feat = self.global_context_layer(compressed_lr_feat, lr_feat)
        if self.semi_conv:
            if self.comp_feat_upsample:
                # mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                # mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, padding=self.lowpass_pad, hamming=self.hamming_lowpass)
                # mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                # mask_lr_lr_feat = F.interpolate(
                #     carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                # mask_lr = mask_lr_hr_feat + mask_lr_lr_feat
                # if self.use_high_pass:
                #     mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, padding=self.lowpass_pad, hamming=self.hamming_lowpass)
                #     mask_hr_lr_feat = F.interpolate(
                #         carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                #     mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                #     mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                if self.use_high_pass:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                    mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel, padding=self.highpass_pad, hamming=self.hamming_highpass)
                    compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat, mask_hr_init, self.highpass_kernel + 2 * self.highpass_pad, self.temp_group, 1)
                    
                    mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                    mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, padding=self.lowpass_pad, hamming=self.hamming_lowpass)
                    mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                    mask_lr_lr_feat = F.interpolate(
                        carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

                    mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, padding=self.lowpass_pad, hamming=self.hamming_lowpass)
                    mask_hr_lr_feat = F.interpolate(
                        carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                else: raise NotImplementedError
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
        else:
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:], mode='nearest') + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_high_pass: 
                mask_hr = self.content_encoder2(compressed_x)
        
        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, padding=self.lowpass_pad, hamming=self.hamming_lowpass)
        if self.semi_conv:
            # if self.feature_align: # 'dyupsample'
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2)
            #     # lr_feat = self.dysampler.compressed_feat_forward(hr_x=compressed_hr_feat, lr_x=compressed_lr_feat, mask_lr=mask_lr, up_kernel=self.lowpass_kernel, up_lr_x=lr_feat)
            #     # if hasattr(self, 'global_context_layer'):
            #     #     global_context_lr_feat = self.dysampler.sample(global_context_lr_feat, offset)
            #     #     lr_feat = 0.5 * (carafe(lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 1) + global_context_lr_feat)
            #     # else:
            #     #     lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2)
            #     #     lr_feat = self.dysampler.sample(lr_feat, offset, scale=2)
            # else:
            #     lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2)
        else:
            lr_feat = resize(
                input=lr_feat,
                size=hr_feat.shape[2:],
                mode=self.upsample_mode,
                align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 1)
            if self.feature_align:
                lr_feat = self.dysampler.sample(lr_feat, offset, scale=2)

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, padding=self.highpass_pad, hamming=self.hamming_highpass)
            if self.hr_residual:
                # print('using hr_residual')
                hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel + 2 * self.highpass_pad, self.temp_group, 1)
                if hasattr(self, 'hf_att_hr_conv') and hasattr(self, 'hf_att_lr_conv'): 
                    hr_feat_hf_att = self.hf_att_hr_conv(compressed_hr_feat) + self.hf_att_lr_conv(F.interpolate(
                            carafe(compressed_lr_feat, mask_lr, self.lowpass_kernel + 2 * self.lowpass_pad, self.temp_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest'))
                    hr_feat_hf_att = torch.tanh(hr_feat_hf_att)
                    hr_feat_hf = hr_feat_hf * hr_feat_hf_att
                hr_feat = hr_feat_hf + hr_feat
            else:
                hr_feat = hr_feat_hf
        # print(mask_lr.shape)
        # print(mask_hr.shape)
        if self.feature_align: # 'dyupsample'
            lr_feat = self.dysampler(hr_x=compressed_hr_feat, 
                                     lr_x=compressed_lr_feat, feat2sample=lr_feat)
                
        if self.use_spatial_gate: 
            sp_gate = self.sp_gate(hr_feat, lr_feat)
            hr_feat *= sp_gate
            lr_feat *= (1.0 - sp_gate)
        # print(mask_lr.std(dim=[1, 2, 3]).mean())
        return  mask_lr, hr_feat, lr_feat


@SEM_SEG_HEADS_REGISTRY.register()
class FreqAwareMSDeformAttnPixelDecoder(MSDeformAttnPixelDecoder):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        conv_dim = kwargs.get('conv_dim', 256)
        self.freqfusions = nn.ModuleList()
        if self.transformer_num_feature_levels > 1:
            # from low resolution to high resolution (res5 -> res2)
            # for in_channels in transformer_in_channels[::-1]:
            for _ in range(self.transformer_num_feature_levels - 1):
                # self.freqfusions.append(
                #     FreqFusion(lr_channels = conv_dim, hr_channels = conv_dim, scale_factor=1, up_kernel=3, up_group=1, 
                #         upsample_mode='nearest', align_corners=False, use_encoder2=False, hr_residual=True, 
                #         use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=64,
                #         use_high_pass=True, use_low_pass=True)
                # )
                self.freqfusions.append(
                    FreqFusion2encoder(lr_channels = conv_dim, hr_channels = conv_dim, 
                        scale_factor=1,
                        lowpass_kernel=5,
                        highpass_kernel=3,
                        lowpass_pad = 0,
                        highpass_pad = 0,
                        padding_mode = 'replicate',
                        hamming_window = False,
                        comp_feat_upsample = True,
                        feature_align = False,
                        # feature_align_group = feature_align_group * (len(self.freqfusions) + 1),
                        feature_align_group = 4,
                        use_channel_att=False,
                        use_dyedgeconv=False,
                        hf_att=False,
                        up_group=1,
                        encoder_kernel=3,
                        encoder_dilation=1,
                        compressed_channels=(conv_dim + conv_dim) // 8, 
                        semi_conv=True,
                        use_spatial_gate=False,  ###
                        upsample_mode='nearest', 
                        align_corners=False, 
                        hr_residual=True, 
                        use_spatial_suppression=False, 
                        use_spatial_residual=False,
                        use_high_pass=True, 
                        use_low_pass=True
                        )
                )
            # self.freqfusion = nn.ModuleList(freqfusion)
        else:
            self.freqfusions = nn.ModuleList([
                FreqFusion(lr_channels = conv_dim, hr_channels = conv_dim, scale_factor=1, up_kernel=3, up_group=1, 
                            upsample_mode='nearest', align_corners=False, use_encoder2=False, hr_residual=True, 
                            use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=64,
                            use_high_pass=True, use_low_pass=True)
                ])
        # for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            # lateral_norm = get_norm(norm, conv_dim)
            # output_norm = get_norm(norm, conv_dim)

            # lateral_conv = Conv2d(
            #     in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            # )
            # output_conv = Conv2d(
            #     conv_dim,
            #     conv_dim,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     bias=use_bias,
            #     norm=output_norm,
            #     activation=F.relu,
            # )
            # weight_init.c2_xavier_fill(lateral_conv)
            # weight_init.c2_xavier_fill(output_conv)
            # self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            # self.add_module("layer_{}".format(idx + 1), output_conv)

            # lateral_convs.append(lateral_conv)
            # output_convs.append(output_conv)
            # freqfusion = 

            # self.freqfusions.append(DyHPFusion2(channels=conv_dim * 2, scale_factor=1, up_kernel=3, up_group=1, 
            #                 upsample_mode='nearest', align_corners=False, use_encoder2=False, hr_residual=True, 
            #                 use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=64,
            #                 use_high_pass=True, use_low_pass=True))

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        # self.lateral_convs = lateral_convs[::-1]
        # self.output_convs = output_convs[::-1]
        self.freqfusions = self.freqfusions[::-1]

    @autocast(enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        #####
        # print(len(srcs))
        # for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
        for idx, f in enumerate(srcs[:-1]): # low res to high
            # print(srcs[idx].shape, srcs[idx + 1].shape)
            _, hr_feat, lr_feat = self.freqfusions[idx](hr_feat=srcs[idx + 1], lr_feat=srcs[idx], use_checkpoint=True)
            srcs[idx + 1] = hr_feat + lr_feat
        #####

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        # for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
        #     x = features[f].float()
        #     lateral_conv = self.lateral_convs[idx]
        #     output_conv = self.output_convs[idx]
        #     cur_fpn = lateral_conv(x)
        #     pre_feat = out[-1]
        #     # print('cur_fpn', cur_fpn.shape, 'pre_feat', pre_feat.shape)
        #     # Following FPN implementation, we use nearest upsampling here
        #     # y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
        #     freqfusion = self.freqfusions[idx]
        #     # freqfusion = freqfusion.to(cur_fpn.device)
        #     _, cur_fpn, pre_feat = freqfusion(hr_feat=cur_fpn, lr_feat=pre_feat, use_checkpoint=False)
        #     y = cur_fpn + pre_feat
        #     y = output_conv(y)
        #     out.append(y)

        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            # print(idx, x.shape)
            # print('self.lateral_convs', self.lateral_convs)
            # print('self.output_convs', self.output_convs)
            lateral_conv = self.lateral_convs[idx]
            # print('lateral_conv', lateral_conv)
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features


### 3xFreqFusion Version
@SEM_SEG_HEADS_REGISTRY.register()
class FreqAwareMSDeformAttnPixelDecoder2(MSDeformAttnPixelDecoder):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        conv_dim = kwargs.get('conv_dim', 256)
        self.freqfusions = nn.ModuleList()
        if self.transformer_num_feature_levels > 1:
            # from low resolution to high resolution (res5 -> res2)
            # for in_channels in transformer_in_channels[::-1]:
            # for _ in range(self.transformer_num_feature_levels - 1):
            for _ in range(3):
                # self.freqfusions.append(
                #     FreqFusion(lr_channels = conv_dim, hr_channels = conv_dim, scale_factor=1, up_kernel=3, up_group=1, 
                #         upsample_mode='nearest', align_corners=False, use_encoder2=False, hr_residual=True, 
                #         use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=64,
                #         use_high_pass=True, use_low_pass=True)
                # )
                self.freqfusions.append(
                    FreqFusion2encoder(lr_channels = conv_dim, hr_channels = conv_dim, 
                        scale_factor=1,
                        lowpass_kernel=5,
                        highpass_kernel=3,
                        lowpass_pad = 0,
                        highpass_pad = 0,
                        padding_mode = 'replicate',
                        hamming_window = False,
                        comp_feat_upsample = True,
                        feature_align = True,
                        # feature_align_group = feature_align_group * (len(self.freqfusions) + 1),
                        feature_align_group = 4,
                        use_channel_att=False,
                        use_dyedgeconv=False,
                        hf_att=False,
                        up_group=1,
                        encoder_kernel=3,
                        encoder_dilation=1,
                        compressed_channels=(conv_dim + conv_dim) // 8, 
                        semi_conv=True,
                        use_spatial_gate=False,  ###
                        upsample_mode='nearest', 
                        align_corners=False, 
                        hr_residual=True, 
                        use_spatial_suppression=False, 
                        use_spatial_residual=False,
                        use_high_pass=True, 
                        use_low_pass=True
                        )
                )
            # self.freqfusion = nn.ModuleList(freqfusion)
        else:
            self.freqfusions = nn.ModuleList([
                FreqFusion(lr_channels = conv_dim, hr_channels = conv_dim, scale_factor=1, up_kernel=3, up_group=1, 
                            upsample_mode='nearest', align_corners=False, use_encoder2=False, hr_residual=True, 
                            use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=64,
                            use_high_pass=True, use_low_pass=True)
                ])
        # for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            # lateral_norm = get_norm(norm, conv_dim)
            # output_norm = get_norm(norm, conv_dim)

            # lateral_conv = Conv2d(
            #     in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            # )
            # output_conv = Conv2d(
            #     conv_dim,
            #     conv_dim,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     bias=use_bias,
            #     norm=output_norm,
            #     activation=F.relu,
            # )
            # weight_init.c2_xavier_fill(lateral_conv)
            # weight_init.c2_xavier_fill(output_conv)
            # self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            # self.add_module("layer_{}".format(idx + 1), output_conv)

            # lateral_convs.append(lateral_conv)
            # output_convs.append(output_conv)
            # freqfusion = 

            # self.freqfusions.append(DyHPFusion2(channels=conv_dim * 2, scale_factor=1, up_kernel=3, up_group=1, 
            #                 upsample_mode='nearest', align_corners=False, use_encoder2=False, hr_residual=True, 
            #                 use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=64,
            #                 use_high_pass=True, use_low_pass=True))

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        # self.lateral_convs = lateral_convs[::-1]
        # self.output_convs = output_convs[::-1]
        self.freqfusions = self.freqfusions[::-1]

    @autocast(enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        #####
        # print(len(srcs))
        # for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
        for idx, f in enumerate(srcs[:-1]): # low res to high
            # print(idx, srcs[idx].shape, srcs[idx + 1].shape)
            _, hr_feat, lr_feat = self.freqfusions[idx](hr_feat=srcs[idx + 1], lr_feat=srcs[idx], use_checkpoint=False)
            srcs[idx + 1] = hr_feat + lr_feat
        #####

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]).contiguous())

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            # print(idx)
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            pre_feat = out[-1]
            # print('cur_fpn', cur_fpn.shape, 'pre_feat', pre_feat.shape)
            # Following FPN implementation, we use nearest upsampling here
            # y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            # freqfusion = 
            # freqfusion = freqfusion.to(cur_fpn.device)
            _, cur_fpn, pre_feat = self.freqfusions[-1](hr_feat=cur_fpn, lr_feat=pre_feat, use_checkpoint=False)
            y = cur_fpn + pre_feat
            y = output_conv(y)
            out.append(y)

        # for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
        #     x = features[f].float()
        #     # print(idx, x.shape)
        #     # print('self.lateral_convs', self.lateral_convs)
        #     # print('self.output_convs', self.output_convs)
        #     lateral_conv = self.lateral_convs[idx]
        #     # print('lateral_conv', lateral_conv)
        #     output_conv = self.output_convs[idx]
        #     cur_fpn = lateral_conv(x)
        #     # Following FPN implementation, we use nearest upsampling here
        #     y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
        #     y = output_conv(y)
        #     out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features
