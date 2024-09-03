import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

        print('spatial', self.spatial)
        print('S', self.S)
        print('D', self.D)
        print('R', self.R)
        print('train_steps', self.train_steps)
        print('eval_steps', self.eval_steps)
        print('inv_t', self.inv_t)
        print('eta', self.eta)
        print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=None,
            act_cfg=None
        )

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


@HEADS.register_module()
class LightHamHead(BaseDecodeHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 **kwargs):
        print(kwargs)
        super(LightHamHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [resize(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output

    def _forward_feature(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [resize(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        # output = self.cls_seg(output)
        return output
    

from .FreqFusion import FreqFusion
@HEADS.register_module()
class LightHamHeadFreqAware(LightHamHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                use_high_pass=True, 
                use_low_pass=True,
                compress_ratio=8,
                semi_conv=True,
                low2high_residual=False,
                high2low_residual=False,
                lowpass_kernel=5,
                highpass_kernel=3,
                hamming_window=False,
                feature_resample=True,
                feature_resample_group=4,
                comp_feat_upsample=True,
                use_checkpoint=False,
                feature_resample_norm=True,
                **kwargs):
        super().__init__(**kwargs)
        self.freqfusions = nn.ModuleList()
        in_channels = kwargs.get('in_channels', [])
        self.feature_resample = feature_resample
        self.feature_resample_group = feature_resample_group
        self.use_checkpoint = use_checkpoint
        # from lr to hr
        in_channels = in_channels[::-1]
        pre_c = in_channels[0]
        for c in in_channels[1:]:
            freqfusion = FreqFusion(
                hr_channels=c, lr_channels=pre_c, scale_factor=1, lowpass_kernel=lowpass_kernel, highpass_kernel=highpass_kernel, up_group=1, 
                upsample_mode='nearest', align_corners=False, 
                feature_resample=feature_resample, feature_resample_group=feature_resample_group,
                comp_feat_upsample=comp_feat_upsample,
                hr_residual=True, 
                hamming_window=hamming_window,
                compressed_channels= (pre_c + c) // compress_ratio,
                use_high_pass=use_high_pass, use_low_pass=use_low_pass, semi_conv=semi_conv, 
                feature_resample_norm=feature_resample_norm,
                )                
            self.freqfusions.append(freqfusion)
            pre_c += c

        # from lr to hr
        assert not (low2high_residual and high2low_residual)
        self.low2high_residual = low2high_residual
        self.high2low_residual = high2low_residual
        if low2high_residual:
            self.low2high_convs = nn.ModuleList()
            pre_c = in_channels[0]
            for c in in_channels[1:]:
                self.low2high_convs.append(nn.Conv2d(pre_c, c, 1))
                pre_c = c
        elif high2low_residual:
            self.high2low_convs = nn.ModuleList()
            pre_c = in_channels[0]
            for c in in_channels[1:]:
                self.high2low_convs.append(nn.Conv2d(c, pre_c, 1))
                pre_c += c

    def _forward_feature(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        # inputs = [resize(
        #     level,
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners
        # ) for level in inputs]

        # from low res to high res
        inputs = inputs[::-1]
        in_channels = self.in_channels[::-1]
        lowres_feat = inputs[0]
        if self.low2high_residual:
            for pre_c, hires_feat, freqfusion, low2high_conv in zip(in_channels[:-1], inputs[1:], self.freqfusions, self.low2high_convs):
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=self.use_checkpoint)
                lowres_feat = torch.cat([hires_feat + low2high_conv(lowres_feat[:, :pre_c]), lowres_feat], dim=1)
            pass
        else:
            for idx, (hires_feat, freqfusion) in enumerate(zip(inputs[1:], self.freqfusions)):
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=self.use_checkpoint)
                if self.feature_resample:
                    b, _, h, w = hires_feat.shape
                    lowres_feat = torch.cat([hires_feat.reshape(b * self.feature_resample_group, -1, h, w), 
                                             lowres_feat.reshape(b * self.feature_resample_group, -1, h, w)], dim=1).reshape(b, -1, h, w)
                else:
                    lowres_feat = torch.cat([hires_feat, lowres_feat], dim=1)

        # inputs = torch.cat(inputs, dim=1)
        inputs = lowres_feat
        x = self.squeeze(inputs)
        x = self.hamburger(x)
        output = self.align(x)

        # output = self.cls_seg(output)
        return output

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    
@HEADS.register_module()
class LightHamHeadFreqAwareDysample(LightHamHeadFreqAware):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.dysample_list = nn.ModuleList()
        self.dysample_list.append(DySample(in_channels=self.channels, scale=2, style='lp', groups=4, dyscope=True))
        self.dysample_list.append(DySample(in_channels=self.channels, scale=2, style='lp', groups=4, dyscope=True))
        # self.dysample_list.append(DySample(in_channels=self.channels, scale=2, style='lp', groups=4, dyscope=True))

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.dysample_list[0](output)
        output = self.dysample_list[1](output)
        # output = self.dysample_list[2](output)
        output = self.cls_seg(output)
        return output




import torch
from torch import nn
import torch.nn.functional as F
from mmcv.ops.carafe import carafe
from mmcv.cnn import xavier_init


class GateGenerator(nn.Module):
    def __init__(self, in_channels):
        super(GateGenerator, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.weights_init_random()

    def forward(self, x):
        return torch.sigmoid(F.interpolate(self.conv(x), scale_factor=2))

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class Aligner(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Aligner, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.weights_init_random()

    def forward(self, x):
        return self.conv(x)

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class SemiShift(nn.Module):
    def __init__(self, in_channels_en, in_channels_de, out_channels, embedding_dim=64, kernel_size=3):
        super(SemiShift, self).__init__()
        self.compressor_en = nn.Conv2d(in_channels_en, embedding_dim, kernel_size=1)
        self.compressor_de = nn.Conv2d(in_channels_de, embedding_dim, kernel_size=1, bias=False)
        self.content_encoder = nn.Conv2d(embedding_dim, out_channels, kernel_size=kernel_size,
                                         padding=kernel_size // 2)
        self.weights_init_random()

    def forward(self, en, de):
        enc = self.compressor_en(en)
        dec = self.compressor_de(de)
        output = self.content_encoder(enc) + F.interpolate(self.content_encoder(dec), scale_factor=2)
        return output

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class SemiShiftDepthWise(nn.Module):
    def __init__(self, in_channels_en, in_channels_de, out_channels, kernel_size=3):
        super(SemiShiftDepthWise, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.compressor_en = nn.Conv2d(in_channels_en, out_channels, kernel_size=1)
        self.compressor_de = nn.Conv2d(in_channels_de, out_channels, kernel_size=1, bias=False)
        self.content_encoder = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                                         padding=kernel_size // 2, groups=out_channels)
        self.weights_init_random()

    def forward(self, en, de):
        enc = self.compressor_en(en)
        dec = self.compressor_de(de)
        output = self.content_encoder(enc) + F.interpolate(self.content_encoder(dec), scale_factor=2)
        return output

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class KernelGenerator(nn.Module):
    def __init__(self, in_channels_en, in_channels_de, conv, up_kernel_size=5):
        super(KernelGenerator, self).__init__()
        self.conv = conv(in_channels_en, in_channels_de, out_channels=up_kernel_size ** 2)

    def forward(self, en, de):
        return F.softmax(self.conv(en, de), dim=1)


class SemiShiftUp(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(SemiShiftUp, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.ker_generator = SemiShift(in_channels_en, in_channels_de, out_channels=up_kernel_size ** 2)

    def forward(self, en, de):
        kernels = F.softmax(self.ker_generator(en, de), dim=1)
        return carafe(de, kernels, self.up_kernel_size, 1, self.scale)


class FADE(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(FADE, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.gate_generator = GateGenerator(in_channels_de)
        # self.aligner = Aligner(in_channels_en, in_channels_de)
        self.ker_generator = SemiShift(in_channels_en, in_channels_de,
                                       out_channels=up_kernel_size ** 2)

    def forward(self, en, de, mapping=nn.Identity()):
        gate = self.gate_generator(de)
        kernels = F.softmax(self.ker_generator(en, de), dim=1)
        return gate * mapping(en) + (1 - gate) * carafe(de, kernels, self.up_kernel_size, 1, self.scale)


class FADELite(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(FADELite, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.gate_generator = GateGenerator(in_channels_de)
        # self.aligner = Aligner(in_channels_en, in_channels_de)
        self.ker_generator = SemiShiftDepthWise(in_channels_en, in_channels_de,
                                                out_channels=up_kernel_size ** 2)

    def forward(self, en, de, mapping=nn.Identity()):
        gate = self.gate_generator(de)
        kernels = F.softmax(self.ker_generator(en, de), dim=1)
        return gate * mapping(en) + (1 - gate) * carafe(de, kernels, self.up_kernel_size, 1, self.scale)

@HEADS.register_module()
class LightHamHeadFADE(LightHamHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                # use_high_pass=True, 
                # use_low_pass=True,
                # compress_ratio=8,
                # semi_conv=True,
                # two_kernel_encoder=True,
                # low2high_residual=False,
                # high2low_residual=False,
                **kwargs):
        super().__init__(**kwargs)
        self.fades = nn.ModuleList()
        in_channels = kwargs.get('in_channels', [])
        _c3, _c4, _c5 = in_channels

        self.fades.append(FADE(in_channels_en=_c4, in_channels_de=_c5))
        self.fades.append(FADE(in_channels_en=_c3, in_channels_de=_c5))
        self.fades.append(FADE(in_channels_en=_c3, in_channels_de=_c4))

        self.conv1x1s = nn.ModuleList()
        self.conv1x1s.append(nn.Conv2d(_c4, _c5, 1))
        self.conv1x1s.append(nn.Conv2d(_c3, _c5, 1))
        self.conv1x1s.append(nn.Conv2d(_c3, _c4, 1))

            
    def _forward_feature(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        x3, x4, x5 = inputs

        x5 = self.fades[0](en=x4, de=x5, mapping=self.conv1x1s[0])
        x5 = self.fades[1](en=x3, de=x5, mapping=self.conv1x1s[1])
        x4 = self.fades[2](en=x3, de=x4, mapping=self.conv1x1s[2])

        inputs = torch.cat([x3, x4, x5], dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        # output = self.cls_seg(output)
        return output

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    



@HEADS.register_module()
class LightHamHeadFreqAware2(LightHamHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                use_high_pass=True, 
                use_low_pass=True,
                compress_ratio=8,
                semi_conv=True,
                two_kernel_encoder=True,
                low2high_residual=False,
                high2low_residual=False,
                **kwargs):
        super().__init__(**kwargs)
        self.freqfusions = nn.ModuleList()
        in_channels = kwargs.get('in_channels', [])
        _c3, _c4, _c5 = in_channels

        self.freqfusions.append(
            FreqFusion2encoder(hr_channels=_c4, lr_channels=_c5, scale_factor=1, lowpass_kernel=5, highpass_kernel=3, up_group=1, 
                                upsample_mode='nearest', align_corners=False,
                                hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, 
                                compressed_channels = (_c4 + _c5) // compress_ratio,
                                use_high_pass=use_high_pass, use_low_pass=use_low_pass, semi_conv=semi_conv))
        self.freqfusions.append(
            FreqFusion2encoder(hr_channels=_c3, lr_channels=_c5, scale_factor=1, lowpass_kernel=5, highpass_kernel=3, up_group=1, 
                                upsample_mode='nearest', align_corners=False,
                                hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, 
                                compressed_channels = (_c3 + _c5) // compress_ratio,
                                use_high_pass=use_high_pass, use_low_pass=use_low_pass, semi_conv=semi_conv))
        self.freqfusions.append(
            FreqFusion2encoder(hr_channels=_c3, lr_channels=_c4, scale_factor=1, lowpass_kernel=5, highpass_kernel=3, up_group=1, 
                                upsample_mode='nearest', align_corners=False,
                                hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, 
                                compressed_channels = (_c3 + _c4) // compress_ratio,
                                use_high_pass=use_high_pass, use_low_pass=use_low_pass, semi_conv=semi_conv))

        self.conv1x1s = nn.ModuleList()
        self.conv1x1s.append(nn.Conv2d(_c4, _c5, 1))
        self.conv1x1s.append(nn.Conv2d(_c3, _c5, 1))
        self.conv1x1s.append(nn.Conv2d(_c3, _c4, 1))
            
    def _forward_feature(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        x3, x4, x5 = inputs

        _, x5_hr, x5_lr = self.freqfusions[0](hr_feat=x4, lr_feat=x5, use_checkpoint=True)
        x5 = self.conv1x1s[0](x5_hr) + x5_lr
        _, x5_hr, x5_lr = self.freqfusions[1](hr_feat=x3, lr_feat=x5, use_checkpoint=True)
        x5 = self.conv1x1s[1](x5_hr) + x5_lr
        _, x4_hr, x4_lr = self.freqfusions[2](hr_feat=x3, lr_feat=x4, use_checkpoint=True)
        x4 = self.conv1x1s[2](x4_hr) + x4_lr

        inputs = torch.cat([x3, x4, x5], dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        # output = self.cls_seg(output)
        return output

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

@HEADS.register_module()
class LightHamHeadDysample(LightHamHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                upsampler='dysample',
                upsampler_group=4,
                **kwargs):
        super().__init__(**kwargs)
        self.freqfusions = nn.ModuleList()
        in_channels = kwargs.get('in_channels', [])
        _c3, _c4, _c5 = in_channels
        if upsampler == 'dysample':
            self.freqfusions.append(DySample(in_channels=_c5, scale=2, style='lp', groups=upsampler_group, dyscope=True))
            self.freqfusions.append(DySample(in_channels=_c5, scale=2, style='lp', groups=upsampler_group, dyscope=True))
            self.freqfusions.append(DySample(in_channels=_c4, scale=2, style='lp', groups=upsampler_group, dyscope=True))
        elif upsampler == 'saliencydysample':
            self.freqfusions.append(SaliencyDySampler(in_channels=_c5, kernel_size=1, out_channels=upsampler_group))
            self.freqfusions.append(SaliencyDySampler(in_channels=_c5, kernel_size=1, out_channels=upsampler_group))
            self.freqfusions.append(SaliencyDySampler(in_channels=_c4, kernel_size=1, out_channels=upsampler_group))
        else:
            raise NotImplementedError
        
            
    def _forward_feature(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        x3, x4, x5 = inputs

        x5 = self.freqfusions[0](x5)
        x5 = self.freqfusions[1](x5)
        x4 = self.freqfusions[2](x4)

        inputs = torch.cat([x3, x4, x5], dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        # output = self.cls_seg(output)
        return output

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
