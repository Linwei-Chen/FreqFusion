import math
from turtle import forward
# from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# from mmcv.cnn import ConvModule
# import matplotlib.pyplot as plt
# from PIL import Image
# import os
# from mmcv.ops.carafe import CARAFEPack


class Adapad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding
        self.reflect_pad = nn.ReflectionPad2d(padding=self.padding)  
        self.repeat_pad = nn.ReplicationPad2d(padding=self.padding)

    def forward(self, x):
        if self.training:
            return self.reflect_pad(x)
        else:
            return self.repeat_pad(x)

def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        
class LocalPixelRelationConv(nn.Module):
    """
    Low-pass filter (dilation = 1) and local relation (dilation = 2, 4, 8 & etc) 
    can be implemented with the same function by using different dilations.

    """
    def __init__(self, in_channels, kernel_size=3, stride=1, groups=1, dilation=1, 
                use_adaptive_residual=False, use_channel=False, 
                # pad_mode='ada'
                pad_mode='replicate'
                # pad_mode='reflect'
                ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups
        self.dilation = dilation
        self.pad = int((kernel_size-1) // 2 * self.dilation)
        self.use_channel = use_channel
        self.use_adaptive_residual = use_adaptive_residual
        self.pad_mode = pad_mode
        # reflect replicate

        # self.weight_net = ConvModule(
        #     in_channels,
        #     groups * kernel_size ** 2,
        #     kernel_size,
        #     padding=self.pad,
        #     dilation=self.dilation,
        #     conv_cfg=None,
        #     # norm_cfg=dict(type='BN', requires_grad=True),
        #     norm_cfg=dict(type='SyncBN', requires_grad=True),
        #     # act_cfg=dict(type='ReLU')
        #     act_cfg=None
        # )
        if self.pad_mode == 'reflect':
            self.conv_pad = nn.ReflectionPad2d(padding=self.pad)  
        elif self.pad_mode == 'replicate':
            self.conv_pad = nn.ReplicationPad2d(padding=self.pad)
        elif self.pad_mode == 'ada':
            self.conv_pad = Adapad(padding=self.pad)
        else:
            raise NotImplementedError
        print(self.conv_pad)
        self.weight_net = nn.Sequential(
            # conv_pad,
            nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride, dilation=self.dilation,
                    kernel_size=kernel_size, bias=False, padding=0, groups=groups),
            # nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride, dilation=self.dilation,
            #         kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
            # nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            nn.SyncBatchNorm(self.groups * kernel_size ** 2), 
            # nn.ReLU(True)
            # nn.Softmax(dim=1)
        )
        # self.weight_net[0].weight[:, :, 1, 1] = 1.
        # xavier_init(self.weight_net[0], distribution='uniform')

        if self.use_adaptive_residual:
            self.use_adaptive_residual = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            # nn.Conv2d(in_channels=in_channels, out_channels= in_channels // 4, kernel_size=1, bias=False),
            # nn.BatchNorm2d(in_channels // 4), 
            # nn.SyncBatchNorm(in_channels // 4), 
            # nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels = 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        if use_channel:
            self.channel_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Conv2d(in_channels=in_channels, out_channels= in_channels // 4, kernel_size=1, bias=False),
                # nn.BatchNorm2d(in_channels // 4), 
                nn.SyncBatchNorm(in_channels // 4), 
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels // 4, out_channels = groups * kernel_size ** 2, kernel_size=1, bias=False),
                # nn.Sigmoid()
            )

        # nn.init.xavier_uniform(self.channel_net[0].weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.weight_net[0].weight, mode='fan_out', nonlinearity='relu')

    def _forward(self, x, feat=None):
        """
        x for generate pixel relations
        """
        _, _, h, w = x.shape
        oh = (h - 1) // self.stride + 1
        ow = (w - 1) // self.stride + 1
        # weight = self.weight_net(x) 
        weight = self.weight_net(self.conv_pad(x)) 
        if self.use_channel:
            weight = weight * self.channel_net(x)

        pixel_relation = weight

        if feat is not None:
            assert x.size(-1) == feat.size(-1)
            assert x.size(-2) == feat.size(-2)
            b, c, _, _ = feat.shape

            weight = weight.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow) 
            weight = weight.repeat(1, 1, c // self.groups, 1, 1, 1)

            if self.use_channel:
                tmp = self.channel_net(x).reshape(b, self.groups, c // self.groups, 1, 1, 1)
                # tmp[tmp < 1.] = tmp[tmp < 1.] ** 2
                # print(weight.shape)
                weight = weight * tmp
            weight = weight.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
            weight = weight.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
            # _weight_copy = weight[0, 0, 0, 0, 0, :, :].clone()
            # print(weight[0, 0, 0, 0, 0, :, :])
            # weight = torch.inverse(weight)
            # print(weight[0, 0, 0, 0, 0, :, :])
            # print(weight[0, 0, 0, 0, 0, :, :] @ _weight_copy)

            # pad_feat = F.pad(feat, pad=[self.pad] * 4, mode='reflect')
            # pad_feat = F.pad(feat, pad=[self.pad] * 4, mode=self.pad_mode)
            pad_feat = self.conv_pad(feat)
            # shape:  B x C x H // stride x W //stride x ksize x ksize
            # pad_x = pad_x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
            # shape:  B x (C x ksize x ksize) x H // stride x W //stride
            pad_feat = F.unfold(pad_feat, kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride, dilation=self.dilation)
            # print(pad_x.shape)
            pad_feat = pad_feat.reshape(b, self.groups, c // self.groups, self.kernel_size, self.kernel_size, oh, ow).permute(0, 1, 2, 5, 6, 3, 4)
            # pad_x = pad_x.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
            # print(pad_x.shape)
            res = weight * pad_feat
            res = res.sum(dim=(-1, -2)).reshape(b, c, oh, ow)

            if self.use_adaptive_residual:
                pixel_weight = self.use_adaptive_residual(x) 
                # print(pixel_weight)
                res = pixel_weight * res + (1 - pixel_weight) * feat

            # channel_weight = self.channel_net(x).pow(2).log()
            # channel_weight = self.channel_net(x).sigmoid()
            # res = res * channel_weight
            return pixel_relation, res
        else:
            return pixel_relation

    def forward(self, x, feat=None, use_checkpoint=False):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, feat)
        else:
            return self._forward(x, feat)


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class Downsample_PASA_group_softmax(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=8):
        super(Downsample_PASA_group_softmax, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, feat=None, use_checkpoint=False):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        out = x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]
        if feat is not None:
            return out, out
        else:
            return out
    # def forward(self, x, feat=None, use_checkpoint=False):
        # return self._forward, 

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
        
class DCTSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = dct.dct_2d(avg_out)
        max_out = dct.dct_2d(avg_out)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = dct.idct_2d(x)
        return self.sigmoid(x)
    
def _test_LocalPixelRelationConv():    
    x = torch.rand(2, 9, 16, 16).cuda()
    # m = LocalPixelRelationConv(in_channels=9, kernel_size=3, dilation=2, use_channel=True, use_adaptive_residual=True).cuda()
    m = LocalPixelRelationConv(in_channels=9, kernel_size=3, dilation=2, use_channel=False, use_adaptive_residual=True).cuda()
    y = m(x, x)
    print(y[0].shape)

def _test_carafe():    
    x = torch.rand(2, 3, 15, 15)
    m = CARAFEPack(channels=3, scale_factor=1)
    y = m(x)
    print(y.shape)

def _test_flops():
    from mmcv.cnn import get_model_complexity_info
    input_shape = (3, 1024, 2048)
    # (K_h * K_w * C_in * C_out) * (H_out * W_out) + (K_h * K_w * C_in) * (H_out * W_out)
    model = LocalPixelRelationConv(in_channels=3, kernel_size=3, dilation=2).cuda()
    # model = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, dilation=2).cuda()
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


def _test_adapad():
    x = torch.rand(1, 1, 3, 3)
    print(x)
    p = Adapad(2)
    print(p(x))
    p.eval()
    print(p(x))


if __name__ == '__main__':
    # _test_flops()
    # _test_LocalPixelRelationConv()
    _test_adapad()
    pass
