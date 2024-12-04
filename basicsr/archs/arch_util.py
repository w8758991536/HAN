import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


def pad_image(x):
    # 获取输入图像的高度和宽度
    height, width = x.size(2), x.size(3)

    # 计算需要填充的高度和宽度
    pad_height = (2 - height % 2) % 2  # 使高度为 2 的倍数
    pad_width = (2 - width % 2) % 2  # 使宽度为 2 的倍数

    # 使用对称填充方式在图像边缘填充
    x_padded = F.pad(x, (0, pad_width, 0, pad_height), mode='reflect')

    return x_padded, pad_height, pad_width


def remove_padding(x, pad_height, pad_width):
    # 移除之前添加的填充
    if pad_height > 0:
        x = x[:, :, :-pad_height, :]
    if pad_width > 0:
        x = x[:, :, :, :-pad_width]
    return x


# def dwt_init(x):
#     x01 = x[:, :, 0::2, :] / 2
#     x02 = x[:, :, 1::2, :] / 2
#     x1 = x01[:, :, :, 0::2]
#     x2 = x02[:, :, :, 0::2]
#     x3 = x01[:, :, :, 1::2]
#     x4 = x02[:, :, :, 1::2]
#
#     min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
#     min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))
#
#     x1 = x1[:, :, :min_height, :min_width]
#     x2 = x2[:, :, :min_height, :min_width]
#     x3 = x3[:, :, :min_height, :min_width]
#     x4 = x4[:, :, :min_height, :min_width]
#
#     x_LL = x1 + x2 + x3 + x4
#     x_HL = -x1 - x2 + x3 + x4
#     x_LH = -x1 + x2 - x3 + x4
#     x_HH = x1 - x2 - x3 + x4
#
#     return x_LL, x_HL, x_LH, x_HH



def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        #h
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        #v
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        #d
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.in_channels)



@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

class fcadw(nn.Module):
    def __init__(self, num_feat, reduction=8):
        super(fcadw, self).__init__()

        self.sig = nn.Sigmoid()
        self.fc1 = nn.Sequential(nn.Linear(num_feat, num_feat // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(num_feat // reduction, num_feat, bias=False)
                                )
        self.fc2 = nn.Sequential(nn.Linear(num_feat, num_feat // reduction, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(num_feat // reduction, num_feat, bias=False)
                                 )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()
        y = fft(x)
        x_avg = self.avgpool(x).view(b, c)
        y_avg = self.avgpool(y).view(b, c)
        x_avg = self.fc1(x_avg).view(b, c, 1, 1)
        y_avg = self.fc2(y_avg).view(b, c, 1, 1)
        x_avg = self.sig(x_avg)
        y_avg =self.sig(y_avg)
        y = (x_avg+y_avg) * 0.5
        return x * y

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class one_conv(nn.Module):
    def __init__(self, inchanels, growth_rate, kernel_size=3, relu=True):
        super(one_conv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(inchanels, inchanels, kernel_size=1)
        self.dconv = nn.Conv2d(inchanels, inchanels, kernel_size=kernel_size, padding=1, stride=1, groups=inchanels)
        self.flag = relu
        self.conv1 = nn.Conv2d(inchanels, inchanels, kernel_size=1)
        self.dconv1 = nn.Conv2d(inchanels, inchanels, kernel_size=kernel_size, padding=1, stride=1, groups=inchanels)
        if relu:
            self.relu = nn.PReLU(inchanels)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)

    def forward(self, x):
        if self.flag == False:
            output = self.weight1(x) + self.weight2(self.dconv1(self.conv1(self.dconv(self.conv(x)))))
        else:
            output = self.weight1(x) + self.weight2(self.dconv1(self.conv1(self.relu(self.dconv(self.conv(x))))))
        return output  # torch.cat((x,output),1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0, fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                           padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class one_module(nn.Module):    # ARFB
    def __init__(self, n_feats):
        super(one_module, self).__init__()
        self.layer1 = one_conv(n_feats, n_feats//2,3)
        self.layer2 = one_conv(n_feats, n_feats//2,3)
        # self.layer3 = one_conv(n_feats, n_feats//2,3)
        self.layer4 = BasicConv(n_feats, n_feats, 3,1,1)
        self.layer5 = BasicConv(n_feats, n_feats,3,1,3,dilation=3)
        self.alise = BasicConv(2*n_feats, n_feats, 1,1,0)
        self.atten = fcadw(n_feats)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)
        self.weight3 = Scale(1)
        self.weight4 = Scale(1)
        self.weight5 = Scale(1)
    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # pdb.set_trace()
        x4 = self.layer4(self.layer5(self.atten(self.alise(torch.cat([self.weight2(x2),self.weight3(x1)],1)))))
        return self.weight4(x)+self.weight5(x4)

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups = 1):
    wn = lambda x:torch.nn.utils.weight_norm(x)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups = groups)

class Updownblock(nn.Module):
    def __init__(self, n_feats):
        super(Updownblock, self).__init__()
        self.encoder = one_module(n_feats)
        self.decoder_low = one_module(n_feats) #nn.Sequential(one_module(n_feats),
        #                     one_module(n_feats),
        #                     one_module(n_feats))
        self.decoder_high = one_module(n_feats)
        self.alise = one_module(n_feats)
        self.alise2 = BasicConv(2*n_feats, n_feats, 1, 1, 0) #one_module(n_feats)
        self.att = fcadw(n_feats)

    def forward(self, x):
        x1 = self.encoder(x)

        high = fft(x1)
        for i in range(5):
            x1 = self.decoder_low(x1)
        x3 = x1
        # x3 = self.decoder_low(x2)
        high = self.decoder_high(high)
        return self.alise(self.att(self.alise2(torch.cat([x3, high], dim=1)))) + x



def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale






class WideActiveB(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(WideActiveB, self).__init__()
        self.res_scale = res_scale
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1up = nn.Conv2d(num_feat, num_feat*6, 1, 1)
        self.conv1x1down = nn.Conv2d(num_feat*6, 25, 1, 1)
        self.conv3x3 = nn.Conv2d(25, num_feat, 3, 1, 1)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv3x3(self.conv1x1down(self.relu(self.conv1x1up(x))))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
def fft(x):
    w_h = torch.nn.Parameter(torch.tensor(6.), requires_grad=True)
    w_w = torch.nn.Parameter(torch.tensor(6.), requires_grad=True)
    b, c, h, w = x.shape

    if w % 2 != 0:
        x = torch.nn.functional.pad(x, (1, 0))
    if h % 2 != 0:
        x = torch.nn.functional.pad(x, (0, 0, 1, 0))
    b, c, h, w = x.shape
    w_h = abs(w_h).to(torch.int)
    w_w = abs(w_w).to(torch.int)
    x = torch.fft.fftshift(x, dim=[2, 3])
    # mask = torch.ones(batch,channel,high,wid)
    hc, wc = int(h/2), int(w/2)
    # mask[:,:,hc-10:hc+10,wc-10:wc+10] = 0.
    a, b = h//w_h, w//w_w
    x[:, :, hc-a:hc+a, wc-b:wc+b] = 0.
    x = torch.fft.ifftshift(x, dim=[2, 3])

    return x
class FFT(nn.Module):
    def __init__(self):
        super(FFT, self).__init__()
        self.w_h = torch.nn.Parameter(torch.tensor(6.), requires_grad=True)
        self.w_w = torch.nn.Parameter(torch.tensor(6.), requires_grad=True)

    def forward(self, x):

        b, c, h, w = x.shape

        if w % 2 != 0:
            x = torch.nn.functional.pad(x, (1, 0))
        if h % 2 != 0:
            x = torch.nn.functional.pad(x, (0, 0, 1, 0))
        b, c, h, w = x.shape
        w_h = abs(self.w_h).to(torch.int)
        w_w = abs(self.w_w).to(torch.int)
        x = torch.fft.fftshift(x, dim=[2, 3])
        # mask = torch.ones(batch,channel,high,wid)
        hc, wc = int(h / 2), int(w / 2)
        # mask[:,:,hc-10:hc+10,wc-10:wc+10] = 0.
        a, b = h // w_h, w // w_w
        x[:, :, hc - a:hc + a, wc - b:wc + b] = 0.
        x = torch.fft.ifftshift(x, dim=[2, 3])

        return x



class My_CA(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, reduction=1, pytorch_init=False):
        super(My_CA, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(nn.Linear(num_feat, num_feat // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(num_feat // reduction, num_feat, bias=False)
                                )
        self.sig = nn.Sigmoid()

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        b, c, h, w = out.size()
        y = fft(out)
        st = torch.std(y, dim=[2, 3], keepdim=True, unbiased=True).view(b, c)
        me = torch.mean(y, dim=[2, 3], keepdim=True).view(b, c)
        me = self.fc(me).view(b, c, 1, 1)
        st = self.fc(st).view(b, c, 1, 1)
        y = self.sig(st + me)
        out = out * y
        return identity + out

class My_CA_g1(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, reduction=4, pytorch_init=False):
        super(My_CA_g1, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(nn.Linear(num_feat, num_feat // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(num_feat // reduction, num_feat, bias=False)
                                )
        self.sig = nn.Sigmoid()

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        b, c, h, w = out.size()
        y = fft(out)
        st = torch.std(y, dim=[2, 3], keepdim=True, unbiased=True).view(b, c)
        st = self.fc(st).view(b, c, 1, 1)
        st = self.sig(st)
        out = out * st
        return identity + out


class conv2formerL(nn.Module):
    def __init__(self, dim):
        super(conv2formerL, self).__init__()
        self.conv1a = nn.Conv2d(dim, dim, 1)
        self.attention = nn.Conv2d(dim, dim, 1)

        # self.attconv1 = nn.Conv2d(dim, dim//3, 1)
        # self.branch0 = nn.Conv2d(dim//3, dim//3, 5, 1, 2, groups=dim//3)
        # self.branch1 = nn.Conv2d(dim // 3, dim // 3, 5, 1, 4, groups=dim // 3, dilation=2)
        # self.branch2 = nn.Conv2d(dim // 3, dim // 3, 5, 1, 6, groups=dim // 3, dilation=3)
        # self.attconv2 = nn.Conv2d((dim//3)*3, dim, 1)

        self.dconv2 = nn.Conv2d(dim, dim, 11, 1, 5, groups=dim)
        self.dfconv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))
        self.gelu = nn.GELU()
    def forward(self,x):
        w = self.attention(x)

        x = self.dconv2(self.conv1a(x))
        x = self.gelu(x * w)
        x = self.dfconv(x)
        return x

class conv2formerB(nn.Module):
    def __init__(self, dim):
        super(conv2formerB, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(dim, dim*4, 1),
                                  nn.GELU(),
                                  nn.Conv2d(dim*4, dim*4, 3, 1, 1, groups=dim*4),
                                  nn.GELU(),
                                  nn.Conv2d(dim*4, dim, 1))
        self.conv2f = conv2formerL(dim=dim)
        self.act = nn.GELU()




    def forward(self, x):
        b, c, h, w = x.size()
        out = self.conv(F.layer_norm(x, [c, h, w])) + x
        out = self.conv2f(F.layer_norm(out, [c, h, w])) + out
        return out

class ESA(nn.Module):
    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = n_feats // 4
        n = f // 2

        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.branch0 = nn.Sequential(nn.Conv2d(f, n, 1),
                                     nn.ReLU(True),
                                     nn.Conv2d(n, n, 3, 1, 1))
        self.branch1 = nn.Sequential(nn.Conv2d(f, n, 1, 1),
                                     nn.ReLU(True),
                                     nn.Conv2d(n, n, kernel_size=(3, 1), stride=1, padding=(1,0)),
                                     nn.ReLU(True),
                                     nn.Conv2d(n, n, 3, 1, 3, dilation=3))
        self.branch2 = nn.Sequential(nn.Conv2d(f, n, 1, 1),
                                     nn.ReLU(True),
                                     nn.Conv2d(n, n, kernel_size=(1,3), stride=1, padding=(0,1)),
                                     nn.ReLU(True),
                                     nn.Conv2d(n, n, 3, 1, 3, dilation=3))
        self.branch3 = nn.Sequential(nn.Conv2d(f, n, 1, 1),
                                     nn.ReLU(True),
                                     nn.Conv2d(n, n, kernel_size=(1, 3), stride=1, padding=(0,1)),
                                     nn.ReLU(True),
                                     nn.Conv2d(n, n, kernel_size=(3, 1), stride=1, padding=(1,0)),
                                     nn.ReLU(True),
                                     nn.Conv2d(n, n, 3, 1, 5, dilation=5))

        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.convdown = nn.Conv2d(4*n, f, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        b0 = self.branch0(v_max)
        b1 = self.branch1(v_max)
        b2 = self.branch2(v_max)
        b3 = self.branch3(v_max)
        c3 = self.convdown(torch.cat([b0, b1, b2, b3], 1))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class SE(nn.Module):
    def __init__(self, nf, ration=4):
        super(SE, self).__init__()
        self.fc = nn.Sequential(nn.Conv2d(nf, nf//ration, 1, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(nf//ration, nf, 1, bias=False)
                                )
        self.sig = nn.Sigmoid()
        self.avg1 = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):


        x_avg = self.avg1(x)
        st = self.fc(x_avg)
        y = self.sig(st)
        out = x * y

        return out

class conv2formerG_SE(nn.Module):
    def __init__(self, dim):
        super(conv2formerG_SE, self).__init__()
        self.c2fb1 = conv2formerB(dim)
        self.c2fb2 = conv2formerB(dim)
        self.c2fb3 = conv2formerB(dim)
        self.c2fb4 = conv2formerB(dim)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv3 = nn.Conv2d(dim, dim // 2, 1)
        self.conv4 = nn.Conv2d(dim, dim // 2, 1)
        self.esa = ESA(dim)
        self.conv = nn.Conv2d(dim*3, dim, 1)
        self.conv3x3 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))
        self.bsconv = nn.Sequential(nn.Conv2d(dim, dim, 1),
                                    nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))
        self.act = nn.GELU()
        self.fca1 = SE(dim)
        self.fca2 = SE(dim)
        self.fca3 = SE(dim)
        self.fca4 = SE(dim)





    def forward(self, x):
        d_c1 = self.conv1(x)
        r_1 = self.fca1(self.c2fb1(x))



        d_c2 = self.conv2(r_1)
        r_2 = self.fca2(self.c2fb2(r_1))

        d_c3 = self.conv3(r_2)
        r_3 = self.fca3(self.c2fb3(r_2))

        d_c4 = self.conv4(r_3)
        r_4 = self.fca4(self.c2fb4(r_3))
        r_4 = self.bsconv(r_4)

        out = torch.cat([d_c1, d_c2, d_c3, d_c4, r_4], dim=1)

        out = self.conv(out)
        out = self.esa(self.conv3x3(out))

        return out + x


class My_SA_fft(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, reduction=4, sain=2, pytorch_init=False):
        super(My_SA_fft, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.branch0 = nn.Sequential(nn.Conv2d(sain, sain, kernel_size=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(sain, sain, kernel_size=3, stride=1, padding=1))

        self.branch1 = nn.Sequential(nn.Conv2d(sain, sain, kernel_size=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(sain, sain, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(sain, sain, kernel_size=3, stride=1, padding=3, dilation=3))

        self.branch2 = nn.Sequential(nn.Conv2d(sain, sain, kernel_size=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(sain, sain, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(sain, sain, kernel_size=3, stride=1, padding=5, dilation=5))

        self.branch3 = nn.Sequential(nn.Conv2d(sain, sain, kernel_size=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(sain, sain, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(sain, sain, kernel_size=3, stride=1, padding=7, dilation=7))

        self.convdown = nn.Conv2d(sain*4, 1, kernel_size=1)
        self.shortcut = nn.Conv2d(sain, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)
    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        y = fft(out)
        ma, _ = torch.max(y, dim=1, keepdim=True)
        avg = torch.mean(y, dim=1, keepdim=True)
        w = torch.cat((ma, avg), dim=1)
        w0 = self.branch0(w)
        w1 = self.branch1(w)
        w2 = self.branch2(w)
        w3 = self.branch3(w)
        w = self.convdown(torch.cat((w0, w1, w2, w3), dim=1))
        short = self.shortcut(y)
        w = self.sig(w + short)
        out = out * w
        return out + identity


class My_SA_new(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, sain=1, pytorch_init=False):
        super(My_SA_new, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.conv7x7 = nn.Conv2d(sain, sain, 7, 1, 3)
    def forward(self,x):
        res = x
        out = self.conv2(self.relu(self.conv1(x)))
        edge = fft(out)
        w = torch.mean(torch.cat((edge, out), dim=1), dim=1, keepdim=True)
        w = self.conv7x7(w)
        w = self.sig(w)
        out = out * w
        return out + res


class No_P_Sa(nn.Module):
    def __init__(self,num_feat=64, res_scale=1, sain=1, pytorch_init=False):
        super(No_P_Sa,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        edge = fft(x)
        F = torch.cat((x, edge), dim=1 )
        F = self.avgpool(F)
        return F * x

class DBlock(nn.Module):
    def __init__(self):
        super(DBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(64, 48, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(48, 32, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.down = nn.Sequential(
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 48, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(48, 80, 3, 1, 1),
            nn.LeakyReLU(inplace=True)

        )
        self.conv1x1 = nn.Conv2d(80, 64, 1, 1, 0)
    def forward(self,x):
        x0 = self.up(x)
        x1, x2 = torch.split(x0, [16, 48], dim=1)
        x1 = torch.cat([x1, x], dim=1)
        x2 = self.down(x2)
        x0 = x2 + x1
        x0 = self.conv1x1(x0)
        return x0

class up(nn.Module):
    def __init__(self, scale,num_feat):
        super(up, self).__init__()

        self.conv1 = nn.Conv2d(num_feat, 3*(scale**2), 3, 1, 1)
        self.up = nn.PixelShuffle(scale)
    def forward(self, x):
        x = self.up(self.conv1(x))
        return x






def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
torch.fft.fftshift

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple



import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTProcessor(nn.Module):
    def __init__(self, w_h=6.0, w_w=6.0):
        """
        使用傅里叶变换对输入进行频域处理。
        :param w_h: 控制屏蔽区域的高度系数，可学习。
        :param w_w: 控制屏蔽区域的宽度系数，可学习。
        """
        super(FFTProcessor, self).__init__()
        # 初始化可学习参数
        self.w_h = nn.Parameter(torch.tensor(w_h, dtype=torch.float32), requires_grad=True)
        self.w_w = nn.Parameter(torch.tensor(w_w, dtype=torch.float32), requires_grad=True)

    # def ensure_even_dimensions(self, x):
    #     """
    #     确保输入的高度和宽度为偶数。
    #     :param x: 输入张量，形状为 (B, C, H, W)。
    #     :return: 调整后的张量。
    #     """
    #     b, c, h, w = x.shape
    #     pad_h = 1 if h % 2 != 0 else 0
    #     pad_w = 1 if w % 2 != 0 else 0
    #     if pad_h > 0 or pad_w > 0:
    #         x = F.pad(x, (0, pad_w, 0, pad_h))
    #     return x

    def forward(self, x):
        """
        前向传播，执行频域屏蔽操作。
        :param x: 输入张量，形状为 (B, C, H, W)。
        :return: 经过频域处理后的张量，形状与输入一致。
        """
        # 确保输入尺寸为偶数
        # x = self.ensure_even_dimensions(x)
        b, c, h, w = x.shape

        # 获取屏蔽区域的大小
        w_h = torch.abs(self.w_h).to(torch.int)
        w_w = torch.abs(self.w_w).to(torch.int)

        # 傅里叶变换
        x_fft = torch.fft.fft2(x, dim=[2, 3])
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=[2, 3])

        # 计算屏蔽区域
        hc, wc = h // 2, w // 2
        a, b = min(hc, h // w_h), min(wc, w // w_w)  # 确保不越界

        # 屏蔽频域中心区域
        x_fft_shifted[:, :, hc - a:hc + a, wc - b:wc + b] = 0

        # 逆傅里叶变换
        x_ifft_shifted = torch.fft.ifftshift(x_fft_shifted, dim=[2, 3])
        x_ifft = torch.fft.ifft2(x_ifft_shifted, dim=[2, 3])

        # 返回实部，确保张量为实数
        x_real = x_ifft.real.to(torch.float32)
        return x_real



import torch
import torch.nn as nn
import torch.nn.functional as F

# class ChannelwiseFFTProcessor(nn.Module):
#     def __init__(self, num_channels, init_w_h=6.0, init_w_w=6.0):
#         """
#         针对每个通道设置独立的可学习频域滤波参数。
#         :param num_channels: 输入的通道数。
#         :param init_w_h: 初始高度方向的滤波参数。
#         :param init_w_w: 初始宽度方向的滤波参数。
#         """
#         super(ChannelwiseFFTProcessor, self).__init__()
#         # 为每个通道初始化独立的滤波参数
#         self.w_h = nn.Parameter(torch.tensor([init_w_h] * num_channels, dtype=torch.float32), requires_grad=True)
#         self.w_w = nn.Parameter(torch.tensor([init_w_w] * num_channels, dtype=torch.float32), requires_grad=True)
#
#     # def ensure_even_dimensions(self, x):
#     #     """
#     #     确保输入的高度和宽度为偶数。
#     #     :param x: 输入张量，形状为 (B, C, H, W)。
#     #     :return: 调整后的张量。
#     #     """
#     #     b, c, h, w = x.shape
#     #     pad_h = 1 if h % 2 != 0 else 0
#     #     pad_w = 1 if w % 2 != 0 else 0
#     #     if pad_h > 0 or pad_w > 0:
#     #         x = F.pad(x, (0, pad_w, 0, pad_h))
#     #     return x
#
#     def forward(self, x):
#         """
#         前向传播，针对每个通道独立执行频域屏蔽操作。
#         :param x: 输入张量，形状为 (B, C, H, W)。
#         :return: 经过频域处理后的张量，形状与输入一致。
#         """
#         # 确保输入尺寸为偶数
#
#         b, c, h, w = x.shape
#
#         # 傅里叶变换
#         x_fft = torch.fft.fft2(x, dim=[2, 3])
#         x_fft_shifted = torch.fft.fftshift(x_fft, dim=[2, 3])
#
#         # 频域屏蔽
#         hc, wc = h // 2, w // 2  # 中心位置
#         for channel in range(c):
#             # 获取当前通道的屏蔽区域大小
#             w_h = torch.abs(self.w_h[channel]).to(torch.int)
#             w_w = torch.abs(self.w_w[channel]).to(torch.int)
#             a, b = min(hc, h // w_h), min(wc, w // w_w)  # 确保不越界
#
#             # 屏蔽频域中心区域
#             x_fft_shifted[:, channel, hc - a:hc + a, wc - b:wc + b] = 0
#
#         # 逆傅里叶变换
#         x_ifft_shifted = torch.fft.ifftshift(x_fft_shifted, dim=[2, 3])
#         x_ifft = torch.fft.ifft2(x_ifft_shifted, dim=[2, 3])
#
#         # 返回实部，确保张量为实数
#         x_real = x_ifft.real.to(torch.float32)
#         return x_real
