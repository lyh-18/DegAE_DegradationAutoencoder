import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN_modulation(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64, degradation_embed_dim=512):
        super(ResidualBlock_noBN_modulation, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #self.modulation_layer = GFM_modulation(degradation_embed_dim, nf, noise=True)
        #print(degradation_embed_dim)
        self.modulation_layer = StyleGAN2_mod(out_channels=nf, modul_channels=degradation_embed_dim, kernel_size=3, noise=True)

    def forward(self, x, embedding):
        #print(x.shape)
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        res = identity + out
        res = self.modulation_layer(res, embedding)
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    if torch.cuda.is_available():
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), bias=bias).cuda()
    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), bias=bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        ).cuda()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, degradation_embed_dim=512):

        super(RCAB, self).__init__()


        self.modules_body1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.modules_body2 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.modules_body3 = CALayer(n_feat, reduction)

        self.act = act


    def forward(self, x):
        res = self.act(self.modules_body1(x))
        res = self.modules_body2(x)
        res = self.modules_body3(x)
        
        res += x

        return res

## Residual Channel Attention Block (RCAB) with modulation
class RCAB_modulation(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, degradation_embed_dim=512):

        super(RCAB_modulation, self).__init__()

        #self.modulation_layer = GFM_modulation(degradation_embed_dim, n_feat, noise=True).cuda()
        #print(degradation_embed_dim)
        self.modulation_layer = StyleGAN2_mod(out_channels=n_feat, modul_channels=degradation_embed_dim, kernel_size=3, noise=True)


        self.modules_body1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.modules_body2 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.modules_body3 = CALayer(n_feat, reduction)

        self.act = act


    def forward(self, x, embedding=None):
        res = self.act(self.modules_body1(x))
        res = self.modules_body2(x)
        res = self.modules_body3(x)
        
        res += x

        res = self.modulation_layer(res, embedding)

        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        #modules_body = []
        self.modules_body1 = RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.modules_body2 = RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.modules_body3 = RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.modules_body4 = RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        
        
        self.after_body = conv(n_feat, n_feat, kernel_size)
        

    def forward(self, x):
        res = self.modules_body1(x)
        res = self.modules_body2(res)
        res = self.modules_body3(res)
        res = self.modules_body4(res)
        res = self.after_body(res)
        res += x
        return res

class ResidualGroup_modulation(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, degradation_embed_dim=512):
        super(ResidualGroup_modulation, self).__init__()
        #modules_body = []
        self.modules_body1 = RCAB_modulation(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, degradation_embed_dim=degradation_embed_dim)
        self.modules_body2 = RCAB_modulation(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, degradation_embed_dim=degradation_embed_dim)
        self.modules_body3 = RCAB_modulation(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, degradation_embed_dim=degradation_embed_dim)
        self.modules_body4 = RCAB_modulation(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, degradation_embed_dim=degradation_embed_dim)
        
        
        self.after_body = conv(n_feat, n_feat, kernel_size)
        

    def forward(self, x, embedding=None):
        res = self.modules_body1(x, embedding)
        res = self.modules_body2(res, embedding)
        res = self.modules_body3(res, embedding)
        res = self.modules_body4(res, embedding)
        res = self.after_body(res)
        res += x
        return res

class GFM_modulation(nn.Module):
    def __init__(self, embed_dim, modul_dim, noise=True):
        super(GFM_modulation, self).__init__()
        print('Modulation Method: GFM_modulation')
        self.GFM_nf = modul_dim
        self.cond_scale = nn.Linear(embed_dim, modul_dim)
        self.cond_shift = nn.Linear(embed_dim, modul_dim)
        self.activation = nn.LeakyReLU(0.1, True)

        self.weight_noise = nn.Parameter(torch.zeros(1))

        self.noise = noise

    def forward(self, x, embedding):
        #print('111', x.shape)
        # x: [B, C, H, W]
        if embedding is not None:
            #print('mod')
            if len(embedding.shape) == 4:
                B, C, _, _ = embedding.shape # [B, 512, 1, 1]
            elif len(embedding.shape) == 2:
                B, C = embedding.shape # [B, 512]
            embedding = embedding.view(B, C) # [B, 512]
            scale = self.cond_scale(embedding) # [B, 64]
            shift = self.cond_shift(embedding) # [B, 64]
            x = x * scale.view(B, self.GFM_nf, 1, 1) + shift.view(B, self.GFM_nf, 1, 1) + x
            x = self.activation(x)

        if self.noise:
            b, _, h, w = x.shape
            n = x.new_empty(b, 1, h, w).normal_()
            x = x + self.weight_noise * n
            x = self.activation(x)
            #print('noise')

        return x

class StyleGAN2_mod(nn.Module):

    def __init__(self, out_channels, modul_channels, kernel_size = 3, noise=True):
        super().__init__()

        print('Modulation Method: StyleGAN2_mod')

        #self.conv = ResBlock(in_channels, out_channels)

        # generate global conv weights
        fan_in = out_channels * kernel_size ** 2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.out_channels = out_channels
        self.scale = 1 / math.sqrt(fan_in)
        self.modulation = nn.Conv2d(modul_channels, self.out_channels, 1)
        self.weight = nn.Parameter(
            torch.randn(1, self.out_channels, self.out_channels, kernel_size, kernel_size)
        )
        self.conv_last = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        self.noise = noise
        if self.noise:        
            self.weight_noise = nn.Parameter(torch.zeros(1))  # for noise injection
        
        self.activation = nn.LeakyReLU(0.1, True)
        
        print('Add random noise layer: ', self.noise)


    def forward(self, x, xg):
        # for global adjustation
        B, _, H, W = x.size()
        
        C = self.out_channels
        if len(xg.shape) == 4: #[B, embed_ch, 1, 1]
            pass
        elif len(xg.shape) == 2: #[B, embed_ch]
            xg = xg.unsqueeze(2).unsqueeze(2) #[B, embed_ch, 1, 1]

        style = self.modulation(xg).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        # demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, C, 1, 1, 1)

        weight = weight.view(
            B * C, C, self.kernel_size, self.kernel_size
        )
        
        
        #x = self.conv(x)
        x = x.contiguous()
        x_input = x.view(1, B * C, H, W)
        x_global = F.conv2d(x_input, weight, padding=self.padding, groups=B)
        x_global = x_global.view(B, C, H, W)
        
        if self.noise:
            b, _, h, w = x_global.shape
            n = x_global.new_empty(b, 1, h, w).normal_()
            x_global += self.weight_noise * n
            

        x = self.conv_last(x_global)
        x = self.activation(x)

        return x

## Residual Channel Attention Network (RCAN)
class RCAN_Head(nn.Module):
    def __init__(self, conv=default_conv, in_c=3, out_c=3, n_feats=64, scale=1, require_modulation=False, degradation_embed_dim=512):
        super(RCAN_Head, self).__init__()
        
        n_resgroups = 1
        n_resblocks = 8
        
        kernel_size = 3
        reduction = 16
        act = nn.ReLU(True)
        
        self.require_modulation = require_modulation
        print('require modulation: ', self.require_modulation)
       
        
        # define head module
        modules_head = [conv(in_c, n_feats, kernel_size)]

        # define body module
        if self.require_modulation:
            self.modules_body = ResidualGroup_modulation(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks, degradation_embed_dim=degradation_embed_dim)
        else:
            self.modules_body = ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks)


        self.after_body = conv(n_feats, n_feats, kernel_size)

        # define tail module
        if scale > 1:
            modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, out_c, kernel_size)]
        elif scale == 1:
            modules_tail = [
            conv(n_feats, out_c, kernel_size)]

        

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, embedding=None):
        
        x = self.head(x)

        if self.require_modulation:
            res = self.modules_body(x, embedding=embedding)
        else:
            res = self.modules_body(x)

        res = self.after_body(res)
        res += x

        x = self.tail(res)
        
        return x 


class MSRResNet_Head(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_c=3, out_c=3, nf=64, scale=1, require_modulation=False, degradation_embed_dim=512):
        super(MSRResNet_Head, self).__init__()
        self.upscale = scale
        self.require_modulation = require_modulation
        print('require modulation: ', self.require_modulation)

        self.conv_first = nn.Conv2d(in_c, nf, 3, 1, 1, bias=True)

        if self.require_modulation:
            self.basic_block1 = ResidualBlock_noBN_modulation(nf=nf, degradation_embed_dim=degradation_embed_dim)
            self.basic_block2 = ResidualBlock_noBN_modulation(nf=nf, degradation_embed_dim=degradation_embed_dim)
            self.basic_block3 = ResidualBlock_noBN_modulation(nf=nf, degradation_embed_dim=degradation_embed_dim)
            self.basic_block4 = ResidualBlock_noBN_modulation(nf=nf, degradation_embed_dim=degradation_embed_dim)
        else:
            self.basic_block1 = ResidualBlock_noBN(nf=nf)
            self.basic_block2 = ResidualBlock_noBN(nf=nf)
            self.basic_block3 = ResidualBlock_noBN(nf=nf)
            self.basic_block4 = ResidualBlock_noBN(nf=nf)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_c, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x, embedding=None):
        out = self.lrelu(self.conv_first(x))

        if self.require_modulation:
            out = self.basic_block1(out, embedding)
            out = self.basic_block2(out, embedding)
            out = self.basic_block3(out, embedding)
            out = self.basic_block4(out, embedding)
        else:
            out = self.basic_block1(out)
            out = self.basic_block2(out)
            out = self.basic_block3(out)
            out = self.basic_block4(out)


        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        
        return out

class MSRResNet_Head_1res(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_c=3, out_c=3, nf=64, scale=1, require_modulation=False, degradation_embed_dim=512):
        super(MSRResNet_Head_1res, self).__init__()
        self.upscale = scale
        self.require_modulation = require_modulation
        print('require modulation: ', self.require_modulation)

        self.conv_first = nn.Conv2d(in_c, nf, 3, 1, 1, bias=True)

        if self.require_modulation:
            self.basic_block1 = ResidualBlock_noBN_modulation(nf=nf, degradation_embed_dim=degradation_embed_dim)
        else:
            self.basic_block1 = ResidualBlock_noBN(nf=nf)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_c, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x, embedding=None):
        out = self.lrelu(self.conv_first(x))

        if self.require_modulation:
            out = self.basic_block1(out, embedding)
        else:
            out = self.basic_block1(out)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        
        return out

class Simple_Head(nn.Module):
    def __init__(self, in_c=3, out_c=3, nf=64, scale=1, require_modulation=False, degradation_embed_dim=512):
        super(Simple_Head, self).__init__()
        self.upscale = scale
        self.require_modulation = require_modulation
        print('Head: Simple_Head')

        self.conv_first = nn.Conv2d(in_c, nf, 3, 1, 1, bias=True)


        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_c, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x, embedding=None):
        out = self.lrelu(self.conv_first(x))

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        
        return out

class One_Conv_Head(nn.Module):
    def __init__(self, in_c=3, out_c=3, nf=64, scale=1, require_modulation=False, degradation_embed_dim=512):
        super(One_Conv_Head, self).__init__()
        self.upscale = scale
        self.require_modulation = require_modulation
        print('Head: One_Conv_Head')
        nf = in_c

        # upsampling
        if self.upscale == 2:
            self.conv_first = nn.Conv2d(in_c, 64, 3, 1, 1, bias=True)
            self.upconv1 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(64, out_c, 3, 1, 1, bias=True)
        elif self.upscale == 3:
            self.conv_first = nn.Conv2d(in_c, 64, 3, 1, 1, bias=True)
            self.upconv1 = nn.Conv2d(64, 64 * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
            self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(64, out_c, 3, 1, 1, bias=True)
        elif self.upscale == 4:
            self.conv_first = nn.Conv2d(in_c, 64, 3, 1, 1, bias=True)
            self.upconv1 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(64, out_c, 3, 1, 1, bias=True)
        elif self.upscale == 1:
            self.conv_last = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=True)
            self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x, embedding=None):
        out = x
        if self.upscale == 4:
            out = self.lrelu(self.conv_first(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(self.lrelu(self.HRconv(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.conv_first(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.conv_last(self.lrelu(self.HRconv(out)))
        elif self.upscale == 1:
            out = self.conv_last(out)

        
        return out

if __name__ == '__main__':
    #model_Head = RCAN_Head(in_c=180, out_c=3, scale=1, require_modulation=True, degradation_embed_dim=512).cuda()
    model_Head = MSRResNet_Head(in_c=180, out_c=3, nf=64, scale=1, require_modulation=True, degradation_embed_dim=512).cuda()

    input_x = torch.randn((2, 180, 480, 500)).cuda()
    embed = torch.randn((2, 512, 1, 1)).cuda()

    output = model_Head(input_x, embed)
    print(output.shape)

    total_params = sum(p.numel() for p in model_Head.parameters())
    print(f'{total_params:,} total parameters.')