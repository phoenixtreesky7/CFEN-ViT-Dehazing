
# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from models import common

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import copy
from torch.nn import init
from torch.optim import lr_scheduler
from models.actnorm import ActNorm2d
def make_model(opt, parent=False):
    return ipt(opt)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(ActNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
        
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def define_G(opt, conv):
    netG = None
    #norm_layer = get_norm_layer(norm_type=norm)
    
    netG = dec_ipt(opt, conv)
    
    #    raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, opt.init_type, opt.gpu_ids)


class dec_ipt(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(dec_ipt, self).__init__()
        
        self.scale_idx = 0
        
        self.opt = opt
        
        n_feats = opt.n_feats
        kernel_size = 5 
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(opt.rgb_range)
        self.add_mean = common.MeanShift(opt.rgb_range, sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(opt.n_colors, n_feats, kernel_size),
                common.ResBlock(conv, n_feats, 3, act=act))
        ])


        #---------------------------
        ## Encoder ##
        
        n_blocks = 2
        mult = 1
        encoder_01 = []
        for i in range(n_blocks):
            encoder_01 += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.encoder_01 = nn.Sequential(*encoder_01)

        mult = 2
        encoder_02 = []
        for i in range(n_blocks):
            encoder_02 += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.encoder_02 = nn.Sequential(*encoder_02)

        mult = 4
        encoder_03 = []
        for i in range(n_blocks):
            encoder_03 += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.encoder_03 = nn.Sequential(*encoder_03)


        ##  Decoder : Reflectance  ##
        decoder_01r = []
        for i in range(n_blocks):
            decoder_01r += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.decoder_01r = nn.Sequential(*decoder_01r)

        mult = 2
        decoder_02r = []
        for i in range(n_blocks):
            decoder_02r += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.decoder_02r = nn.Sequential(*decoder_02r)

        mult = 4
        decoder_03r = []
        for i in range(n_blocks):
            decoder_03r += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.decoder_03r = nn.Sequential(*decoder_03r)

        ##  Decoder : Shading  ##
        decoder_01r = []
        for i in range(n_blocks):
            decoder_01r += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.decoder_01r = nn.Sequential(*decoder_01r)

        mult = 2
        decoder_02r = []
        for i in range(n_blocks):
            decoder_02r += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.decoder_02r = nn.Sequential(*decoder_02r)

        mult = 4
        decoder_03r = []
        for i in range(n_blocks):
            decoder_03r += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.decoder_03r = nn.Sequential(*decoder_03r)

        ##  Decoder : Dehazing  ##
        decoder_01d = []
        for i in range(n_blocks):
            decoder_01d += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.decoder_01d = nn.Sequential(*decoder_01d)

        mult = 2
        decoder_02d = []
        for i in range(n_blocks):
            decoder_02d += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.decoder_02d = nn.Sequential(*decoder_02d)

        mult = 4
        decoder_03d = []
        for i in range(n_blocks):
            decoder_03d += [ResnetBlock(n_feats * mult, padding_type=padding_type, num_layers=opt.num_layers)]

        self.decoder_03d = nn.Sequential(*decoder_03d)


        ################################################################
        ###         Convolution: Down-scale and Up-scale             ###
        ################################################################

        ## Down-scale Convolution at the tail of the Encoder Block##
        ds_conv_e02 = [nn.Conv2d(n_feats, int(2*n_feats), kernel_size=3, stride=2, padding=1), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.ds_conv_e02 = nn.Sequential(*ds_conv_e02)
        ds_conv_e03 = [nn.Conv2d(int(2*n_feats), int(4*n_feats), kernel_size=3, stride=2, padding=1), ActNorm2d(int(4*n_feats)), nn.ReLU(True)]
        self.ds_conv_e03 = nn.Sequential(*ds_conv_e03)

        ## Up-scale Convolution at the head of the Deocder Block (Reflectance) ##
        us_conv_d03r = [nn.ConvTranspose2d(int(4*n_feats), int(2*n_feats), kernel_size=4, stride=2, padding=1), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.us_conv_d03r = nn.Sequential(*us_conv_d03r)
        us_conv_d02r = [nn.ConvTranspose2d(int(2*n_feats), int(n_feats), kernel_size=4, stride=2, padding=1), ActNorm2d(int(n_feats)), nn.ReLU(True)]
        self.us_conv_d02r = nn.Sequential(*us_conv_d02r)

        ## Up-scale Convolution at the head of the Deocder Block (Shading) ##
        us_conv_d03s = [nn.ConvTranspose2d(int(4*n_feats), int(2*n_feats), kernel_size=4, stride=2, padding=1), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.us_conv_d03s = nn.Sequential(*us_conv_d03s)
        us_conv_d02s = [nn.ConvTranspose2d(int(2*n_feats), int(n_feats), kernel_size=4, stride=2, padding=1), ActNorm2d(int(n_feats)), nn.ReLU(True)]
        self.us_conv_d02s = nn.Sequential(*us_conv_d02s)

        ## Up-scale Convolution at the head of the Deocder Block (Dehazed) ##
        us_conv_d03d = [nn.ConvTranspose2d(int(4*n_feats), int(2*n_feats), kernel_size=4, stride=2, padding=1), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.us_conv_d03d = nn.Sequential(*us_conv_d03d)
        us_conv_d02d = [nn.ConvTranspose2d(int(2*n_feats), int(n_feats), kernel_size=4, stride=2, padding=1), ActNorm2d(int(n_feats)), nn.ReLU(True)]
        self.us_conv_d02d = nn.Sequential(*us_conv_d02d)

        #####################################################################
        ###          Convolution: Skip-conection Concatnation             ###
        #####################################################################

        ## Skip-conection Concatnation Convolution (Reflectance) ##
        sk_conv_d03r = [nn.Conv2d(int(4*n_feats), int(2*n_feats), kernel_size=1, stride=1, padding=0), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.sk_conv_d03r = nn.Sequential(*sk_conv_d03r)
        sk_conv_d02r = [nn.Conv2d(int(2*n_feats), n_feats, kernel_size=1, stride=1, padding=0), ActNorm2d(n_feats), nn.ReLU(True)]
        self.sk_conv_d02r = nn.Sequential(*sk_conv_d02r)

        ## Skip-conection Concatnation Convolution (shading) ##
        sk_conv_d03s = [nn.Conv2d(int(4*n_feats), int(2*n_feats), kernel_size=1, stride=1, padding=0), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.sk_conv_d03s = nn.Sequential(*sk_conv_d03s)
        sk_conv_d02s = [nn.Conv2d(int(2*n_feats), n_feats, kernel_size=1, stride=1, padding=0), ActNorm2d(n_feats), nn.ReLU(True)]
        self.sk_conv_d02s = nn.Sequential(*sk_conv_d02s)

        ## Skip-conection Concatnation Convolution (dehazed) ##
        sk_conv_d03d = [nn.Conv2d(int(6*n_feats), int(2*n_feats), kernel_size=1, stride=1, padding=0), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.sk_conv_d03d = nn.Sequential(*sk_conv_d03d)
        sk_conv_d02d = [nn.Conv2d(int(3*n_feats), n_feats, kernel_size=1, stride=1, padding=0), ActNorm2d(n_feats), nn.ReLU(True)]
        self.sk_conv_d02d = nn.Sequential(*sk_conv_d02d)



        self.tail_color = nn.ModuleList([
            nn.Sequential(
                common.Upsampler(conv, n_feats, act=False),
                nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1), 
                ActNorm2d(n_feats), 
                nn.ReLU(True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(n_feats, opt.n_colors, kernel_size=7, padding=0),
                nn.Tanh()
                #conv(n_feats, opt.n_colors, kernel_size)
            )
        ])
        self.tail_gray = nn.ModuleList([
            nn.Sequential(
                common.Upsampler(conv, n_feats, act=False),
                nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1), 
                #ActNorm2d(n_feats), 
                nn.ReLU(True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(n_feats, 1, kernel_size=7, padding=0),
                nn.Tanh()
                #conv(n_feats, opt.n_colors, kernel_size)
            )
        ])

        

        self.crop2x2 = Crop2x2()
        self.join2x2 = Join2x2()

        self.sp = SpatialPyramid()

    def forward(self, input):
        ### Feature Extraction ###
        #print('x', x.shape)
        xf = self.head[self.scale_idx](input)
        ###########################################################
        ##                     Shared Encoder                    ##
        ###########################################################
        ### Encoder Level One ###
        x_e_01 = self.encoder_01(xf) + xf
        x_e_01_ds = self.ds_conv_e02(x_e_01)

        x_e_02 = self.encoder_02(x_e_01_ds) + x_e_01_ds
        x_e_02_ds = self.ds_conv_e03(x_e_02)

        x_e_03 = self.encoder_03(x_e_02_ds) + x_e_02_ds
        


        ############################################################
        ##                Reflectance Image Decoder               ##
        ############################################################
        ### Dcoder Level Three ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        r_d_03_in = x_e_03
        r_d_03 = self.decoder_03r(r_d_03_in) + r_d_03_in
        r_d_03_us = self.us_conv_d03r(r_d_03)

        r_d_02_in = self.sk_conv_d03r(torch.cat((r_d_03_us, x_e_02),1))
        r_d_02 = self.decoder_02r(r_d_02_in) + r_d_02_in
        r_d_02_us = self.us_conv_d02r(r_d_02)

        r_d_01_in = self.sk_conv_d02r(torch.cat((r_d_02_us, x_e_01),1))
        r_d_01 = self.decoder_01r(r_d_01_in) + r_d_01_in
        

        xr = self.tail_color[self.scale_idx](r_d_01+xf)
        #print('tail(x)', x.shape)
        #x = self.add_mean(x)


        ############################################################
        ##                Shading Image Decoder               ##
        ############################################################
        ### Dcoder Level Three ###
        
        s_d_03_in = x_e_03
        s_d_03 = self.decoder_03s(s_d_03_in) + s_d_03_in
        s_d_03_us = self.us_conv_d03s(s_d_03)

        s_d_02_in = self.sk_conv_d03s(torch.cat((s_d_03_us, x_e_02),1))
        s_d_02 = self.decoder_02s(s_d_02_in) + s_d_02_in
        s_d_02_us = self.us_conv_d02s(s_d_02)

        s_d_01_in = self.sk_conv_d02s(torch.cat((s_d_02_us, x_e_01),1))
        s_d_01 = self.decoder_01s(s_d_01_in) + s_d_01_in
        

        xs = self.tail_gray[self.scale_idx](s_d_01+xf)
        #print('tail(x)', x.shape)
        #x = self.add_mean(x)


        ############################################################
        ##                Dehazed Image Decoder               ##
        ############################################################
        ### Dcoder Level Three ###
        
        d_d_03_in = x_e_03
        d_d_03 = self.decoder_03d(d_d_03_in) + d_d_03_in
        d_d_03_us = self.us_conv_d03d(d_d_03)

        d_d_02_in = self.sk_conv_d03d(torch.cat((d_d_03_us, r_d_03_us, s_d_03_us),1))
        d_d_02 = self.decoder_02d(d_d_02_in) + d_d_02_in
        d_d_02_us = self.us_conv_d02d(s_d_02)

        d_d_01_in = self.sk_conv_d02d(torch.cat((d_d_02_us, r_d_02_us, s_d_02_us),1))
        d_d_01 = self.decoder_01d(d_d_01_in) + d_d_01_in
        

        xd = self.tail_color[self.scale_idx](d_d_01+xf)

        
        #print('tail(x)', x.shape)
        #x = self.add_mean(x)

        #xdh_temp = torch.zeros_like(input)
        #xdh_temp[:,0,:,:] = xr[:,0,:,:]*xs
        #xdh_temp[:,1,:,:] = xr[:,1,:,:]*xs
        #xdh_temp[:,2,:,:] = xr[:,2,:,:]*xs
        #xdh = self.sp(torch.cat((input, xr, xs, xd),1))

        return [xr, xs, xd] 

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


################################
################################
###        ResNet Block       ##
################################
################################
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer)
    
    def build_conv_block(self, dim, padding_type, norm_layer):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]
        
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class basic_Conv2d(nn.Module):
    def __init__(self, input_channle, output_channle, kernel_size=3, stride=1, padding=1, norm="instance",
                 activation="relu"):
        super(basic_Conv2d).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.norm=nn.ActNorm2d(output_channle, affine=True)

        self.bc = nn.Conv2d(input_channle, output_channle, kernel_size, stride, padding)
        
    def forward(self, x):
        x = self.relu(self.norm(self.bc(x)))

        return x


class SpatialPyramid(nn.Module):
    def __init__(self):
        super(SpatialPyramid, self).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.tanh=nn.Tanh()

        self.refine1= nn.Conv2d(10, 32, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(32, 16, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(32, 16, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(32, 16, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(32, 16, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1050 = nn.Conv2d(32, 16, kernel_size=1,stride=1,padding=0)  # 1mm

        #refine3 = [nn.ReflectionPad2d(3)]
        refine3 = [nn.Conv2d(32+5*16, 3, kernel_size=3, stride=1, padding=1)]
        refine3 += [nn.Tanh()]
        self.refine3 = nn.Sequential(*refine3)
        self.upsample = F.upsample_bilinear

        self.batch20 = ActNorm2d(20)
        self.batch1 = ActNorm2d(1)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)

        x105 = F.avg_pool2d(dehaze, 2)

        x1010 = self.upsample(self.relu((self.conv1010(x101))),size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))),size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))),size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))),size=shape_out)
        x1050 = self.upsample(self.relu((self.conv1050(x105))),size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x1050, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))

        return dehaze