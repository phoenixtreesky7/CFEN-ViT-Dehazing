
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


        ################################################################
        ###                            GViT                          ###
        ################################################################
        #---------------------------
        ## Encoder ##
        # Global ViT Encoder Stage One
        # img_dim=256/4=64 / patch_dim=2 / n_feats=32 / embedding_dim=4x32=128 / num_heads=2 / num_layers=1 / num_patches=(64/2)**2=32x32
        self.globalvit_encoder_01 = GViT(img_dim=int(int(opt.loadSize)/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=n_feats, embedding_dim=n_feats*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=opt.num_heads, num_layers=opt.num_layers, hidden_dim=n_feats*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Global ViT Encoder Stage Two
        # img_dim=128/4=32 / patch_dim=2 / n_feats=64 / embedding_dim=4x64=256 / num_heads=4 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.globalvit_encoder_02 = GViT(img_dim=int(int(opt.loadSize//2)/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(2*n_feats), embedding_dim=int(2*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=int(opt.num_heads*2), num_layers=opt.num_layers, hidden_dim=int(2*n_feats)*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Global ViT Encoder Stage Three
        # img_dim=64/4=16 / patch_dim=2 / n_feats=128 / embedding_dim=4x128=512 / num_heads=8 / num_layers=1 / num_patches=(16/2)**2=8x8
        self.globalvit_encoder_03 = GViT(img_dim=int(int(opt.loadSize//4)/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(4*n_feats), embedding_dim=int(4*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=int(opt.num_heads*4), num_layers=opt.num_layers, hidden_dim=int(4*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        #---------------------------
        ## Decoder for Reflectance Image ##
        # Global ViT Decoder Stage Three
        # img_dim=64/4=16 / patch_dim=4 / n_feats=128 / embedding_dim=4x128=512 / num_heads=8 / num_layers=1 / num_patches=(16/2)**2=8x8
        self.globalvit_decoder_03r = GViT(img_dim=int(int(opt.loadSize//4)/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(4*n_feats), embedding_dim=int(4*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=int(opt.num_heads*4), num_layers=opt.num_layers, hidden_dim=int(4*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Global ViT Decoder Stage Two
         # img_dim=128/4=32 / patch_dim=4 / n_feats=64 / embedding_dim=4x64=256 / num_heads=4 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.globalvit_decoder_02r = GViT(img_dim=int(int(opt.loadSize//2)/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(2*n_feats), embedding_dim=int(2*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=int(opt.num_heads*2), num_layers=opt.num_layers, hidden_dim=int(2*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Global ViT Decoder Stage One
        # img_dim=256/4=64 / patch_dim=4 / n_feats=32 / embedding_dim=4x32=128 / num_heads=2 / num_layers=1 / num_patches=(64/2)**2=32x32
        self.globalvit_decoder_01r = GViT(img_dim=int(opt.loadSize/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(n_feats), embedding_dim=n_feats*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=opt.num_heads, num_layers=opt.num_layers, hidden_dim=n_feats*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        #---------------------------
        ## Decoder for Shading Image ##
        # Global ViT Decoder Stage Three
        # img_dim=64/4=16 / patch_dim=4 / n_feats=128 / embedding_dim=4x128=512 / num_heads=8 / num_layers=1 / num_patches=(16/2)**2=8x8
        self.globalvit_decoder_03s = GViT(img_dim=int(int(opt.loadSize//4)/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(4*n_feats), embedding_dim=int(4*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=int(opt.num_heads*4), num_layers=opt.num_layers, hidden_dim=int(4*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Global ViT Decoder Stage Two
         # img_dim=128/4=32 / patch_dim=4 / n_feats=64 / embedding_dim=4x64=256 / num_heads=4 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.globalvit_decoder_02s = GViT(img_dim=int(int(opt.loadSize//2)/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(2*n_feats), embedding_dim=int(2*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=int(opt.num_heads*2), num_layers=opt.num_layers, hidden_dim=int(2*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Global ViT Decoder Stage One
        # img_dim=256/4=64 / patch_dim=4 / n_feats=32 / embedding_dim=4x32=128 / num_heads=2 / num_layers=1 / num_patches=(64/2)**2=32x32
        self.globalvit_decoder_01s = GViT(img_dim=int(opt.loadSize/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(n_feats), embedding_dim=n_feats*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=opt.num_heads, num_layers=opt.num_layers, hidden_dim=n_feats*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        #---------------------------
        ## Decoder for Dehazed Image ##
        # Global ViT Decoder Stage Three
        # img_dim=64/4=16 / patch_dim=4 / n_feats=128 / embedding_dim=4x128=512 / num_heads=8 / num_layers=1 / num_patches=(16/2)**2=8x8
        self.globalvit_decoder_03d = GViT(img_dim=int(int(opt.loadSize//4)/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(4*n_feats), embedding_dim=int(4*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=int(opt.num_heads*4), num_layers=opt.num_layers, hidden_dim=int(4*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Global ViT Decoder Stage Two
         # img_dim=128/4=32 / patch_dim=4 / n_feats=64 / embedding_dim=4x64=256 / num_heads=4 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.globalvit_decoder_02d = GViT(img_dim=int(int(opt.loadSize//2)/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(2*n_feats), embedding_dim=int(2*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=int(opt.num_heads*2), num_layers=opt.num_layers, hidden_dim=int(2*n_feats)*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Global ViT Decoder Stage One
        # img_dim=256/4=64 / patch_dim=4 / n_feats=32 / embedding_dim=4x32=128 / num_heads=2 / num_layers=1 / num_patches=(64/2)**2=32x32
        self.globalvit_decoder_01d = GViT(img_dim=int(opt.loadSize/int(opt.patch_dim*2)), patch_dim=int(opt.patch_dim*2), num_channels=int(n_feats), embedding_dim=n_feats*int(opt.patch_dim*2)*int(opt.patch_dim*2), num_heads=opt.num_heads, num_layers=opt.num_layers, hidden_dim=n_feats*int(opt.patch_dim*2)*int(opt.patch_dim*2)*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)




        ################################################################
        ###         Convolution: Down-scale and Up-scale             ###
        ################################################################

        ## Down-scale Convolution at the tail of the Encoder Block##
        ds_conv_e02 = [nn.Conv2d(n_feats, int(2*n_feats), kernel_size=3, stride=2, padding=1), nn.InstanceNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.ds_conv_e02 = nn.Sequential(*ds_conv_e02)
        ds_conv_e03 = [nn.Conv2d(int(2*n_feats), int(4*n_feats), kernel_size=3, stride=2, padding=1), nn.InstanceNorm2d(int(4*n_feats)), nn.ReLU(True)]
        self.ds_conv_e03 = nn.Sequential(*ds_conv_e03)

        ## Up-scale Convolution at the head of the Deocder Block (Reflectance) ##
        us_conv_d03r = [nn.ConvTranspose2d(int(4*n_feats), int(2*n_feats), kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.us_conv_d03r = nn.Sequential(*us_conv_d03r)
        us_conv_d02r = [nn.ConvTranspose2d(int(2*n_feats), n_feats, kernel_size=4, stride=2, padding=1), ActNorm2d(n_feats), nn.ReLU(True)]
        self.us_conv_d02r = nn.Sequential(*us_conv_d02r)

        ## Up-scale Convolution at the head of the Deocder Block (Shading) ##
        us_conv_d03s = [nn.ConvTranspose2d(int(4*n_feats), int(2*n_feats), kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.us_conv_d03s = nn.Sequential(*us_conv_d03s)
        us_conv_d02s = [nn.ConvTranspose2d(int(2*n_feats), n_feats, kernel_size=4, stride=2, padding=1), ActNorm2d(n_feats), nn.ReLU(True)]
        self.us_conv_d02s = nn.Sequential(*us_conv_d02s)

        ## Up-scale Convolution at the head of the Deocder Block (Dehazed) ##
        us_conv_d03d = [nn.ConvTranspose2d(int(4*n_feats), int(2*n_feats), kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.us_conv_d03d = nn.Sequential(*us_conv_d03d)
        us_conv_d02d = [nn.ConvTranspose2d(int(2*n_feats), n_feats, kernel_size=4, stride=2, padding=1), ActNorm2d(n_feats), nn.ReLU(True)]
        self.us_conv_d02d = nn.Sequential(*us_conv_d02d)

        #####################################################################
        ###          Convolution: Skip-conection Concatnation             ###
        #####################################################################

        ## Skip-conection Concatnation Convolution (Reflectance) ##
        sk_conv_d03r = [nn.ConvTranspose2d(int(4*n_feats), int(2*n_feats), kernel_size=1, stride=1, padding=0), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.sk_conv_d03r = nn.Sequential(*sk_conv_d03r)
        sk_conv_d02r = [nn.ConvTranspose2d(int(2*n_feats), n_feats, kernel_size=1, stride=1, padding=0), ActNorm2d(n_feats), nn.ReLU(True)]
        self.sk_conv_d02r = nn.Sequential(*sk_conv_d02r)

        ## Skip-conection Concatnation Convolution (Reflectance) ##
        sk_conv_d03s = [nn.ConvTranspose2d(int(4*n_feats), int(2*n_feats), kernel_size=1, stride=1, padding=0), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.sk_conv_d03s = nn.Sequential(*sk_conv_d03s)
        sk_conv_d02s = [nn.ConvTranspose2d(int(2*n_feats), n_feats, kernel_size=1, stride=1, padding=0), ActNorm2d(n_feats), nn.ReLU(True)]
        self.sk_conv_d02s = nn.Sequential(*sk_conv_d02s)

        ## Skip-conection Concatnation Convolution (Reflectance) ##
        sk_conv_d03d = [nn.ConvTranspose2d(int(6*n_feats), int(2*n_feats), kernel_size=1, stride=1, padding=0), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.sk_conv_d03d = nn.Sequential(*sk_conv_d03d)
        sk_conv_d02d = [nn.ConvTranspose2d(int(3*n_feats), n_feats, kernel_size=1, stride=1, padding=0), ActNorm2d(n_feats), nn.ReLU(True)]
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
                ActNorm2d(n_feats), 
                nn.ReLU(True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(n_feats, 1, kernel_size=7, padding=0),
                nn.Tanh()
                #conv(n_feats, opt.n_colors, kernel_size)
            )
        ])

        

        #self.crop2x2 = Crop2x2()
        #self.join2x2 = Join2x2()

        #self.sp = SpatialPyramid()

    def forward(self, input):
        ### Feature Extraction ###
        #print('x', x.shape)
        xf = self.head[self.scale_idx](input)
        ###########################################################
        ##                     Shared Encoder                    ##
        ###########################################################
        ### Encoder Level One ###
        ## Local ViT ##
        # Origional feature maps are cropped to 2x2 regions
        

        ## Global ViT
        x_gve_01 = self.globalvit_encoder_01(xf)

        # Merge the Local ViT and Global ViT, and downscaling
        x_e_01 = x_gve_01 + xf

        ### Encoder Level Two ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        x_e_01_ds = self.ds_conv_e02(x_e_01)
        

        ## Global ViT
        x_gve_02 = self.globalvit_encoder_02(x_e_01_ds)

        # Merge the Local ViT and Global ViT, and downscaling
        x_e_02 = x_gve_02 + x_e_01_ds

        ### Encoder Level Three ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        x_e_02_ds = self.ds_conv_e03(x_e_02)
        

        ## Global ViT
        x_gve_03 = self.globalvit_encoder_03(x_e_02_ds)

        # Merge the Local ViT and Global ViT, and upscaling
        x_e_03 = x_gve_03 + x_e_02_ds
        


        ############################################################
        ##                Reflectance Image Decoder               ##
        ############################################################
        ### Dcoder Level Three ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        r_d_03_in = x_e_03
        

        ## Global ViT

        r_gvd_03 = self.globalvit_decoder_03r(r_d_03_in)

        # Merge the Local ViT and Global ViT, and upscaling
        #print('r_lvd_03', r_lvd_03.shape, 'r_gvd_03', r_gvd_03.shape, 'r_d_03_in', r_d_03_in.shape,)
        r_d_03 = r_gvd_03 + r_d_03_in
        

        ### Decoder Level Two ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        r_d_03_us = self.us_conv_d03r(r_d_03)
        r_d_02_in = self.sk_conv_d03r(torch.cat((r_d_03_us, x_e_02),1))
        

        ## Global ViT
        r_gvd_02 = self.globalvit_decoder_02r(r_d_02_in)

        # Merge the Local ViT and Global ViT, and upscaling
        r_d_02 = r_gvd_02 + r_d_02_in



        ### Decoder Level One ###
        ## Local ViT ##
        # Origional feature maps are cropped to 2x2 regions
        r_d_02_us = self.us_conv_d02r(r_d_02)
        r_d_01_in = self.sk_conv_d02r(torch.cat((r_d_02_us, x_e_01),1))
        

        ## Global ViT
        r_gvd_01 = self.globalvit_decoder_01r(r_d_01_in)

        # Merge the Local ViT and Global ViT, and downscaling
        r_d_01 = r_gvd_01 + r_d_01_in
        

        xr = self.tail_color[self.scale_idx](r_d_01+xf)
        #print('tail(x)', x.shape)
        #x = self.add_mean(x)


        ############################################################
        ##                Shading Image Decoder               ##
        ############################################################
        ### Dcoder Level Three ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        s_d_03_in = x_e_03
        
        ## Global ViT

        s_gvd_03 = self.globalvit_decoder_03s(s_d_03_in)

        # Merge the Local ViT and Global ViT, and upscaling
        s_d_03 = s_gvd_03 + s_d_03_in
        
        ### Decoder Level Two ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        s_d_03_us = self.us_conv_d03s(s_d_03) 
        s_d_02_in = self.sk_conv_d03s(torch.cat((s_d_03_us, x_e_02),1))
        

        ## Global ViT
        s_gvd_02 = self.globalvit_decoder_02s(s_d_02_in)

        # Merge the Local ViT and Global ViT, and upscaling
        s_d_02 = s_gvd_02 + s_d_02_in


        ### Encoder Level One ###
        ## Local ViT ##
        # Origional feature maps are cropped to 2x2 regions
        s_d_02_us = self.us_conv_d02s(s_d_02) 
        s_d_01_in = self.sk_conv_d02s(torch.cat((s_d_02_us, x_e_01),1))
        

        ## Global ViT
        s_gvd_01 = self.globalvit_decoder_01s(s_d_01_in)

        # Merge the Local ViT and Global ViT, and downscaling
        s_d_01 = s_gvd_01 + s_d_01_in

        xs = self.tail_gray[self.scale_idx](s_d_01+xf)
        #print('tail(x)', x.shape)
        #x = self.add_mean(x)


        ############################################################
        ##                Dehazed Image Decoder               ##
        ############################################################
        ### Dcoder Level Three ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        d_d_03_in = x_e_03
        
        ## Global ViT

        d_gvd_03 = self.globalvit_decoder_03d(d_d_03_in)

        # Merge the Local ViT and Global ViT, and upscaling
        d_d_03 = d_gvd_03 + d_d_03_in
        
        ### Decoder Level Two ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        d_d_03_us = self.us_conv_d03d(d_d_03) 
        d_d_02_in = self.sk_conv_d03d(torch.cat((d_d_03_us, r_d_03_us, s_d_03_us),1))
        
        ## Global ViT
        d_gvd_02 = self.globalvit_decoder_02d(d_d_02_in)

        # Merge the Local ViT and Global ViT, and upscaling
        d_d_02 = d_gvd_02 + d_d_02_in


        ### Encoder Level One ###
        ## Local ViT ##
        # Origional feature maps are cropped to 2x2 regions
        d_d_02_us = self.us_conv_d02d(s_d_02) 
        d_d_01_in = self.sk_conv_d02d(torch.cat((d_d_02_us, r_d_02_us, s_d_02_us),1))
        

        ## Global ViT
        d_gvd_01 = self.globalvit_decoder_01d(d_d_01_in)

        # Merge the Local ViT and Global ViT, and downscaling
        d_d_01 = d_gvd_01 + d_d_01_in

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

class Crop2x2(nn.Module):
    def __init__(self):
        super(Crop2x2, self).__init__()

    def forward(self, x):
        batch_size,C,H,W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        local_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]
        
        return [local_feat, feat_sub_lu, feat_sub_ld, feat_sub_ru, feat_sub_rd]

class Join2x2(nn.Module):
    def __init__(self):
        super(Join2x2, self).__init__()

    def forward(self, x, lu, ld, ru, rd):
        batch_size,C,H1,W1 = lu.shape
        #print('lu batch_size,C,H1,W1', batch_size,C,H1,W1)
        #batch_size,C,H2,W2 = ld.shape
        #print('lu batch_size,C,H2,W2', batch_size,C,H2,W2)
        x[:, :, :H1, :W1] = lu
        x[:, :, H1:, :W1] = ld
        x[:, :, :H1, W1:] = ru
        x[:, :, H1:, W1:] = rd
        
        return x
################################                ||   \\        //||========
################################                ||    \\      //      ||
###      Local ViT Block      ##                ||     \\    //  ||   ||
################################                ||      \\  //   ||   ||
################################                ||====== \\//    ||   ||
class LViT(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        num_queries,
        positional_encoding_type="learned",
        dropout_rate=0,
        no_norm=False,
        mlp=False,
        pos_every=False,
        no_pos = False
    ):
        super(LViT, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        
        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels
        
        self.out_dim = patch_dim * patch_dim * num_channels
        
        self.no_pos = no_pos
        
        if self.mlp==False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )
        
            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        
        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                )
            
        self.dropout_layer1 = nn.Dropout(dropout_rate)
        
        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std = 1/m.weight.size(1))

    def forward(self, x, con=False):
        #print('img_dim', self.img_dim)
        #print('x', x.shape)
        x = torch.nn.functional.unfold(x,self.patch_dim,stride=self.patch_dim).transpose(1,2).transpose(0,1).contiguous()
        #print('unfold(x)', x.shape, 'mlp', self.mlp)
        if self.mlp==False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x

            #query_embed = self.query_embed.weight[query_idx].view(-1,1,self.embedding_dim).repeat(1,x.size(1), 1)
            #print('query_embed.shape', query_embed.shape) # 64 x 1 x5 12
        else:
            query_embed = None

        
        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0,1)
            #print('pos', pos.shape)

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            #print('pos_every-encoder(x)', x.shape)
            #x = self.decoder(x, x, pos=pos, query_pos=query_embed)
            #print('pos_every-decoder(x)', x.shape)
        elif self.no_pos:
            x = self.encoder(x)
            #print('no_pos-encoder(x)', x.shape)
            #x = self.decoder(x, x, query_pos=query_embed)
            #print('no_pos-decoder(x)', x.shape)
        else:
            x = self.encoder(x+pos)
            #print('pos-encoder(x)', x.shape)
            #x = self.decoder(x, x, query_pos=query_embed)
            #print('pos-decoder(x)', x.shape)
        
        
        if self.mlp==False:
            x = self.mlp_head(x) + x
            #print('mlp_head(x)', x.shape)
        
        x = x.transpose(0,1).contiguous().view(x.size(1), -1, self.flatten_dim)
        #print('transpose(x)', x.shape)
        
        if con:
            con_x = x
            #print('con_x', con_x.shape)
            x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
            #print('x', x.shape)
            return x, con_x
        
        x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
        #print('final output x', x.shape)
        
        return x


#################################             ========\\        //||========
#################################             ||       \\      //      ||
###      Global ViT Block      ##             ||   ==== \\    //  ||   ||
#################################             ||    ||   \\  //   ||   ||
#################################             ||=======   \\//    ||   ||
class GViT(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        num_queries,
        positional_encoding_type="learned",
        dropout_rate=0,
        no_norm=False,
        mlp=False,
        pos_every=False,
        no_pos = False
    ):
        super(GViT, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        
        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels
        
        self.out_dim = patch_dim * patch_dim * num_channels
        
        self.no_pos = no_pos

        ave_pool2 = [nn.AvgPool2d(2, stride=2)]
        self.ave_pool2 = nn.Sequential(*ave_pool2)

        if self.mlp==False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )
        
            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        
        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                )
            
        self.dropout_layer1 = nn.Dropout(dropout_rate)
        
        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std = 1/m.weight.size(1))
        upsam = [nn.Upsample(scale_factor=2, mode='bilinear')]
        self.upsam = nn.Sequential(*upsam)

    def forward(self, x, con=False):
        #print('global x', x.shape)
        #x = self.ave_pool2(self.ave_pool2(x))
        x = torch.nn.functional.unfold(x,self.patch_dim,stride=self.patch_dim).transpose(1,2).transpose(0,1).contiguous()
        #print('unfold(x)', x.shape, 'mlp', self.mlp)
        if self.mlp==False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x

            #query_embed = self.query_embed.weight[query_idx].view(-1,1,self.embedding_dim).repeat(1,x.size(1), 1)
            #print('query_embed.shape', query_embed.shape) # 64 x 1 x5 12
        else:
            query_embed = None

        
        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0,1)
            #print('pos', pos.shape)

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            #print('pos_every-encoder(x)', x.shape)
            #x = self.decoder(x, x, pos=pos, query_pos=query_embed)
            #print('pos_every-decoder(x)', x.shape)
        elif self.no_pos:
            x = self.encoder(x)
            #print('no_pos-encoder(x)', x.shape)
            #x = self.decoder(x, x, query_pos=query_embed)
            #print('no_pos-decoder(x)', x.shape)
        else:
            x = self.encoder(x+pos)
            #print('pos-encoder(x)', x.shape)
            #x = self.decoder(x, x, query_pos=query_embed)
            #print('pos-decoder(x)', x.shape)
        
        
        if self.mlp==False:
            x = self.mlp_head(x) + x
            #print('mlp_head(x)', x.shape)
        
        x = x.transpose(0,1).contiguous().view(x.size(1), -1, self.flatten_dim)
        #print('transpose(x)', x.shape)
        
        if con:
            con_x = x
            #print('con_x', con_x.shape)
            x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
            #print('x', x.shape)
            return x, con_x
        
        x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
        #print('final output x', x.shape)
        x = self.upsam(self.upsam(x))
        
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings
    
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src, pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos = None, query_pos = None):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output

    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos = None, query_pos = None):
        tgt2 = self.norm1(tgt)
        #print('tgt2', tgt2.shape, 'query_pos', query_pos.shape)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


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

        self.batch20 = ActNorm2d(20, affine=True)
        self.batch1 = ActNorm2d(1, affine=True)

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