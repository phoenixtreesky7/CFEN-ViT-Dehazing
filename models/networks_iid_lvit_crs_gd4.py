
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
        norm_layer = functools.partial(ActNorm2d)
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
        ###                            LViT                          ###
        ################################################################
        #---------------------------
        ## Encoder ##
        # Local ViT Encoder Stage One
        # img_dim=32 / patch_dim=2 / n_feats=32 / embedding_dim=4x32=128 / num_heads=4 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_encoder_01 = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=n_feats, embedding_dim=n_feats*opt.patch_dim*opt.patch_dim, num_heads=opt.num_heads, num_layers=opt.num_layers, hidden_dim=n_feats*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Local ViT Encoder Stage Two
        # img_dim=32 / patch_dim=2 / n_feats=64 / embedding_dim=4x64=256 / num_heads=8 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_encoder_02 = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=int(2*n_feats), embedding_dim=int(2*n_feats)*opt.patch_dim*opt.patch_dim, num_heads=int(opt.num_heads*2), num_layers=opt.num_layers, hidden_dim=int(2*n_feats)*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Local ViT Encoder Stage Three
        # img_dim=32 / patch_dim=2 / n_feats=128 / embedding_dim=4x128=512 / num_heads=16 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_encoder_03 = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=int(4*n_feats), embedding_dim=int(4*n_feats)*opt.patch_dim*opt.patch_dim, num_heads=int(opt.num_heads*4), num_layers=opt.num_layers, hidden_dim=int(4*n_feats)*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        #---------------------------
        ## Decoder for Reflectance Image ##
        # Local ViT Decoder Stage Three
        # img_dim=32 / patch_dim=2 / n_feats=128*2 / embedding_dim=4x128=512 / num_heads=16 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_decoder_03r = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=int(4*n_feats), embedding_dim=int(4*n_feats)*opt.patch_dim*opt.patch_dim, num_heads=int(opt.num_heads*4), num_layers=opt.num_layers, hidden_dim=int(4*n_feats)*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Local ViT Decoder Stage Two
        # img_dim=32 / patch_dim=2 / n_feats=64*2 / embedding_dim=4x64=256 / num_heads=16 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_decoder_02r = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=int(2*n_feats), embedding_dim=int(2*n_feats)*opt.patch_dim*opt.patch_dim, num_heads=int(opt.num_heads*2), num_layers=opt.num_layers, hidden_dim=int(2*n_feats)*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Local ViT Decoder Stage One
        # img_dim=32 / patch_dim=2 / n_feats=128*2 / embedding_dim=4x64=256 / num_heads=16 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_decoder_01r = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=n_feats, embedding_dim=n_feats*opt.patch_dim*opt.patch_dim, num_heads=opt.num_heads, num_layers=opt.num_layers, hidden_dim=n_feats*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        #---------------------------
        ## Decoder for Shading Image ##
        # Local ViT Decoder Stage Three
        # img_dim=32 / patch_dim=2 / n_feats=128*2 / embedding_dim=4x128=512 / num_heads=16 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_decoder_03s = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=int(4*n_feats), embedding_dim=int(4*n_feats)*opt.patch_dim*opt.patch_dim, num_heads=int(opt.num_heads*4), num_layers=opt.num_layers, hidden_dim=int(4*n_feats)*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Local ViT Decoder Stage Two
        # img_dim=32 / patch_dim=2 / n_feats=64*2 / embedding_dim=4x64=256 / num_heads=16 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_decoder_02s = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=int(2*n_feats), embedding_dim=int(2*n_feats)*opt.patch_dim*opt.patch_dim, num_heads=int(opt.num_heads*2), num_layers=opt.num_layers, hidden_dim=int(2*n_feats)*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Local ViT Decoder Stage One
        # img_dim=32 / patch_dim=2 / n_feats=128*2 / embedding_dim=4x64=256 / num_heads=16 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_decoder_01s = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=n_feats, embedding_dim=n_feats*opt.patch_dim*opt.patch_dim, num_heads=opt.num_heads, num_layers=opt.num_layers, hidden_dim=n_feats*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        #---------------------------
        ## Decoder for Dehazed Image ##
        # Local ViT Decoder Stage Three
        # img_dim=32 / patch_dim=2 / n_feats=128*2 / embedding_dim=4x128=512 / num_heads=16 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_decoder_03d = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=int(4*n_feats), embedding_dim=int(4*n_feats)*opt.patch_dim*opt.patch_dim, num_heads=int(opt.num_heads*4), num_layers=opt.num_layers, hidden_dim=int(4*n_feats)*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Local ViT Decoder Stage Two
        # img_dim=32 / patch_dim=2 / n_feats=64*2 / embedding_dim=4x64=256 / num_heads=16 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_decoder_02d = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=int(2*n_feats), embedding_dim=int(2*n_feats)*opt.patch_dim*opt.patch_dim, num_heads=int(opt.num_heads*2), num_layers=opt.num_layers, hidden_dim=int(2*n_feats)*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)

        # Local ViT Decoder Stage One
        # img_dim=32 / patch_dim=2 / n_feats=128*2 / embedding_dim=4x64=256 / num_heads=16 / num_layers=1 / num_patches=(32/2)**2=16x16
        self.localvit_decoder_01d = LViT(img_dim=opt.patch_size, patch_dim=opt.patch_dim, num_channels=n_feats, embedding_dim=n_feats*opt.patch_dim*opt.patch_dim, num_heads=opt.num_heads, num_layers=opt.num_layers, hidden_dim=n_feats*opt.patch_dim*opt.patch_dim*opt.hidden_dim_ratio, num_queries = opt.num_queries, dropout_rate=opt.dropout_rate, mlp=opt.no_mlp ,pos_every=opt.pos_every,no_pos=opt.no_pos,no_norm=opt.no_norm)


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

        ## Skip-conection Concatnation Convolution (shading) ##
        sk_conv_d03s = [nn.ConvTranspose2d(int(4*n_feats), int(2*n_feats), kernel_size=1, stride=1, padding=0), ActNorm2d(int(2*n_feats)), nn.ReLU(True)]
        self.sk_conv_d03s = nn.Sequential(*sk_conv_d03s)
        sk_conv_d02s = [nn.ConvTranspose2d(int(2*n_feats), n_feats, kernel_size=1, stride=1, padding=0), ActNorm2d(n_feats), nn.ReLU(True)]
        self.sk_conv_d02s = nn.Sequential(*sk_conv_d02s)

        ## Skip-conection Concatnation Convolution (dehazed) ##
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
        ## Local ViT ##
        # Origional feature maps are cropped to 2x2 regions
        [x_lve_01, lu, ld, ru, rd] = self.crop2x2(xf)

        # Origional feature mape cropped to 4x4 regions
        [xlu_lv, lulu, luld, luru, lurd] = self.crop2x2(lu)
        [xld_lv, ldlu, ldld, ldru, ldrd] = self.crop2x2(ld)
        [xru_lv, rulu, ruld, ruru, rurd] = self.crop2x2(ru)
        [xrd_lv, rdlu, rdld, rdru, rdrd] = self.crop2x2(rd)

        # Origional feature maps are cropped to 8x8 regions
        [xlulu_lv, lululu, lululd, luluru, lulurd] = self.crop2x2(lulu)
        [xluld_lv, luldlu, luldld, luldru, luldrd] = self.crop2x2(luld)
        [xluru_lv, lurulu, luruld, lururu, lururd] = self.crop2x2(luru)
        [xlurd_lv, lurdlu, lurdld, lurdru, lurdrd] = self.crop2x2(lurd)
        

        [xldlu_lv, ldlulu, ldluld, ldluru, ldlurd] = self.crop2x2(ldlu)
        [xldld_lv, ldldlu, ldldld, ldldru, ldldrd] = self.crop2x2(ldld)
        [xldru_lv, ldrulu, ldruld, ldruru, ldrurd] = self.crop2x2(ldru)
        [xldrd_lv, ldrdlu, ldrdld, ldrdru, ldrdrd] = self.crop2x2(ldrd)


        [xrulu_lv, rululu, rululd, ruluru, rulurd] = self.crop2x2(rulu)
        [xruld_lv, ruldlu, ruldld, ruldru, ruldrd] = self.crop2x2(ruld)
        [xruru_lv, rurulu, ruruld, rururu, rururd] = self.crop2x2(ruru)
        [xrurd_lv, rurdlu, rurdld, rurdru, rurdrd] = self.crop2x2(rurd)
        

        [xrdlu_lv, rdlulu, rdluld, rdluru, rdlurd] = self.crop2x2(rdlu)
        [xrdld_lv, rdldlu, rdldld, rdldru, rdldrd] = self.crop2x2(rdld)
        [xrdru_lv, rdrulu, rdruld, rdruru, rdrurd] = self.crop2x2(rdru)
        [xrdrd_lv, rdrdlu, rdrdld, rdrdru, rdrdrd] = self.crop2x2(rdrd)


        # Local ViT for each (total 64) regions
        lululu, lululd, luluru, lulurd = self.localvit_encoder_01(lululu),self.localvit_encoder_01(lululd),self.localvit_encoder_01(luluru),self.localvit_encoder_01(lulurd)
        luldlu, luldld, luldru, luldrd = self.localvit_encoder_01(luldlu),self.localvit_encoder_01(luldld),self.localvit_encoder_01(luldru),self.localvit_encoder_01(luldrd)
        lurulu, luruld, lururu, lururd = self.localvit_encoder_01(lurulu),self.localvit_encoder_01(luruld),self.localvit_encoder_01(lururu),self.localvit_encoder_01(lururd)
        lurdlu, lurdld, lurdru, lurdrd = self.localvit_encoder_01(lurdlu),self.localvit_encoder_01(lurdld),self.localvit_encoder_01(lurdru),self.localvit_encoder_01(lurdrd)
        # Reconstruction
        xlulu_lv = self.join2x2(xlulu_lv, lululu, lululd, luluru, lulurd)
        xluld_lv = self.join2x2(xluld_lv, luldlu, luldld, luldru, luldrd)
        xluru_lv = self.join2x2(xluru_lv, lurulu, luruld, lururu, lururd)
        xlurd_lv = self.join2x2(xlurd_lv, lurdlu, lurdld, lurdru, lurdrd)
        xlu_lv = self.join2x2(xlu_lv, xlulu_lv, xluld_lv, xluru_lv, xlurd_lv)
        

        ldlulu, ldluld, ldluru, ldlurd = self.localvit_encoder_01(ldlulu),self.localvit_encoder_01(ldluld),self.localvit_encoder_01(ldluru),self.localvit_encoder_01(ldlurd)
        ldldlu, ldldld, ldldru, ldldrd = self.localvit_encoder_01(ldldlu),self.localvit_encoder_01(ldldld),self.localvit_encoder_01(ldldru),self.localvit_encoder_01(ldldrd)
        ldrulu, ldruld, ldruru, ldrurd = self.localvit_encoder_01(ldrulu),self.localvit_encoder_01(ldruld),self.localvit_encoder_01(ldruru),self.localvit_encoder_01(ldrurd)
        ldrdlu, ldrdld, ldrdru, ldrdrd = self.localvit_encoder_01(ldrdlu),self.localvit_encoder_01(ldrdld),self.localvit_encoder_01(ldrdru),self.localvit_encoder_01(ldrdrd)
        xldlu_lv = self.join2x2(xldlu_lv, ldlulu, ldluld, ldluru, ldlurd)
        xldld_lv = self.join2x2(xldld_lv, ldldlu, ldldld, ldldru, ldldrd)
        xldru_lv = self.join2x2(xldru_lv, ldrulu, ldruld, ldruru, ldrurd)
        xldrd_lv = self.join2x2(xldrd_lv, ldrdlu, ldrdld, ldrdru, ldrdrd)
        xld_lv = self.join2x2(xld_lv, xldlu_lv, xldld_lv, xldru_lv, xldrd_lv)
        

        rululu, rululd, ruluru, rulurd = self.localvit_encoder_01(rululu),self.localvit_encoder_01(rululd),self.localvit_encoder_01(ruluru),self.localvit_encoder_01(rulurd)
        ruldlu, ruldld, ruldru, ruldrd = self.localvit_encoder_01(ruldlu),self.localvit_encoder_01(ruldld),self.localvit_encoder_01(ruldru),self.localvit_encoder_01(ruldrd)
        rurulu, ruruld, rururu, rururd = self.localvit_encoder_01(rurulu),self.localvit_encoder_01(ruruld),self.localvit_encoder_01(rururu),self.localvit_encoder_01(rururd)
        rurdlu, rurdld, rurdru, rurdrd = self.localvit_encoder_01(rurdlu),self.localvit_encoder_01(rurdld),self.localvit_encoder_01(rurdru),self.localvit_encoder_01(rurdrd)
        xrulu_lv = self.join2x2(xrulu_lv, rululu, rululd, ruluru, rulurd)
        xruld_lv = self.join2x2(xruld_lv, ruldlu, ruldld, ruldru, ruldrd)
        xruru_lv = self.join2x2(xruru_lv, rurulu, ruruld, rururu, rururd)
        xrurd_lv = self.join2x2(xrurd_lv, rurdlu, rurdld, rurdru, rurdrd)
        xru_lv = self.join2x2(xru_lv, xrulu_lv, xruld_lv, xruru_lv, xrurd_lv)
        

        rdlulu, rdluld, rdluru, rdlurd = self.localvit_encoder_01(rdlulu),self.localvit_encoder_01(rdluld),self.localvit_encoder_01(rdluru),self.localvit_encoder_01(rdlurd)
        rdldlu, rdldld, rdldru, rdldrd = self.localvit_encoder_01(rdldlu),self.localvit_encoder_01(rdldld),self.localvit_encoder_01(rdldru),self.localvit_encoder_01(rdldrd)
        rdrulu, rdruld, rdruru, rdrurd = self.localvit_encoder_01(rdrulu),self.localvit_encoder_01(rdruld),self.localvit_encoder_01(rdruru),self.localvit_encoder_01(rdrurd)
        rdrdlu, rdrdld, rdrdru, rdrdrd = self.localvit_encoder_01(rdrdlu),self.localvit_encoder_01(rdrdld),self.localvit_encoder_01(rdrdru),self.localvit_encoder_01(rdrdrd)
        xrdlu_lv = self.join2x2(xrdlu_lv, rdlulu, rdluld, rdluru, rdlurd)
        xrdld_lv = self.join2x2(xrdld_lv, rdldlu, rdldld, rdldru, rdldrd)
        xrdru_lv = self.join2x2(xrdru_lv, rdrulu, rdruld, rdruru, rdrurd)
        xrdrd_lv = self.join2x2(xrdrd_lv, rdrdlu, rdrdld, rdrdru, rdrdrd)
        xrd_lv = self.join2x2(xrd_lv, xrdlu_lv, xrdld_lv, xrdru_lv, xrdrd_lv)

        # Reconstruction
        x_lve_01 =self.join2x2(x_lve_01, xlu_lv, xld_lv, xru_lv, xrd_lv)

        ## Global ViT
        #x_gve_01 = self.globalvit_encoder_01(xf)

        # Merge the Local ViT and Global ViT, and downscaling
        x_e_01 = x_lve_01 + xf

        ### Encoder Level Two ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        x_e_01_ds = self.ds_conv_e02(x_e_01)
        [x_lve_02, lu, ld, ru, rd] = self.crop2x2(x_e_01_ds)

        # Origional feature maps are cropped to 4x4 regions
        [xlu_lv, lulu, luld, luru, lurd] = self.crop2x2(lu)
        [xld_lv, ldlu, ldld, ldru, ldrd] = self.crop2x2(ld)
        [xru_lv, rulu, ruld, ruru, rurd] = self.crop2x2(ru)
        [xrd_lv, rdlu, rdld, rdru, rdrd] = self.crop2x2(rd)

        # Local ViT for each (total 16) regions
        lulu, luld, luru, lurd = self.localvit_encoder_02(lulu),self.localvit_encoder_02(luld),self.localvit_encoder_02(luru),self.localvit_encoder_02(lurd)
        ldlu, ldld, ldru, ldrd = self.localvit_encoder_02(ldlu),self.localvit_encoder_02(ldld),self.localvit_encoder_02(ldru),self.localvit_encoder_02(ldrd)
        rulu, ruld, ruru, rurd = self.localvit_encoder_02(rulu),self.localvit_encoder_02(ruld),self.localvit_encoder_02(ruru),self.localvit_encoder_02(rurd)
        rdlu, rdld, rdru, rdrd = self.localvit_encoder_02(rdlu),self.localvit_encoder_02(rdld),self.localvit_encoder_02(rdru),self.localvit_encoder_02(rdrd)

        # Reconstruction
        xlu_lv = self.join2x2(xlu_lv, lulu, luld, luru, lurd)
        xld_lv = self.join2x2(xld_lv, ldlu, ldld, ldru, ldrd)
        xru_lv = self.join2x2(xru_lv, rulu, ruld, ruru, rurd)
        xrd_lv = self.join2x2(xrd_lv, rdlu, rdld, rdru, rdrd)

        # Reconstruction
        x_lve_02 =self.join2x2(x_lve_02, xlu_lv, xld_lv, xru_lv, xrd_lv)

        ## Global ViT
        #x_gve_02 = self.globalvit_encoder_02(x_e_01_ds)

        # Merge the Local ViT and Global ViT, and downscaling
        x_e_02 = x_lve_02 + x_e_01_ds

        ### Encoder Level Three ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        x_e_02_ds = self.ds_conv_e03(x_e_02)
        [x_lve, lu, ld, ru, rd] = self.crop2x2(x_e_02_ds)

        # Local ViT for each (total 4) regions
        lu, ld, ru, rd = self.localvit_encoder_03(lu),self.localvit_encoder_03(ld),self.localvit_encoder_03(ru),self.localvit_encoder_03(rd)

        # Reconstruction
        x_lve_03 =self.join2x2(x_lve, lu, ld, ru, rd)

        ## Global ViT
        #x_gve_03 = self.globalvit_encoder_03(x_e_02_ds)

        # Merge the Local ViT and Global ViT, and upscaling
        x_e_03 = x_lve_03 + x_e_02_ds
        


        ############################################################
        ##                Reflectance Image Decoder               ##
        ############################################################
        ### Dcoder Level Three ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        r_d_03_in = x_e_03
        [r_lvd_03, lu, ld, ru, rd] = self.crop2x2(r_d_03_in)

        # Local ViT for each (total 4) regions
        lu, ld, ru, rd = self.localvit_decoder_03r(lu),self.localvit_decoder_03r(ld),self.localvit_decoder_03r(ru),self.localvit_decoder_03r(rd)

        # Reconstruction
        r_lvd_03 =self.join2x2(r_lvd_03, lu, ld, ru, rd)

        ## Global ViT

        #r_gvd_03 = self.globalvit_decoder_03r(r_d_03_in)

        # Merge the Local ViT and Global ViT, and upscaling
        #print('r_lvd_03', r_lvd_03.shape, 'r_gvd_03', r_gvd_03.shape, 'r_d_03_in', r_d_03_in.shape,)
        r_d_03 = r_lvd_03 + r_d_03_in
        

        ### Decoder Level Two ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        r_d_03_us = self.us_conv_d03r(r_d_03)
        r_d_02_in = self.sk_conv_d03r(torch.cat((r_d_03_us, x_e_02),1))
        [r_lvd_02, lu, ld, ru, rd] = self.crop2x2(r_d_02_in)

        # Origional feature maps are cropped to 4x4 regions
        [xlu_lv, lulu, luld, luru, lurd] = self.crop2x2(lu)
        [xld_lv, ldlu, ldld, ldru, ldrd] = self.crop2x2(ld)
        [xru_lv, rulu, ruld, ruru, rurd] = self.crop2x2(ru)
        [xrd_lv, rdlu, rdld, rdru, rdrd] = self.crop2x2(rd)

        # Local ViT for each (total 16) regions
        lulu, luld, luru, lurd = self.localvit_decoder_02r(lulu),self.localvit_decoder_02r(luld),self.localvit_decoder_02r(luru),self.localvit_decoder_02r(lurd)
        ldlu, ldld, ldru, ldrd = self.localvit_decoder_02r(ldlu),self.localvit_decoder_02r(ldld),self.localvit_decoder_02r(ldru),self.localvit_decoder_02r(ldrd)
        rulu, ruld, ruru, rurd = self.localvit_decoder_02r(rulu),self.localvit_decoder_02r(ruld),self.localvit_decoder_02r(ruru),self.localvit_decoder_02r(rurd)
        rdlu, rdld, rdru, rdrd = self.localvit_decoder_02r(rdlu),self.localvit_decoder_02r(rdld),self.localvit_decoder_02r(rdru),self.localvit_decoder_02r(rdrd)

        # Reconstruction
        xlu_lv = self.join2x2(xlu_lv, lulu, luld, luru, lurd)
        xld_lv = self.join2x2(xld_lv, ldlu, ldld, ldru, ldrd)
        xru_lv = self.join2x2(xru_lv, rulu, ruld, ruru, rurd)
        xrd_lv = self.join2x2(xrd_lv, rdlu, rdld, rdru, rdrd)

        # Reconstruction
        r_lvd_02 =self.join2x2(r_lvd_02, xlu_lv, xld_lv, xru_lv, xrd_lv)

        ## Global ViT
        #r_gvd_02 = self.globalvit_decoder_02r(r_d_02_in)

        # Merge the Local ViT and Global ViT, and upscaling
        r_d_02 = r_lvd_02 + r_d_02_in



        ### Decoder Level One ###
        ## Local ViT ##
        # Origional feature maps are cropped to 2x2 regions
        r_d_02_us = self.us_conv_d02r(r_d_02)
        r_d_01_in = self.sk_conv_d02r(torch.cat((r_d_02_us, x_e_01),1))
        [r_lvd_01, lu, ld, ru, rd] = self.crop2x2(r_d_01_in)

        # Origional feature mape cropped to 4x4 regions
        [xlu_lv, lulu, luld, luru, lurd] = self.crop2x2(lu)
        [xld_lv, ldlu, ldld, ldru, ldrd] = self.crop2x2(ld)
        [xru_lv, rulu, ruld, ruru, rurd] = self.crop2x2(ru)
        [xrd_lv, rdlu, rdld, rdru, rdrd] = self.crop2x2(rd)

        # Origional feature maps are cropped to 8x8 regions
        [xlulu_lv, lululu, lululd, luluru, lulurd] = self.crop2x2(lulu)
        [xluld_lv, luldlu, luldld, luldru, luldrd] = self.crop2x2(luld)
        [xluru_lv, lurulu, luruld, lururu, lururd] = self.crop2x2(luru)
        [xlurd_lv, lurdlu, lurdld, lurdru, lurdrd] = self.crop2x2(lurd)
        

        [xldlu_lv, ldlulu, ldluld, ldluru, ldlurd] = self.crop2x2(ldlu)
        [xldld_lv, ldldlu, ldldld, ldldru, ldldrd] = self.crop2x2(ldld)
        [xldru_lv, ldrulu, ldruld, ldruru, ldrurd] = self.crop2x2(ldru)
        [xldrd_lv, ldrdlu, ldrdld, ldrdru, ldrdrd] = self.crop2x2(ldrd)


        [xrulu_lv, rululu, rululd, ruluru, rulurd] = self.crop2x2(rulu)
        [xruld_lv, ruldlu, ruldld, ruldru, ruldrd] = self.crop2x2(ruld)
        [xruru_lv, rurulu, ruruld, rururu, rururd] = self.crop2x2(ruru)
        [xrurd_lv, rurdlu, rurdld, rurdru, rurdrd] = self.crop2x2(rurd)
        

        [xrdlu_lv, rdlulu, rdluld, rdluru, rdlurd] = self.crop2x2(rdlu)
        [xrdld_lv, rdldlu, rdldld, rdldru, rdldrd] = self.crop2x2(rdld)
        [xrdru_lv, rdrulu, rdruld, rdruru, rdrurd] = self.crop2x2(rdru)
        [xrdrd_lv, rdrdlu, rdrdld, rdrdru, rdrdrd] = self.crop2x2(rdrd)


        # Local ViT for each (total 64) regions
        lululu, lululd, luluru, lulurd = self.localvit_decoder_01r(lululu),self.localvit_decoder_01r(lululd),self.localvit_decoder_01r(luluru),self.localvit_decoder_01r(lulurd)
        luldlu, luldld, luldru, luldrd = self.localvit_decoder_01r(luldlu),self.localvit_decoder_01r(luldld),self.localvit_decoder_01r(luldru),self.localvit_decoder_01r(luldrd)
        lurulu, luruld, lururu, lururd = self.localvit_decoder_01r(lurulu),self.localvit_decoder_01r(luruld),self.localvit_decoder_01r(lururu),self.localvit_decoder_01r(lururd)
        lurdlu, lurdld, lurdru, lurdrd = self.localvit_decoder_01r(lurdlu),self.localvit_decoder_01r(lurdld),self.localvit_decoder_01r(lurdru),self.localvit_decoder_01r(lurdrd)
        # Reconstruction
        xlulu_lv = self.join2x2(xlulu_lv, lululu, lululd, luluru, lulurd)
        xluld_lv = self.join2x2(xluld_lv, luldlu, luldld, luldru, luldrd)
        xluru_lv = self.join2x2(xluru_lv, lurulu, luruld, lururu, lururd)
        xlurd_lv = self.join2x2(xlurd_lv, lurdlu, lurdld, lurdru, lurdrd)
        xlu_lv = self.join2x2(xlu_lv, xlulu_lv, xluld_lv, xluru_lv, xlurd_lv)
        

        ldlulu, ldluld, ldluru, ldlurd = self.localvit_decoder_01r(ldlulu),self.localvit_decoder_01r(ldluld),self.localvit_decoder_01r(ldluru),self.localvit_decoder_01r(ldlurd)
        ldldlu, ldldld, ldldru, ldldrd = self.localvit_decoder_01r(ldldlu),self.localvit_decoder_01r(ldldld),self.localvit_decoder_01r(ldldru),self.localvit_decoder_01r(ldldrd)
        ldrulu, ldruld, ldruru, ldrurd = self.localvit_decoder_01r(ldrulu),self.localvit_decoder_01r(ldruld),self.localvit_decoder_01r(ldruru),self.localvit_decoder_01r(ldrurd)
        ldrdlu, ldrdld, ldrdru, ldrdrd = self.localvit_decoder_01r(ldrdlu),self.localvit_decoder_01r(ldrdld),self.localvit_decoder_01r(ldrdru),self.localvit_decoder_01r(ldrdrd)
        xldlu_lv = self.join2x2(xldlu_lv, ldlulu, ldluld, ldluru, ldlurd)
        xldld_lv = self.join2x2(xldld_lv, ldldlu, ldldld, ldldru, ldldrd)
        xldru_lv = self.join2x2(xldru_lv, ldrulu, ldruld, ldruru, ldrurd)
        xldrd_lv = self.join2x2(xldrd_lv, ldrdlu, ldrdld, ldrdru, ldrdrd)
        xld_lv = self.join2x2(xld_lv, xldlu_lv, xldld_lv, xldru_lv, xldrd_lv)
        

        rululu, rululd, ruluru, rulurd = self.localvit_decoder_01r(rululu),self.localvit_decoder_01r(rululd),self.localvit_decoder_01r(ruluru),self.localvit_decoder_01r(rulurd)
        ruldlu, ruldld, ruldru, ruldrd = self.localvit_decoder_01r(ruldlu),self.localvit_decoder_01r(ruldld),self.localvit_decoder_01r(ruldru),self.localvit_decoder_01r(ruldrd)
        rurulu, ruruld, rururu, rururd = self.localvit_decoder_01r(rurulu),self.localvit_decoder_01r(ruruld),self.localvit_decoder_01r(rururu),self.localvit_decoder_01r(rururd)
        rurdlu, rurdld, rurdru, rurdrd = self.localvit_decoder_01r(rurdlu),self.localvit_decoder_01r(rurdld),self.localvit_decoder_01r(rurdru),self.localvit_decoder_01r(rurdrd)
        xrulu_lv = self.join2x2(xrulu_lv, rululu, rululd, ruluru, rulurd)
        xruld_lv = self.join2x2(xruld_lv, ruldlu, ruldld, ruldru, ruldrd)
        xruru_lv = self.join2x2(xruru_lv, rurulu, ruruld, rururu, rururd)
        xrurd_lv = self.join2x2(xrurd_lv, rurdlu, rurdld, rurdru, rurdrd)
        xru_lv = self.join2x2(xru_lv, xrulu_lv, xruld_lv, xruru_lv, xrurd_lv)
        

        rdlulu, rdluld, rdluru, rdlurd = self.localvit_decoder_01r(rdlulu),self.localvit_decoder_01r(rdluld),self.localvit_decoder_01r(rdluru),self.localvit_decoder_01r(rdlurd)
        rdldlu, rdldld, rdldru, rdldrd = self.localvit_decoder_01r(rdldlu),self.localvit_decoder_01r(rdldld),self.localvit_decoder_01r(rdldru),self.localvit_decoder_01r(rdldrd)
        rdrulu, rdruld, rdruru, rdrurd = self.localvit_decoder_01r(rdrulu),self.localvit_decoder_01r(rdruld),self.localvit_decoder_01r(rdruru),self.localvit_decoder_01r(rdrurd)
        rdrdlu, rdrdld, rdrdru, rdrdrd = self.localvit_decoder_01r(rdrdlu),self.localvit_decoder_01r(rdrdld),self.localvit_decoder_01r(rdrdru),self.localvit_decoder_01r(rdrdrd)
        xrdlu_lv = self.join2x2(xrdlu_lv, rdlulu, rdluld, rdluru, rdlurd)
        xrdld_lv = self.join2x2(xrdld_lv, rdldlu, rdldld, rdldru, rdldrd)
        xrdru_lv = self.join2x2(xrdru_lv, rdrulu, rdruld, rdruru, rdrurd)
        xrdrd_lv = self.join2x2(xrdrd_lv, rdrdlu, rdrdld, rdrdru, rdrdrd)
        xrd_lv = self.join2x2(xrd_lv, xrdlu_lv, xrdld_lv, xrdru_lv, xrdrd_lv)

        # Reconstruction
        r_lvd_01 =self.join2x2(r_lvd_01, xlu_lv, xld_lv, xru_lv, xrd_lv)

        ## Global ViT
        #r_gvd_01 = self.globalvit_decoder_01r(r_d_01_in)

        # Merge the Local ViT and Global ViT, and downscaling
        r_d_01 = r_lvd_01 + r_d_01_in
        

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
        [s_lvd_03, lu, ld, ru, rd] = self.crop2x2(s_d_03_in)

        # Local ViT for each (total 4) regions
        lu, ld, ru, rd = self.localvit_decoder_03s(lu),self.localvit_decoder_03s(ld),self.localvit_decoder_03s(ru),self.localvit_decoder_03s(rd)

        # Reconstruction
        s_lvd_03 =self.join2x2(s_lvd_03, lu, ld, ru, rd)

        ## Global ViT

        #s_gvd_03 = self.globalvit_decoder_03s(s_d_03_in)

        # Merge the Local ViT and Global ViT, and upscaling
        s_d_03 = s_lvd_03 + s_d_03_in
        
        ### Decoder Level Two ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        s_d_03_us = self.us_conv_d03s(s_d_03) 
        s_d_02_in = self.sk_conv_d03s(torch.cat((s_d_03_us, x_e_02),1))
        [s_lvd_02, lu, ld, ru, rd] = self.crop2x2(s_d_02_in)

        # Origional feature maps are cropped to 4x4 regions
        [xlu_lv, lulu, luld, luru, lurd] = self.crop2x2(lu)
        [xld_lv, ldlu, ldld, ldru, ldrd] = self.crop2x2(ld)
        [xru_lv, rulu, ruld, ruru, rurd] = self.crop2x2(ru)
        [xrd_lv, rdlu, rdld, rdru, rdrd] = self.crop2x2(rd)

        # Local ViT for each (total 16) regions
        lulu, luld, luru, lurd = self.localvit_decoder_02s(lulu),self.localvit_decoder_02s(luld),self.localvit_decoder_02s(luru),self.localvit_decoder_02s(lurd)
        ldlu, ldld, ldru, ldrd = self.localvit_decoder_02s(ldlu),self.localvit_decoder_02s(ldld),self.localvit_decoder_02s(ldru),self.localvit_decoder_02s(ldrd)
        rulu, ruld, ruru, rurd = self.localvit_decoder_02s(rulu),self.localvit_decoder_02s(ruld),self.localvit_decoder_02s(ruru),self.localvit_decoder_02s(rurd)
        rdlu, rdld, rdru, rdrd = self.localvit_decoder_02s(rdlu),self.localvit_decoder_02s(rdld),self.localvit_decoder_02s(rdru),self.localvit_decoder_02s(rdrd)

        # Reconstruction
        xlu_lv = self.join2x2(xlu_lv, lulu, luld, luru, lurd)
        xld_lv = self.join2x2(xld_lv, ldlu, ldld, ldru, ldrd)
        xru_lv = self.join2x2(xru_lv, rulu, ruld, ruru, rurd)
        xrd_lv = self.join2x2(xrd_lv, rdlu, rdld, rdru, rdrd)

        # Reconstruction
        s_lvd_02 =self.join2x2(s_lvd_02, xlu_lv, xld_lv, xru_lv, xrd_lv)

        ## Global ViT
        #s_gvd_02 = self.globalvit_decoder_02s(s_d_02_in)

        # Merge the Local ViT and Global ViT, and upscaling
        s_d_02 = s_lvd_02 + s_d_02_in


        ### Encoder Level One ###
        ## Local ViT ##
        # Origional feature maps are cropped to 2x2 regions
        s_d_02_us = self.us_conv_d02s(s_d_02) 
        s_d_01_in = self.sk_conv_d02s(torch.cat((s_d_02_us, x_e_01),1))
        [s_lvd_01, lu, ld, ru, rd] = self.crop2x2(s_d_01_in)

        # Origional feature mape cropped to 4x4 regions
        [xlu_lv, lulu, luld, luru, lurd] = self.crop2x2(lu)
        [xld_lv, ldlu, ldld, ldru, ldrd] = self.crop2x2(ld)
        [xru_lv, rulu, ruld, ruru, rurd] = self.crop2x2(ru)
        [xrd_lv, rdlu, rdld, rdru, rdrd] = self.crop2x2(rd)

        # Origional feature maps are cropped to 8x8 regions
        [xlulu_lv, lululu, lululd, luluru, lulurd] = self.crop2x2(lulu)
        [xluld_lv, luldlu, luldld, luldru, luldrd] = self.crop2x2(luld)
        [xluru_lv, lurulu, luruld, lururu, lururd] = self.crop2x2(luru)
        [xlurd_lv, lurdlu, lurdld, lurdru, lurdrd] = self.crop2x2(lurd)
        

        [xldlu_lv, ldlulu, ldluld, ldluru, ldlurd] = self.crop2x2(ldlu)
        [xldld_lv, ldldlu, ldldld, ldldru, ldldrd] = self.crop2x2(ldld)
        [xldru_lv, ldrulu, ldruld, ldruru, ldrurd] = self.crop2x2(ldru)
        [xldrd_lv, ldrdlu, ldrdld, ldrdru, ldrdrd] = self.crop2x2(ldrd)


        [xrulu_lv, rululu, rululd, ruluru, rulurd] = self.crop2x2(rulu)
        [xruld_lv, ruldlu, ruldld, ruldru, ruldrd] = self.crop2x2(ruld)
        [xruru_lv, rurulu, ruruld, rururu, rururd] = self.crop2x2(ruru)
        [xrurd_lv, rurdlu, rurdld, rurdru, rurdrd] = self.crop2x2(rurd)
        

        [xrdlu_lv, rdlulu, rdluld, rdluru, rdlurd] = self.crop2x2(rdlu)
        [xrdld_lv, rdldlu, rdldld, rdldru, rdldrd] = self.crop2x2(rdld)
        [xrdru_lv, rdrulu, rdruld, rdruru, rdrurd] = self.crop2x2(rdru)
        [xrdrd_lv, rdrdlu, rdrdld, rdrdru, rdrdrd] = self.crop2x2(rdrd)


        # Local ViT for each (total 64) regions
        lululu, lululd, luluru, lulurd = self.localvit_decoder_01s(lululu),self.localvit_decoder_01s(lululd),self.localvit_decoder_01s(luluru),self.localvit_decoder_01s(lulurd)
        luldlu, luldld, luldru, luldrd = self.localvit_decoder_01s(luldlu),self.localvit_decoder_01s(luldld),self.localvit_decoder_01s(luldru),self.localvit_decoder_01s(luldrd)
        lurulu, luruld, lururu, lururd = self.localvit_decoder_01s(lurulu),self.localvit_decoder_01s(luruld),self.localvit_decoder_01s(lururu),self.localvit_decoder_01s(lururd)
        lurdlu, lurdld, lurdru, lurdrd = self.localvit_decoder_01s(lurdlu),self.localvit_decoder_01s(lurdld),self.localvit_decoder_01s(lurdru),self.localvit_decoder_01s(lurdrd)
        # Reconstruction
        xlulu_lv = self.join2x2(xlulu_lv, lululu, lululd, luluru, lulurd)
        xluld_lv = self.join2x2(xluld_lv, luldlu, luldld, luldru, luldrd)
        xluru_lv = self.join2x2(xluru_lv, lurulu, luruld, lururu, lururd)
        xlurd_lv = self.join2x2(xlurd_lv, lurdlu, lurdld, lurdru, lurdrd)
        xlu_lv = self.join2x2(xlu_lv, xlulu_lv, xluld_lv, xluru_lv, xlurd_lv)
        

        ldlulu, ldluld, ldluru, ldlurd = self.localvit_decoder_01s(ldlulu),self.localvit_decoder_01s(ldluld),self.localvit_decoder_01s(ldluru),self.localvit_decoder_01s(ldlurd)
        ldldlu, ldldld, ldldru, ldldrd = self.localvit_decoder_01s(ldldlu),self.localvit_decoder_01s(ldldld),self.localvit_decoder_01s(ldldru),self.localvit_decoder_01s(ldldrd)
        ldrulu, ldruld, ldruru, ldrurd = self.localvit_decoder_01s(ldrulu),self.localvit_decoder_01s(ldruld),self.localvit_decoder_01s(ldruru),self.localvit_decoder_01s(ldrurd)
        ldrdlu, ldrdld, ldrdru, ldrdrd = self.localvit_decoder_01s(ldrdlu),self.localvit_decoder_01s(ldrdld),self.localvit_decoder_01s(ldrdru),self.localvit_decoder_01s(ldrdrd)
        xldlu_lv = self.join2x2(xldlu_lv, ldlulu, ldluld, ldluru, ldlurd)
        xldld_lv = self.join2x2(xldld_lv, ldldlu, ldldld, ldldru, ldldrd)
        xldru_lv = self.join2x2(xldru_lv, ldrulu, ldruld, ldruru, ldrurd)
        xldrd_lv = self.join2x2(xldrd_lv, ldrdlu, ldrdld, ldrdru, ldrdrd)
        xld_lv = self.join2x2(xld_lv, xldlu_lv, xldld_lv, xldru_lv, xldrd_lv)
        

        rululu, rululd, ruluru, rulurd = self.localvit_decoder_01s(rululu),self.localvit_decoder_01s(rululd),self.localvit_decoder_01s(ruluru),self.localvit_decoder_01s(rulurd)
        ruldlu, ruldld, ruldru, ruldrd = self.localvit_decoder_01s(ruldlu),self.localvit_decoder_01s(ruldld),self.localvit_decoder_01s(ruldru),self.localvit_decoder_01s(ruldrd)
        rurulu, ruruld, rururu, rururd = self.localvit_decoder_01s(rurulu),self.localvit_decoder_01s(ruruld),self.localvit_decoder_01s(rururu),self.localvit_decoder_01s(rururd)
        rurdlu, rurdld, rurdru, rurdrd = self.localvit_decoder_01s(rurdlu),self.localvit_decoder_01s(rurdld),self.localvit_decoder_01s(rurdru),self.localvit_decoder_01s(rurdrd)
        xrulu_lv = self.join2x2(xrulu_lv, rululu, rululd, ruluru, rulurd)
        xruld_lv = self.join2x2(xruld_lv, ruldlu, ruldld, ruldru, ruldrd)
        xruru_lv = self.join2x2(xruru_lv, rurulu, ruruld, rururu, rururd)
        xrurd_lv = self.join2x2(xrurd_lv, rurdlu, rurdld, rurdru, rurdrd)
        xru_lv = self.join2x2(xru_lv, xrulu_lv, xruld_lv, xruru_lv, xrurd_lv)
        

        rdlulu, rdluld, rdluru, rdlurd = self.localvit_decoder_01s(rdlulu),self.localvit_decoder_01s(rdluld),self.localvit_decoder_01s(rdluru),self.localvit_decoder_01s(rdlurd)
        rdldlu, rdldld, rdldru, rdldrd = self.localvit_decoder_01s(rdldlu),self.localvit_decoder_01s(rdldld),self.localvit_decoder_01s(rdldru),self.localvit_decoder_01s(rdldrd)
        rdrulu, rdruld, rdruru, rdrurd = self.localvit_decoder_01s(rdrulu),self.localvit_decoder_01s(rdruld),self.localvit_decoder_01s(rdruru),self.localvit_decoder_01s(rdrurd)
        rdrdlu, rdrdld, rdrdru, rdrdrd = self.localvit_decoder_01s(rdrdlu),self.localvit_decoder_01s(rdrdld),self.localvit_decoder_01s(rdrdru),self.localvit_decoder_01s(rdrdrd)
        xrdlu_lv = self.join2x2(xrdlu_lv, rdlulu, rdluld, rdluru, rdlurd)
        xrdld_lv = self.join2x2(xrdld_lv, rdldlu, rdldld, rdldru, rdldrd)
        xrdru_lv = self.join2x2(xrdru_lv, rdrulu, rdruld, rdruru, rdrurd)
        xrdrd_lv = self.join2x2(xrdrd_lv, rdrdlu, rdrdld, rdrdru, rdrdrd)
        xrd_lv = self.join2x2(xrd_lv, xrdlu_lv, xrdld_lv, xrdru_lv, xrdrd_lv)

        # Reconstruction
        s_lvd_01 =self.join2x2(s_lvd_01, xlu_lv, xld_lv, xru_lv, xrd_lv)

        ## Global ViT
        #s_gvd_01 = self.globalvit_decoder_01s(s_d_01_in)

        # Merge the Local ViT and Global ViT, and downscaling
        s_d_01 = s_lvd_01 + s_d_01_in

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
        [d_lvd_03, lu, ld, ru, rd] = self.crop2x2(d_d_03_in)

        # Local ViT for each (total 4) regions
        lu, ld, ru, rd = self.localvit_decoder_03d(lu),self.localvit_decoder_03d(ld),self.localvit_decoder_03d(ru),self.localvit_decoder_03d(rd)

        # Reconstruction
        d_lvd_03 =self.join2x2(d_lvd_03, lu, ld, ru, rd)

        ## Global ViT

        #d_gvd_03 = self.globalvit_decoder_03d(d_d_03_in)

        # Merge the Local ViT and Global ViT, and upscaling
        d_d_03 = d_lvd_03 + d_d_03_in
        
        ### Decoder Level Two ###
        ## Loval ViT
        # Origional feature maps are cropped to 2x2 regions
        d_d_03_us = self.us_conv_d03d(d_d_03) 
        d_d_02_in = self.sk_conv_d03d(torch.cat((d_d_03_us, r_d_03_us, s_d_03_us),1))
        [d_lvd_02, lu, ld, ru, rd] = self.crop2x2(d_d_02_in)

        # Origional feature maps are cropped to 4x4 regions
        [xlu_lv, lulu, luld, luru, lurd] = self.crop2x2(lu)
        [xld_lv, ldlu, ldld, ldru, ldrd] = self.crop2x2(ld)
        [xru_lv, rulu, ruld, ruru, rurd] = self.crop2x2(ru)
        [xrd_lv, rdlu, rdld, rdru, rdrd] = self.crop2x2(rd)

        # Local ViT for each (total 16) regions
        lulu, luld, luru, lurd = self.localvit_decoder_02d(lulu),self.localvit_decoder_02d(luld),self.localvit_decoder_02d(luru),self.localvit_decoder_02d(lurd)
        ldlu, ldld, ldru, ldrd = self.localvit_decoder_02d(ldlu),self.localvit_decoder_02d(ldld),self.localvit_decoder_02d(ldru),self.localvit_decoder_02d(ldrd)
        rulu, ruld, ruru, rurd = self.localvit_decoder_02d(rulu),self.localvit_decoder_02d(ruld),self.localvit_decoder_02d(ruru),self.localvit_decoder_02d(rurd)
        rdlu, rdld, rdru, rdrd = self.localvit_decoder_02d(rdlu),self.localvit_decoder_02d(rdld),self.localvit_decoder_02d(rdru),self.localvit_decoder_02d(rdrd)

        # Reconstruction
        xlu_lv = self.join2x2(xlu_lv, lulu, luld, luru, lurd)
        xld_lv = self.join2x2(xld_lv, ldlu, ldld, ldru, ldrd)
        xru_lv = self.join2x2(xru_lv, rulu, ruld, ruru, rurd)
        xrd_lv = self.join2x2(xrd_lv, rdlu, rdld, rdru, rdrd)

        # Reconstruction
        d_lvd_02 =self.join2x2(d_lvd_02, xlu_lv, xld_lv, xru_lv, xrd_lv)

        ## Global ViT
        #d_gvd_02 = self.globalvit_decoder_02d(d_d_02_in)

        # Merge the Local ViT and Global ViT, and upscaling
        d_d_02 = d_lvd_02 + d_d_02_in


        ### Encoder Level One ###
        ## Local ViT ##
        # Origional feature maps are cropped to 2x2 regions
        d_d_02_us = self.us_conv_d02d(s_d_02) 
        d_d_01_in = self.sk_conv_d02d(torch.cat((d_d_02_us, r_d_02_us, s_d_02_us),1))
        [d_lvd_01, lu, ld, ru, rd] = self.crop2x2(d_d_01_in)

        # Origional feature mape cropped to 4x4 regions
        [xlu_lv, lulu, luld, luru, lurd] = self.crop2x2(lu)
        [xld_lv, ldlu, ldld, ldru, ldrd] = self.crop2x2(ld)
        [xru_lv, rulu, ruld, ruru, rurd] = self.crop2x2(ru)
        [xrd_lv, rdlu, rdld, rdru, rdrd] = self.crop2x2(rd)

        # Origional feature maps are cropped to 8x8 regions
        [xlulu_lv, lululu, lululd, luluru, lulurd] = self.crop2x2(lulu)
        [xluld_lv, luldlu, luldld, luldru, luldrd] = self.crop2x2(luld)
        [xluru_lv, lurulu, luruld, lururu, lururd] = self.crop2x2(luru)
        [xlurd_lv, lurdlu, lurdld, lurdru, lurdrd] = self.crop2x2(lurd)
        

        [xldlu_lv, ldlulu, ldluld, ldluru, ldlurd] = self.crop2x2(ldlu)
        [xldld_lv, ldldlu, ldldld, ldldru, ldldrd] = self.crop2x2(ldld)
        [xldru_lv, ldrulu, ldruld, ldruru, ldrurd] = self.crop2x2(ldru)
        [xldrd_lv, ldrdlu, ldrdld, ldrdru, ldrdrd] = self.crop2x2(ldrd)


        [xrulu_lv, rululu, rululd, ruluru, rulurd] = self.crop2x2(rulu)
        [xruld_lv, ruldlu, ruldld, ruldru, ruldrd] = self.crop2x2(ruld)
        [xruru_lv, rurulu, ruruld, rururu, rururd] = self.crop2x2(ruru)
        [xrurd_lv, rurdlu, rurdld, rurdru, rurdrd] = self.crop2x2(rurd)
        

        [xrdlu_lv, rdlulu, rdluld, rdluru, rdlurd] = self.crop2x2(rdlu)
        [xrdld_lv, rdldlu, rdldld, rdldru, rdldrd] = self.crop2x2(rdld)
        [xrdru_lv, rdrulu, rdruld, rdruru, rdrurd] = self.crop2x2(rdru)
        [xrdrd_lv, rdrdlu, rdrdld, rdrdru, rdrdrd] = self.crop2x2(rdrd)


        # Local ViT for each (total 64) regions
        lululu, lululd, luluru, lulurd = self.localvit_decoder_01d(lululu),self.localvit_decoder_01d(lululd),self.localvit_decoder_01d(luluru),self.localvit_decoder_01d(lulurd)
        luldlu, luldld, luldru, luldrd = self.localvit_decoder_01d(luldlu),self.localvit_decoder_01d(luldld),self.localvit_decoder_01d(luldru),self.localvit_decoder_01d(luldrd)
        lurulu, luruld, lururu, lururd = self.localvit_decoder_01d(lurulu),self.localvit_decoder_01d(luruld),self.localvit_decoder_01d(lururu),self.localvit_decoder_01d(lururd)
        lurdlu, lurdld, lurdru, lurdrd = self.localvit_decoder_01d(lurdlu),self.localvit_decoder_01d(lurdld),self.localvit_decoder_01d(lurdru),self.localvit_decoder_01d(lurdrd)
        # Reconstruction
        xlulu_lv = self.join2x2(xlulu_lv, lululu, lululd, luluru, lulurd)
        xluld_lv = self.join2x2(xluld_lv, luldlu, luldld, luldru, luldrd)
        xluru_lv = self.join2x2(xluru_lv, lurulu, luruld, lururu, lururd)
        xlurd_lv = self.join2x2(xlurd_lv, lurdlu, lurdld, lurdru, lurdrd)
        xlu_lv = self.join2x2(xlu_lv, xlulu_lv, xluld_lv, xluru_lv, xlurd_lv)
        

        ldlulu, ldluld, ldluru, ldlurd = self.localvit_decoder_01d(ldlulu),self.localvit_decoder_01d(ldluld),self.localvit_decoder_01d(ldluru),self.localvit_decoder_01d(ldlurd)
        ldldlu, ldldld, ldldru, ldldrd = self.localvit_decoder_01d(ldldlu),self.localvit_decoder_01d(ldldld),self.localvit_decoder_01d(ldldru),self.localvit_decoder_01d(ldldrd)
        ldrulu, ldruld, ldruru, ldrurd = self.localvit_decoder_01d(ldrulu),self.localvit_decoder_01d(ldruld),self.localvit_decoder_01d(ldruru),self.localvit_decoder_01d(ldrurd)
        ldrdlu, ldrdld, ldrdru, ldrdrd = self.localvit_decoder_01d(ldrdlu),self.localvit_decoder_01d(ldrdld),self.localvit_decoder_01d(ldrdru),self.localvit_decoder_01d(ldrdrd)
        xldlu_lv = self.join2x2(xldlu_lv, ldlulu, ldluld, ldluru, ldlurd)
        xldld_lv = self.join2x2(xldld_lv, ldldlu, ldldld, ldldru, ldldrd)
        xldru_lv = self.join2x2(xldru_lv, ldrulu, ldruld, ldruru, ldrurd)
        xldrd_lv = self.join2x2(xldrd_lv, ldrdlu, ldrdld, ldrdru, ldrdrd)
        xld_lv = self.join2x2(xld_lv, xldlu_lv, xldld_lv, xldru_lv, xldrd_lv)
        

        rululu, rululd, ruluru, rulurd = self.localvit_decoder_01d(rululu),self.localvit_decoder_01d(rululd),self.localvit_decoder_01d(ruluru),self.localvit_decoder_01d(rulurd)
        ruldlu, ruldld, ruldru, ruldrd = self.localvit_decoder_01d(ruldlu),self.localvit_decoder_01d(ruldld),self.localvit_decoder_01d(ruldru),self.localvit_decoder_01d(ruldrd)
        rurulu, ruruld, rururu, rururd = self.localvit_decoder_01d(rurulu),self.localvit_decoder_01d(ruruld),self.localvit_decoder_01d(rururu),self.localvit_decoder_01d(rururd)
        rurdlu, rurdld, rurdru, rurdrd = self.localvit_decoder_01d(rurdlu),self.localvit_decoder_01d(rurdld),self.localvit_decoder_01d(rurdru),self.localvit_decoder_01d(rurdrd)
        xrulu_lv = self.join2x2(xrulu_lv, rululu, rululd, ruluru, rulurd)
        xruld_lv = self.join2x2(xruld_lv, ruldlu, ruldld, ruldru, ruldrd)
        xruru_lv = self.join2x2(xruru_lv, rurulu, ruruld, rururu, rururd)
        xrurd_lv = self.join2x2(xrurd_lv, rurdlu, rurdld, rurdru, rurdrd)
        xru_lv = self.join2x2(xru_lv, xrulu_lv, xruld_lv, xruru_lv, xrurd_lv)
        

        rdlulu, rdluld, rdluru, rdlurd = self.localvit_decoder_01d(rdlulu),self.localvit_decoder_01d(rdluld),self.localvit_decoder_01d(rdluru),self.localvit_decoder_01d(rdlurd)
        rdldlu, rdldld, rdldru, rdldrd = self.localvit_decoder_01d(rdldlu),self.localvit_decoder_01d(rdldld),self.localvit_decoder_01d(rdldru),self.localvit_decoder_01d(rdldrd)
        rdrulu, rdruld, rdruru, rdrurd = self.localvit_decoder_01d(rdrulu),self.localvit_decoder_01d(rdruld),self.localvit_decoder_01d(rdruru),self.localvit_decoder_01d(rdrurd)
        rdrdlu, rdrdld, rdrdru, rdrdrd = self.localvit_decoder_01d(rdrdlu),self.localvit_decoder_01d(rdrdld),self.localvit_decoder_01d(rdrdru),self.localvit_decoder_01d(rdrdrd)
        xrdlu_lv = self.join2x2(xrdlu_lv, rdlulu, rdluld, rdluru, rdlurd)
        xrdld_lv = self.join2x2(xrdld_lv, rdldlu, rdldld, rdldru, rdldrd)
        xrdru_lv = self.join2x2(xrdru_lv, rdrulu, rdruld, rdruru, rdrurd)
        xrdrd_lv = self.join2x2(xrdrd_lv, rdrdlu, rdrdld, rdrdru, rdrdrd)
        xrd_lv = self.join2x2(xrd_lv, xrdlu_lv, xrdld_lv, xrdru_lv, xrdrd_lv)

        # Reconstruction
        d_lvd_01 =self.join2x2(d_lvd_01, xlu_lv, xld_lv, xru_lv, xrd_lv)

        ## Global ViT
        #d_gvd_01 = self.globalvit_decoder_01d(d_d_01_in)

        # Merge the Local ViT and Global ViT, and downscaling
        d_d_01 = d_lvd_01 + d_d_01_in

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
        x = self.ave_pool2(self.ave_pool2(x))
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