### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from epdn import common
import torch.nn.functional as F
from functools import reduce
#from epdn.train_options import TrainOptions
#opt = TrainOptions().parse()
from options.train_options import TrainOptions
#from epdn.OmniDepth_network import ConELUBlock
###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=4, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, n_width=256, n_height=128, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'omnilocal':        
        netG = OmniLocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, n_width, n_height, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            #print('input', np.array(input).shape)
            loss = 0
            for input_i in input:
                #print('input device', np.array(input_i).device)
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss


##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=4, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.dehaze=Dehaze()
        self.dehaze2=Dehaze()
    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            #print('prev', output_prev.shape, 'input_i',model_downsample(input_i).shape)  

            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        tmp=torch.cat((output_prev,input), 1)
        dehaze=self.dehaze(tmp)
        tmp=torch.cat((output_prev,dehaze),1)
        dehaze=self.dehaze2(tmp)
        return output_prev,dehaze

##############################################################################
# Generator
##############################################################################
class OmniLocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=4, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, n_width=256, n_height=128, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(OmniLocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        activation = nn.ReLU(True) 
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        self.extractor_coarse = OmniFeatureExtractor(input_nc, output_nc, ngf, n_width=int(n_width/2), n_height=int(n_height/2), norm_layer=nn.InstanceNorm2d, padding_type='reflect')
        self.extractor_fine = OmniFeatureExtractor(input_nc, output_nc, ngf, n_width=n_width, n_height=n_height, norm_layer=nn.InstanceNorm2d, padding_type='reflect')    

        #model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_coarse = []
        ### downsample
        for i in range(n_downsample_global):
            mult = 2**(i)
            model_coarse += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
            model_coarse += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### resnet blocks
        mult = 2**(n_downsample_global)
        for i in range(n_blocks_global):
            model_coarse += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsample_global):
            mult = 2**(n_downsample_global - i)
            model_coarse += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
            model_coarse += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        #model_coarse += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       #norm_layer(int(ngf * mult / 2)), activation] 

        self.model_coarse = nn.Sequential(*model_coarse)     

        model_fine = []
        ###### local enhancer layers #####
        for i in range(n_downsample_global):
            mult = 2**(i)
            model_fine += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
            model_fine += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### resnet blocks
        mult = 2**(n_downsample_global)
        for i in range(n_blocks_global):
            model_fine += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsample_global-1):
            mult = 2**(n_downsample_global - i)
            model_fine += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]
            model_fine += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.model_fine = nn.Sequential(*model_fine)


        model_final = []
        model_final += [nn.ConvTranspose2d(int(ngf * mult / 2)+int(ngf * mult / 4), ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf), nn.ReLU(True)]  
        
        for i in range(n_blocks_local):
            model_final += [ResnetBlock(ngf * 2, padding_type=padding_type, norm_layer=norm_layer)]

        ### final convolution
        model_final += [nn.ReflectionPad2d(2), nn.Conv2d(ngf*2, ngf, kernel_size=5, padding=0)]
        model_final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]

        self.model_final = nn.Sequential(*model_final)     
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.dehaze=Dehaze()
        self.dehaze2=Dehaze()
    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))
        ### output at coarest level
        feature_fuse_coarse = self.extractor_coarse(input_downsampled[-1])
        output_coarse = self.model_coarse(feature_fuse_coarse)

        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            feature_fuse_fine = self.extractor_fine(input_i)
            output_fine = self.model_fine(feature_fuse_fine)
            output_prev = self.model_final(torch.cat((output_fine, output_coarse), dim=1))
            #print('output_prev', output_prev.shape)

        tmp=torch.cat((output_prev,input), 1)
        dehaze=self.dehaze(tmp)
        tmp=torch.cat((output_prev,dehaze),1)
        dehaze=self.dehaze2(tmp)
        return output_prev,dehaze

class Dehaze(nn.Module):
    def __init__(self):
        super(Dehaze, self).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.tanh=nn.Tanh()

        self.refine1= nn.Conv2d(6, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

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
        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))

        return dehaze

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.InstanceNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             
'''
class OmniFeatureExtractor(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_width=256, n_height=128, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(OmniFeatureExtractor, self).__init__()        
        activation = nn.ReLU(True)

        self.rwsff_0 = HeightWise_SFF_Model(int(ngf/2),height=n_height,reduction=4,bias=False,norm_layer=norm_layer)
        self.rwsff_1 = HeightWise_SFF_Model(ngf,height=n_height,reduction=4,bias=False,norm_layer=norm_layer)

        self.extractor_0_0 = ConELUBlock(input_nc, int(ngf/2), (3, 9), padding=(1, 4))
        self.extractor_0_1 = ConELUBlock(input_nc, int(ngf/2), (5, 11), padding=(2, 5))
        self.extractor_0_2 = ConELUBlock(input_nc, int(ngf/2), (5, 7), padding=(2, 3))
        self.extractor_0_3 = ConELUBlock(input_nc, int(ngf/2), 7, padding=3)

        self.extractor_1_0 = ConELUBlock(int(ngf/2), ngf, (3, 9), padding=(1, 4))
        self.extractor_1_1 = ConELUBlock(int(ngf/2), ngf, (3, 7), padding=(1, 3))
        self.extractor_1_2 = ConELUBlock(int(ngf/2), ngf, (3, 5), padding=(1, 2))
        self.extractor_1_3 = ConELUBlock(int(ngf/2), ngf, 5, padding=2) 

            
    def forward(self, input):
        feature_0_0 = self.extractor_0_0(input)
        feature_0_1 = self.extractor_0_1(input)
        feature_0_2 = self.extractor_0_2(input)
        feature_0_3 = self.extractor_0_3(input)
        #feature_0_cat = torch.cat((feature_0_0, feature_0_1, feature_0_2, feature_0_3), 1)

        feature_fuse_0, sh0 = self.rwsff_0(feature_0_0, feature_0_1, feature_0_2, feature_0_3)

        # Second filter bank
        feature_1_0 = self.extractor_1_0(feature_fuse_0)
        feature_1_1 = self.extractor_1_1(feature_fuse_0)
        feature_1_2 = self.extractor_1_2(feature_fuse_0)
        feature_1_3 = self.extractor_1_3(feature_fuse_0)
        #feature_1_cat = torch.cat((feature_1_0, feature_1_1, feature_1_2, feature_1_3), 1)

        feature_fuse_1, sh1 = self.rwsff_1(feature_1_0, feature_1_1, feature_1_2, feature_1_3)

        return feature_fuse_1

class HeightWise_SFF_Model(nn.Module):
    def __init__(self, input_channel, height=128, reduction=4, bias=False, norm_layer=nn.InstanceNorm2d,):
        super(HeightWise_SFF_Model, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        d = max(int(height/reduction),4)
        self.conv_squeeze = nn.Sequential(nn.Conv2d(height, d, 1, padding=0, bias=bias), nn.PReLU())
        self.fcs_f0 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f1 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f2 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f3 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)

        self.sigmoid = nn.Softmax(dim=2)

        self.conv_smooth = ConELUBlock(input_channel, input_channel, (5, 3), padding=(2, 1))

    def forward(self, input0, input1, input2, input3):

        input0_trans = torch.transpose(input0, 1, 2)
        input1_trans = torch.transpose(input1, 1, 2)
        input2_trans = torch.transpose(input2, 1, 2)
        input3_trans = torch.transpose(input3, 1, 2)

        feature_fuse_1 = input0_trans+input1_trans+input2_trans+input3_trans
        #print('feature_fuse_1',feature_fuse_1.shape)

        pooling = self.global_avg_pool(feature_fuse_1)
        #print('pooling',pooling.shape)
        squeeze = self.conv_squeeze(pooling)
        #print('squeeze',squeeze.shape)

        score_f0 = self.fcs_f0(squeeze)
        score_f1 = self.fcs_f1(squeeze)
        score_f2 = self.fcs_f2(squeeze)
        score_f3 = self.fcs_f3(squeeze)
        #print('score_f0',score_f0.shape)

        score_cat = torch.cat((score_f0, score_f1, score_f2, score_f3),2)
        #print('score_cat',score_cat.shape)
        score_att = self.sigmoid(score_cat)
        

        #print('score_att',score_att.shape)
        score_chunk = torch.chunk(score_att, 4, 2)

        output_f0 = score_chunk[0] * input0_trans
        output_f1 = score_chunk[1] * input1_trans
        output_f2 = score_chunk[2] * input2_trans
        output_f3 = score_chunk[3] * input3_trans
        #print('output_f0',output_f0.shape)

        output = torch.transpose(output_f0+output_f1+output_f2+output_f3 + feature_fuse_1,1,2)
        #print('output',output.shape)
        output = self.conv_smooth(output)

        return output, score_att

'''

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
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
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

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




class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()        
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            indices = (inst == i).nonzero() # n x 4            
            for j in range(self.output_nc):
                output_ins = outputs[indices[:,0], indices[:,1] + j, indices[:,2], indices[:,3]]                    
                mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                outputs_mean[indices[:,0], indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                        
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
