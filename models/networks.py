import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
#from epdn.OmniDepth_network import ConELUBlock
import numpy as np
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
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
	
	print('initialize network with %s' % init_type)
	net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert (torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)
	print('init_type', init_type)
	init_weights(net, init_type)
	return net


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)
	

def define_D(input_nc, ndf, which_model_netD, n_layers_D=4, norm='instance', use_sigmoid=False, init_type='normal', gpu_ids=[]):
	netD = None
	print('norm', norm)
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netD == 'basic':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'n_layers':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'pixel':
		netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	else:
		raise NotImplementedError('Discriminator model name [%s] is not recognized' %
		                          which_model_netD)
	return init_net(netD, init_type, gpu_ids)


def define_C(output_nc, ndf, init_type='normal', gpu_ids=[]):
	# if output_nc == 3:
	#    netC = get_model('DTN', num_cls=10)
	# else:
	#    Exception('classifier only implemented for 32x32x3 images')
	netC = Classifier(output_nc, ndf)
	return init_net(netC, init_type, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
	def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		if use_lsgan:
			self.loss = nn.MSELoss()
		else:
			self.loss = nn.BCELoss()
	
	def get_target_tensor(self, input, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(input)
	
	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(ResnetGenerator, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		model = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.ReLU(True)]
		
		n_downsampling = 2
		for i in range(n_downsampling):
			mult = 2 ** i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.ReLU(True)]
		
		mult = 2 ** n_downsampling
		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]
		
		for i in range(n_downsampling):
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
			                             kernel_size=3, stride=2,
			                             padding=1, output_padding=1,
			                             bias=use_bias),
			          norm_layer(int(ngf * mult / 2)),
			          nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3)]
		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model += [nn.Tanh()]
		
		self.model = nn.Sequential(*model)
	
	def forward(self, input):
		return self.model(input)


# Define a resnet block
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


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
		super(UnetGenerator, self).__init__()
		
		# construct unet structure
		unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
		for i in range(num_downs - 5):
			unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
		
		self.model = unet_block
	
	def forward(self, input):
		return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		if input_nc is None:
			input_nc = outer_nc
		downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm_layer(inner_nc)
		uprelu = nn.ReLU(True)
		upnorm = norm_layer(outer_nc)
		
		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]
			
			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up
		
		self.model = nn.Sequential(*model)
	
	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:
			return torch.cat([x, self.model(x)], 1)

'''
class OmniUNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_blocks_global=3, 
                 n_width=256, n_height=128, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(OmniUNet, self).__init__()
        self.n_width = n_width
        self.n_height = n_height
        activation = nn.ReLU(True)
        self.rwsff_0 = HeightWise_SFF_Model(int(ngf/2),height=n_height,reduction=4,bias=False,norm_layer=norm_layer)
        self.rwsff_1 = HeightWise_SFF_Model(ngf,height=n_height,reduction=4,bias=False,norm_layer=norm_layer)
        
        ###### DehazeGenerator #####
        ### feature extractor

        self.extractor_0_0 = ConELUBlock(input_nc, int(ngf/2), (3, 9), padding=(1, 4))
        self.extractor_0_1 = ConELUBlock(input_nc, int(ngf/2), (5, 11), padding=(2, 5))
        self.extractor_0_2 = ConELUBlock(input_nc, int(ngf/2), (5, 7), padding=(2, 3))
        self.extractor_0_3 = ConELUBlock(input_nc, int(ngf/2), 7, padding=3)

        self.extractor_1_0 = ConELUBlock(int(ngf/2), ngf, (3, 9), padding=(1, 4))
        self.extractor_1_1 = ConELUBlock(int(ngf/2), ngf, (3, 7), padding=(1, 3))
        self.extractor_1_2 = ConELUBlock(int(ngf/2), ngf, (3, 5), padding=(1, 2))
        self.extractor_1_3 = ConELUBlock(int(ngf/2), ngf, 5, padding=2)

        mult = 1
        g_dehaze_e0 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e0 = nn.Sequential(*g_dehaze_e0)

        mult = 2
        g_dehaze_e1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e1 = nn.Sequential(*g_dehaze_e1)

        mult = 4
        g_dehaze_e2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e2 = nn.Sequential(*g_dehaze_e2)

        ### resnet blocks
        g_dehaze_t = []
        mult = 2**(3)
        for i in range(n_blocks_global):
            g_dehaze_t += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            self.g_dehaze_t = nn.Sequential(*g_dehaze_t)
        
        ### decoder         
        mult = 8        
        g_dehaze_d2 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d2 = nn.Sequential(*g_dehaze_d2)

        mult = 4        
        g_dehaze_d1 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d1 = nn.Sequential(*g_dehaze_d1)

        mult = 2        
        g_dehaze_d0 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d0 = nn.Sequential(*g_dehaze_d0)


        final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]

        self.final = nn.Sequential(*final)           
        
        self.downsample = nn.AvgPool2d(4, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        # First filter bank
        feature_0_0 = self.extractor_0_0(input)
        feature_0_1 = self.extractor_0_1(input)
        feature_0_2 = self.extractor_0_2(input)
        feature_0_3 = self.extractor_0_3(input)
        #feature_0_cat = torch.cat((feature_0_0, feature_0_1, feature_0_2, feature_0_3), 1)
        #print('input',input.shape)

        feature_fuse_0, sh0 = self.rwsff_0(feature_0_0,feature_0_1,feature_0_2,feature_0_3)

        score_att_np = sh0.cpu().detach().numpy()
        score_att_np_squeeze = np.squeeze(score_att_np)
        #x = random.randint(100000)
        #np.savetxt('score_att_h/score_att_sh0_'+ str(x) +'.txt', score_att_np_squeeze)

        # Second filter bank
        feature_1_0 = self.extractor_1_0(feature_fuse_0)
        feature_1_1 = self.extractor_1_1(feature_fuse_0)
        feature_1_2 = self.extractor_1_2(feature_fuse_0)
        feature_1_3 = self.extractor_1_3(feature_fuse_0)
        #feature_1_cat = torch.cat((feature_1_0, feature_1_1, feature_1_2, feature_1_3), 1)

        feature_fuse_1,sh1 = self.rwsff_1(feature_1_0,feature_1_1,feature_1_2,feature_1_3)

        #score_att_np = sh1.cpu().detach().numpy()
        #score_att_np_squeeze = np.squeeze(score_att_np)
        #x = random.randint(100000)
        #np.savetxt('score_att_h/score_att_sh1_'+ str(x) +'.txt', score_att_np_squeeze)

        
        # encoder

        encode_out1 = self.g_dehaze_e0(feature_fuse_1)
        encode_out2 = self.g_dehaze_e1(encode_out1)
        encode_out3 = self.g_dehaze_e2(encode_out2)
        #print('dehaze_d3',dehaze_d3.shape)

        encode_t = self.g_dehaze_t(encode_out3)
        #print('t',dehaze_t.shape)

        # decoder
        tmp = torch.cat((encode_t, encode_out3), 1)
        decoder_out3 = self.g_dehaze_d2(tmp)

        tmp = torch.cat((decoder_out3, encode_out2), 1)
        decoder_out2 = self.g_dehaze_d1(tmp)

        tmp = torch.cat((decoder_out2, encode_out1), 1)
        decoder_out1 = self.g_dehaze_d0(tmp)

        final_out = self.final(decoder_out1)

        return final_out

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




# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
		super(NLayerDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		kw = 4
		padw = 1
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
		]
		
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
				          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]
		
		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
			          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]
		
		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
		
		if use_sigmoid:
			sequence += [nn.Sigmoid()]
		
		self.model = nn.Sequential(*sequence)
	
	def forward(self, input):
		return self.model(input)


class PixelDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
		super(PixelDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		self.net = [
			nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
			norm_layer(ndf * 2),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
		
		if use_sigmoid:
			self.net.append(nn.Sigmoid())
		
		self.net = nn.Sequential(*self.net)
	
	def forward(self, input):
		return self.net(input)


class Classifier(nn.Module):
	def __init__(self, input_nc, ndf, norm_layer=nn.InstanceNorm2d):
		super(Classifier, self).__init__()
		
		kw = 3
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2),
			nn.LeakyReLU(0.2, True)
		]
		
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(3):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
				          kernel_size=kw, stride=2),
				norm_layer(ndf * nf_mult, affine=True),
				nn.LeakyReLU(0.2, True)
			]
		self.before_linear = nn.Sequential(*sequence)
		
		sequence = [
			nn.Linear(ndf * nf_mult, 1024),
			nn.Linear(1024, 10)
		]
		
		self.after_linear = nn.Sequential(*sequence)
	
	def forward(self, x):
		bs = x.size(0)
		out = self.after_linear(self.before_linear(x).view(bs, -1))
		return out
#       return nn.functional.log_softmax(out, dim=1)
