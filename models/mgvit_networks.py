import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
from einops import rearrange


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
	init_weights(net, init_type)
	return net


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

#############################
#############################
## Selections of the Model ##
#############################
#############################
def define_G_E_A(input_nc, output_nc, ngf, n_blocks, which_model_netG, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'fine':
		netG = FineGrainedViT_UNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif which_model_netG == 'coarse':
		netG = CoarseGrainedViT_UNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
	elif which_model_netG == 'multi_add':
		netG = MultiGrainedViT_UNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'multi_mul':
		netG = NormalViT_UNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'multi_cat':
		netG = HDRUNetEncoderMSClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'normal':
		netG = HDRUNetEncoderCCAClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)




#######################################

def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
	netD = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netD == 'basic':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'nl':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'fd':
		netD = NLayerFeatureDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'pixel':
		netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	else:
		raise NotImplementedError('Discriminator model name [%s] is not recognized' %
		                          which_model_netD)
	return init_net(netD, init_type, gpu_ids)


###########################################
###########################################
##              Sub Functions            ##
###########################################
###########################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        #self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        #self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        #x = self.proj_drop(x)
        

        return x


class ViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(ViTBlock).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, H, W):
        #x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        #x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + (self.attn(self.norm1(x), H, W))
        x = x + (self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=64, input_nc=3, embed_dim=768):
        super(PatchEmbed).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(input_nc, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)



def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


########################################################################

##############################################
##############################################
## Hierarchical Disentangled Representation ##
##       Haze Encoder for Clear Image       ##
##############################################
##############################################


########################################################################
## Encoder for Clear Image
########################################################################
class MGViTEncoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, img_size=256, patch_size=16, nh=1, num_stages=6, mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1], norm_layer_conv=nn.InstanceNorm2d, norm_layer_vit=nn.LayerNorm, qkv_bias=False, qk_scale=None, use_dropout=False, n_blocks=3, padding_type='reflect',):
		assert (n_blocks >= 0)
		super(HDRUNetEncoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.nh = nh
		embed_dims = self.ngf * np.array([1,2,4,8,16,16])
		num_heads =self.nh * np.array([1,2,4,8,16,16]),
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     input_nc=input_nc if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            #pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([ViTBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            #setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            trunc_normal_(pos_embed, std=.02)

        # init weights
        self.apply(self._init_weights)


		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

	def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)
            
    def vit(self, x, i):
        outs = []

        B = x.shape[0]

        patch_embed = getattr(self, f"patch_embed{i + 1}")
        pos_embed = getattr(self, f"pos_embed{i + 1}")
        #pos_drop = getattr(self, f"pos_drop{i + 1}")
        block = getattr(self, f"block{i + 1}")
        x, (H, W) = patch_embed(x)
        if i == self.num_stages - 1:
            pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
        else:
            pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

        x = (x + pos_embed) #pos_drop(x + pos_embed)
        for blk in block:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return outs

    def forward(self, x):
    	i = 0
    	x = self.vit(x, i)

        x = self.forward_features(x)
        return x

	def forward(self, input):
		ecfe = self.encode_contfe(input)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		

		return [ecfe, ec1, ec2, ec3]

##########################################################
## Decoder for Clear image generation
##########################################################
class HDRUNetDecoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HDRUNetDecoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		n_downsampling = 3
		mult = 2 ** n_downsampling
		decode_ft = []
		for i in range(n_blocks):
			decode_ft += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.decode_ft = nn.Sequential(*decode_ft)

		mult = 2 ** (n_downsampling+1)
		decoder_up3 = [nn.Upsample(scale_factor=2, mode='bilinear'),
						nn.Conv2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		decoder_up3 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]

		self.decoder_up3 = nn.Sequential(*decoder_up3)

		mult = 2 ** (n_downsampling)
		decoder_up2 = [nn.Upsample(scale_factor=2, mode='bilinear'),
						nn.Conv2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		decoder_up2 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]

		self.decoder_up2 = nn.Sequential(*decoder_up2)

		mult = 2 ** (n_downsampling-1)
		decoder_up1 = [nn.Upsample(scale_factor=2, mode='bilinear'),
						nn.Conv2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		decoder_up1 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]

		self.decoder_up1 = nn.Sequential(*decoder_up1)

		mult = 2 ** (n_downsampling-2)
		decoder_fn = [nn.Conv2d(ngf * mult + int(ngf/2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]  #+ int(ngf/2)
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#
		skip_fn = [nn.Conv2d(3, int(ngf/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		self.skip_fn = nn.Sequential(*skip_fn)
		#skip_fe = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fe = nn.Sequential(*skip_fe)
		#skip_1 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 2), nn.ReLU(True)]
		#self.skip_1 = nn.Sequential(*skip_1)
		#skip_2 = [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 4), nn.ReLU(True)]
		#self.skip_2 = nn.Sequential(*skip_2)
		#skip_3 = [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 8), nn.ReLU(True)]
		#self.skip_3 = nn.Sequential(*skip_3)

		#self.dehaze = Dehaze()

	
	def forward(self, input, c):
		dft = self.decode_ft(c[3])
		concat = torch.cat((c[3], dft), 1) # self.skip_3(c[3])
		dup3 = self.decoder_up3(concat)

		concat = torch.cat((c[2], dup3), 1)  # self.skip_2(c[2])
		dup2 = self.decoder_up2(concat)

		concat = torch.cat((c[1], dup2), 1)  # self.skip_1(c[1])
		dup1 = self.decoder_up1(concat)

		concat = torch.cat((c[0], self.skip_fn(input), dup1), 1) # self.skip_fe(c[0]), self.skip_fn(input)
		out = self.decoder_fn(concat)
#
		return out#, dehaze

# 
# Define a resnet block
class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
	
	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
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
		
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
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
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
		               norm_layer(dim)]
		
		return nn.Sequential(*conv_block)
	
	def forward(self, x):
		out = x + self.conv_block(x)
		return out

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
        self.conv1050 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        #refine3 = [nn.ReflectionPad2d(3)]
        refine3 = [nn.Conv2d(20+5, 3, kernel_size=3, stride=1, padding=1)]
        refine3 += [nn.Tanh()]
        self.refine3 = nn.Sequential(*refine3)
        self.upsample = F.upsample_bilinear

        self.batch20 = nn.InstanceNorm2d(20, affine=True)
        self.batch1 = nn.InstanceNorm2d(1, affine=True)

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


class ChannelAttention(nn.Module):
    def __init__(self, in_nc, bk_nc):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_nc, bk_nc, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(bk_nc, in_nc, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = (avg_out + max_out)
        return self.sigmoid(out)#*x + x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 5), 'kernel size must be 3 or 5'
        padding = 2 if kernel_size == 5 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        avg_out = torch.mean(input, dim=1, keepdim=True)
        max_out, _ = torch.max(input, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        out = self.sigmoid(x)*input + input
        return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerFeatureDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
		super(NLayerFeatureDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		kw = 4
		padw = 1
		rf_e = [nn.ReflectionPad2d(3),
			nn.Conv2d(input_nc, ndf, kernel_size=7, stride=1),
			nn.LeakyReLU(0.2, True)]

		rf_e += [ nn.Conv2d(ndf, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * 2),
				nn.LeakyReLU(0.2, True)
			]

		#rf_2 = [ nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
		#		norm_layer(ndf * 4),
		#		nn.LeakyReLU(0.2, True)
		#	]

		#rf_3 = [ nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
		#		norm_layer(ndf * 8),
		#		nn.LeakyReLU(0.2, True)
		#	]
		
		self.rf_e = nn.Sequential(*rf_e)
		#self.rf_1 = nn.Sequential(*rf_1)
		#self.rf_2 = nn.Sequential(*rf_2)
		#self.rf_3 = nn.Sequential(*rf_3)

		ff_e = [nn.ReflectionPad2d(3),
			nn.Conv2d(input_nc, ndf, kernel_size=7, stride=1),
			nn.LeakyReLU(0.2, True)]

		ff_e += [ nn.Conv2d(ndf, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * 2),
				nn.LeakyReLU(0.2, True)
			]

		#ff_2 = [ nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
		#		norm_layer(ndf * 4),
		#		nn.LeakyReLU(0.2, True)
		#	]

		#ff_3 = [ nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
		#		norm_layer(ndf * 8),
		#		nn.LeakyReLU(0.2, True)
		#	]
		
		self.ff_e = nn.Sequential(*ff_e)
		#self.ff_1 = nn.Sequential(*ff_1)
		#self.ff_2 = nn.Sequential(*ff_2)
		#self.ff_3 = nn.Sequential(*ff_3)

		fuse1 = [nn.Conv2d(ndf * 4, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
				norm_layer(ndf * 2),
				nn.LeakyReLU(0.2, True)
				]
		self.fuse1 = nn.Sequential(*fuse1)

		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
		reduction = 16
		d = max(int(ndf * 2/reduction),4)
		self.conv_squeeze = nn.Sequential(nn.Conv2d(ndf * 2, d, 1, padding=0, bias=use_bias), nn.PReLU())
		self.fcs_f0 = nn.Conv2d(d, ndf * 2, kernel_size=1, stride=1,bias=use_bias)
		self.fcs_f1 = nn.Conv2d(d, ndf * 2, kernel_size=1, stride=1,bias=use_bias)
		self.softmax = nn.Softmax(dim=2)

		fuse2 = [
				nn.Conv2d(ndf * 2 , ndf * 2,
				          kernel_size=3, stride=1, padding=padw, bias=use_bias),
				norm_layer(ndf * 2),
				nn.LeakyReLU(0.2, True)
			]
		self.fuse2 = nn.Sequential(*fuse2)

		cross_f_1 = [
				nn.Conv2d(ndf * 2 , ndf * 4,
				          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * 4),
				nn.LeakyReLU(0.2, True)
			]

		cross_f_2 = [
				nn.Conv2d(ndf * 4, ndf * 8, 
				          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * 8),
				nn.LeakyReLU(0.2, True)
			]

		cross_f_3 = [
				nn.Conv2d(ndf * 8, ndf * 8,
				          kernel_size=3, stride=1, padding=1, bias=use_bias),
				norm_layer(ndf * 8),
				nn.LeakyReLU(0.2, True)
			]

		#cross_f_5 = [
		#		nn.Conv2d(ndf * 8, ndf * 8,
		#		          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
		#		norm_layer(ndf * 8),
		#		nn.LeakyReLU(0.2, True)
		#	]

		#cross_att = [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=(4,8), stride=1, padding=0)]

		self.cross_f_1 = nn.Sequential(*cross_f_1)
		self.cross_f_2 = nn.Sequential(*cross_f_2)
		self.cross_f_3 = nn.Sequential(*cross_f_3)
		#self.cross_f_4 = nn.Sequential(*cross_f_4)
		#self.cross_f_5 = nn.Sequential(*cross_f_5)
		#self.cross_att = nn.Sequential(*cross_att)

		class_3 = [nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1)]
		#class_4 = [nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=padw)]
		#class_5 = [nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=padw)]

		self.class_3 = nn.Sequential(*class_3)
		#self.class_4 = nn.Sequential(*class_4)
		#self.class_5 = nn.Sequential(*class_5)

		
		sigm = [nn.Sigmoid()]
		
		self.sigm = nn.Sequential(*sigm)


	def forward(self, real, fake):
		real_fe = self.rf_e(real)
		#real_f1 = self.rf_1(real_fe)
		#real_f2 = self.rf_2(real_f1)
		#real_f3 = self.rf_3(real_f2)

		fake_fe = self.ff_e(fake)
		#fake_f1 = self.ff_1(fake_fe)
		#fake_f2 = self.ff_2(fake_f1)
		#fake_f3 = self.ff_3(fake_f2)

		rf_fuse = self.fuse1(torch.cat((real_fe, fake_fe),1))
		pooling = self.global_avg_pool(rf_fuse)
		squeeze = self.conv_squeeze(pooling)
		score_f0 = self.fcs_f0(squeeze)
		score_f1 = self.fcs_f1(squeeze)
		score_cat = torch.cat((score_f0, score_f1),2)
		score_att = self.softmax(score_cat)
		score_chunk = torch.chunk(score_att, 4, 2)
		real_fe = score_chunk[0] * real_fe
		fake_fe = score_chunk[1] * fake_fe
		rf_fuse = self.fuse2(real_fe + fake_fe)

		cf1 = self.cross_f_1(rf_fuse)
		cf2 = self.cross_f_2(cf1)
		cf3 = self.cross_f_3(cf2)
		#cf4 = self.cross_f_4(cf3)
		#cf5 = self.cross_f_5(cf4)

		#cca = self.sigm(self.cross_att(cf3))

		#print('cf3', cf3.shape, 'cf4', cf4.shape, 'cca', cca.shape)

		cls3 = self.class_3(cf3)
		#cls4 = self.class_4(cca * cf4 + cf4)
		#cls5 = self.class_4(cca * cf5 + cf5)
		#print(cls3.shape, cls4.shape,cls5.shape,)

		out = self.sigm(cls3)
		#print(out.shape)
		return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
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
	def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
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
	def __init__(self, input_nc, ndf, norm_layer=nn.BatchNorm2d):
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
