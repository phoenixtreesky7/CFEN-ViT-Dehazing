import argparse
import os

import torch
from util import util


class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False
	
	def initialize(self):
		self.parser.add_argument('--dataroot', required=True, default='D:/dzhao/dehazing_360/360hazy_dataset/madan/reside',help='path to images (should have subfolders trainA, trainB, etc)')
		self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
		self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
		self.parser.add_argument('--fineSize', type=int, default=128, help='then crop to this size')
		self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
		self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
		self.parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
		self.parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
		self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
		self.parser.add_argument('--model_G', type=str, default='iid_hlgvit_crs_gd4', help='selects model to use for netG')
		self.parser.add_argument('--ca_type', type=str, default='cross_ca', help='disentangling the haze representations using which channel attention model: cross channel attenion [cross_ca]; channal attention for each level [level_ca]; or no channel attention [none_ca]')
		self.parser.add_argument('--fuse_model', type=str, default='cat', help='selects model to fuse the content and hazy style features, cat|csfm')
		self.parser.add_argument('--hl', type=int, default=3, help=' layers of the Hierarchy DR')
		self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
		self.parser.add_argument('--unet_layer', type=int, default=3, help='only used if which_model_netD==n_layers')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
		self.parser.add_argument('--dataset_mode', type=str, default='vit', help='chooses how datasets are loaded:dh_hdr_sspv, dh_hdr_uspv.')
		self.parser.add_argument('--model', type=str, default='vit', help='chooses which model to use. hdru, cycle_gan, pix2pix, test')
		
		self.parser.add_argument('--max_epoch', default=300, type=int)
		self.parser.add_argument('--current_epoch', default=0, type=int)
		self.parser.add_argument('--weights_init', type=str)
		self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
		self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
		self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
		self.parser.add_argument('--sb', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
		self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
		self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
		self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
		self.parser.add_argument('--display_port', type=int, default=3000, help='visdom port of the web display')
		self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
		self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, ' 'only a subset is loaded.')
		self.parser.add_argument('--resize_or_crop', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
		self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
		self.parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
		self.parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
		self.parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
		self.parser.add_argument('--out_all', action='store_true', help='output all stylized images(fake_B_{})')
		self.parser.add_argument('--dehazing_netG', type=str, default='local', help='selects model to use for netG')
		self.parser.add_argument('--epdn_ngf', type=int, default=32, help='# of gen filters in first conv layer')
		self.parser.add_argument('--n_downsample_global', type=int, default=2, help='number of downsampling layers in netG')
		self.parser.add_argument('--n_blocks', type=int, default=2, help='number of residual blocks in the global generator network')
		
		self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
		self.parser.add_argument('--imagepool', action='store_true', help='use the image pool')

		# opt of the pretarined model
		
		

		self.parser.add_argument('--debug', action='store_true',
		                    help='Enables debug mode')
		self.parser.add_argument('--template', default='.',
		                    help='You can set various templates in option.py')

# Hardware specifications
		self.parser.add_argument('--n_threads', type=int, default=6,
		                    help='number of threads for data loading')
		self.parser.add_argument('--cpu', action='store_true',
		                    help='use cpu only')
		self.parser.add_argument('--n_GPUs', type=int, default=1,
		                    help='number of GPUs')
		self.parser.add_argument('--seed', type=int, default=1,
		                    help='random seed')

# Data specifications
		self.parser.add_argument('--dir_data', type=str, default='D:/dzhao/dehazing_360/360hazy_dataset/',
		                    help='dataset directory')
		self.parser.add_argument('--dir_demo', type=str, default='../test',
		                    help='demo image directory')
		self.parser.add_argument('--data_train', type=str, default='madan/reside/clear/resize/',
		                    help='train dataset name')
		self.parser.add_argument('--data_test', type=str, default='ssdh/test/reside_ohaze_nahaze/hazy/resize',
		                    help='test dataset name')
		#self.parser.add_argument('--data_range', type=str, default='1-800/801-810',
		#                    help='train/test data range')
		self.parser.add_argument('--ext', type=str, default='sep',
		                    help='dataset file extension')
		self.parser.add_argument('--scale', type=str, default='1',
		                    help='super resolution scale')
		self.parser.add_argument('--patch_size', type=int, default=32,
		                    help='output patch size')
		self.parser.add_argument('--rgb_range', type=int, default=255,
		                    help='maximum value of RGB')
		self.parser.add_argument('--n_colors', type=int, default=3,
		                    help='number of color channels to use')
		self.parser.add_argument('--no_augment', action='store_true',
		                    help='do not use data augmentation')
		self.parser.add_argument('--hidden_dim_ratio', type=int, default=6,
		                    help='hidden_dim extension ratio in MLP')
		self.parser.add_argument('--l2g_ratio', type=int, default=4,
		                    help='downsampling ratio from local to global')

		# Model specifications
		self.parser.add_argument('--n_feats', type=int, default=32,
		                    help='number of feature maps')
		self.parser.add_argument('--shift_mean', default=True,
		                    help='subtract pixel mean from the input')
		self.parser.add_argument('--precision', type=str, default='single',
		                    choices=('single', 'half'),
		                    help='FP precision for test (single | half)')

		# Training specifications
		self.parser.add_argument('--reset', action='store_true',
		                    help='reset the training')
		self.parser.add_argument('--test_every', type=int, default=1000,
		                    help='do test per every N batches')
		self.parser.add_argument('--epochs', type=int, default=300,
		                    help='number of epochs to train')
		self.parser.add_argument('--batch_size', type=int, default=1,
		                    help='input batch size for training')
		self.parser.add_argument('--test_batch_size', type=int, default=1,
		                    help='input batch size for training')
		self.parser.add_argument('--crop_batch_size', type=int, default=64,
		                    help='input batch size for training')
		self.parser.add_argument('--split_batch', type=int, default=1,
		                    help='split the batch into smaller chunks')
		self.parser.add_argument('--self_ensemble', action='store_true',
		                    help='use self-ensemble method for test')
		self.parser.add_argument('--test_only', action='store_true',
		                    help='set this option to test the model')
		self.parser.add_argument('--gan_k', type=int, default=1,
		                    help='k value for adversarial loss')

		# Optimization specifications
		self.parser.add_argument('--lr', type=float, default=1e-4,
		                    help='learning rate')
		self.parser.add_argument('--decay', type=str, default='200',
		                    help='learning rate decay type')
		self.parser.add_argument('--gamma', type=float, default=0.5,
		                    help='learning rate decay factor for step decay')
		self.parser.add_argument('--optimizer', default='ADAM',
		                    choices=('SGD', 'ADAM', 'RMSprop'),
		                    help='optimizer to use (SGD | ADAM | RMSprop)')
		self.parser.add_argument('--momentum', type=float, default=0.9,
		                    help='SGD momentum')
		self.parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
		                    help='ADAM beta')
		self.parser.add_argument('--epsilon', type=float, default=1e-8,
		                    help='ADAM epsilon for numerical stability')
		self.parser.add_argument('--weight_decay', type=float, default=0,
		                    help='weight decay')
		self.parser.add_argument('--gclip', type=float, default=0,
		                    help='gradient clipping threshold (0 = no clipping)')

		# Loss specifications
		self.parser.add_argument('--loss', type=str, default='1*L1',
		                    help='loss function configuration')
		self.parser.add_argument('--skip_threshold', type=float, default='1e8',
		                    help='skipping batch that has large error')

		# Log specifications
		self.parser.add_argument('--save', type=str, default='/cache/results/ipt/',
		                    help='file name to save')
		self.parser.add_argument('--load', type=str, default='',
		                    help='file name to load')
		self.parser.add_argument('--resume', type=int, default=0,
		                    help='resume from specific checkpoint')
		self.parser.add_argument('--save_models', action='store_true',
		                    help='save all intermediate models')
		self.parser.add_argument('--print_every', type=int, default=100,
		                    help='how many batches to wait before logging training status')
		self.parser.add_argument('--save_results', action='store_true',
		                    help='save output results')
		self.parser.add_argument('--save_gt', action='store_true',
		                    help='save low-resolution and high-resolution images together')

		#cloud
		self.parser.add_argument('--moxfile', type=int, default=1)
		self.parser.add_argument('--data_url', type=str,help='path to dataset')
		self.parser.add_argument('--train_url', type=str, help='train_dir')
		self.parser.add_argument('--pretrain', type=str, default='')
		self.parser.add_argument('--load_query', type=int, default=0)

		#transformer
		self.parser.add_argument('--patch_dim', type=int, default=2)
		self.parser.add_argument('--num_heads', type=int, default=4)
		self.parser.add_argument('--num_layers', type=int, default=1)
		self.parser.add_argument('--dropout_rate', type=float, default=0)
		self.parser.add_argument('--no_norm', action='store_true')
		self.parser.add_argument('--freeze_norm', action='store_true')
		self.parser.add_argument('--post_norm', action='store_true')
		self.parser.add_argument('--no_mlp', action='store_true')
		self.parser.add_argument('--pos_every', action='store_true')
		self.parser.add_argument('--no_pos', action='store_true')
		self.parser.add_argument('--num_queries', type=int, default=1)

		#denoise
		#self.parser.add_argument('--denoise', action='store_true')
		#self.parser.add_argument('--sigma', type=float, default=30)

		#dehazing
		self.parser.add_argument('--dehazing', action='store_true')
		self.parser.add_argument('--dehazing_test', type=int, default=1)

		self.initialized = True
	
	def parse(self):
		if not self.initialized:
			self.initialize()
		opt = self.parser.parse_args()
		opt.isTrain = self.isTrain  # train or test
		
		str_ids = opt.gpu_ids.split(',')
		opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				opt.gpu_ids.append(id)
		
		# set gpu ids
		if len(opt.gpu_ids) > 0:
			torch.cuda.set_device(opt.gpu_ids[0])
		
		args = vars(opt)
		
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')
		
		if opt.suffix:
			suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
			opt.name = opt.name + suffix
		# save to the disk
		expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		self.opt = opt
		return self.opt
