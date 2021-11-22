import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
unloader = transforms.ToPILImage()

class BaseDataset(data.Dataset):
	def __init__(self):
		super(BaseDataset, self).__init__()
	
	def name(self):
		return 'BaseDataset'
	
	def initialize(self, opt):
		pass


# TODO: 增加crop的部分
def get_transform(opt):
	transform_list = []
	if opt.resize_or_crop == 'resize_and_crop':
		osize = [int(opt.loadSize), int(opt.loadSize/2)]
		transform_list.append(transforms.Resize(osize, interpolation=Image.BICUBIC))
		transform_list.append(transforms.RandomCrop(opt.fineSize))
	elif opt.resize_or_crop == 'resize_only':
		osize = [int(opt.loadSize), int(opt.loadSize)]
		transform_list.append(transforms.Resize(opt.loadSize, interpolation=Image.BICUBIC))
	elif opt.resize_or_crop == 'crop':
		transform_list.append(transforms.RandomCrop(opt.fineSize))
	elif opt.resize_or_crop == 'scale_width':
		transform_list.append(transforms.Resize(opt.loadSize, interpolation=Image.BICUBIC))
	elif opt.resize_or_crop == 'scale_width_and_crop':
		transform_list.append(transforms.Resize(opt.loadSize, interpolation=Image.BICUBIC))
		transform_list.append(transforms.RandomCrop(opt.fineSize))
	elif opt.resize_or_crop == 'none':
		print('I do noting')
		#osize = [int(opt.loadSize), int(opt.loadSize)]
		#transform_list.append()
	
	#if opt.isTrain and not opt.no_flip:
	#	transform_list.append(transforms.RandomHorizontalFlip())
	
	transform_list += [transforms.ToTensor(),
	                   transforms.Normalize((0.5, 0.5, 0.5),
	                                        (0.5, 0.5, 0.5))]
	return transforms.Compose(transform_list)


def get_depth_transform(opt):
	transform_list = []
	if opt.resize_or_crop == 'resize_and_crop':
		osize = [opt.loadSize, int(opt.loadSize/2)]
		transform_list.append(transforms.Resize(osize, interpolation=Image.BICUBIC))
		transform_list.append(transforms.RandomCrop(opt.fineSize))
	elif opt.resize_or_crop == 'resize_only':
		osize = [opt.loadSize, int(opt.loadSize/2)]
		transform_list.append(transforms.Resize(osize, interpolation=Image.BICUBIC))
	elif opt.resize_or_crop == 'crop':
		transform_list.append(transforms.RandomCrop(opt.fineSize))
	elif opt.resize_or_crop == 'scale_width':
		transform_list.append(transforms.Resize(opt.loadSize, interpolation=Image.BICUBIC))
	elif opt.resize_or_crop == 'scale_width_and_crop':
		transform_list.append(transforms.Resize(opt.loadSize, interpolation=Image.BICUBIC))
		transform_list.append(transforms.RandomCrop(opt.fineSize))
	# transform_list.append(transforms.RandomCrop(opt.fineSize))
	
	if opt.isTrain and not opt.no_flip:
		transform_list.append(transforms.RandomHorizontalFlip())
	
	transform_list += [transforms.ToTensor(),
	                   transforms.Normalize((0.5),
	                                        (0.5))]
	return transforms.Compose(transform_list)


def __scale_width(img, target_width):
	ow, oh = img.size
	if (ow == target_width):
		return img
	w = target_width
	h = int(target_width * oh / ow)
	return img.resize((w, h), Image.BICUBIC)


def to_tensor_raw(im):
	return torch.from_numpy(np.array(im, np.int64, copy=False))

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image