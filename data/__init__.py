import sys

import torch.utils.data
from data.base_data_loader import BaseDataLoader

sys.path.append('D:/dzhao/dehazing_360/SSDH-HDR/')



def CreateDataLoader(opt):
	data_loader = CustomDatasetDataLoader()
	print(data_loader.name())
	data_loader.initialize(opt)
	return data_loader


def CreateDataset(opt):
	dataset = None
	if opt.dataset_mode == 'dec_vit':
		from data.dec_vit_data import DECVITDATA
		dataset = DECVITDATA()

	elif opt.dataset_mode == 'vit': 
		from data.vit_data import VITDATA
		dataset = VITDATA()



	else:
		raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
	
	print("dataset [%s] was created" % (dataset.name()))
	dataset.initialize(opt)
	return dataset


class CustomDatasetDataLoader(BaseDataLoader):
	def name(self):
		return 'CustomDatasetDataLoader'
	
	def initialize(self, opt):
		BaseDataLoader.initialize(self, opt)
		self.dataset = CreateDataset(opt)
		self.dataloader = torch.utils.data.DataLoader(
			self.dataset,
			batch_size=opt.batchSize,
			shuffle=not opt.sb,
			num_workers=int(opt.nThreads))
	
	def load_data(self):
		return self
	
	def __len__(self):
		return min(len(self.dataset), self.opt.max_dataset_size)
	
	def __iter__(self):
		for i, data in enumerate(self.dataloader):
			if i * self.opt.batchSize >= self.opt.max_dataset_size:
				break
			yield data
