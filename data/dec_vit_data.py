import os.path
import random

import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset, get_transform
#from data.cityscapes import remap_labels_to_train_ids
from data.image_folder import make_cs_labels, make_dataset

# This dataset is used to conduct double cyclegan for both GTAV->CityScapes and Synthia->CityScapes
class DECVITDATA(BaseDataset):
	def initialize(self, opt):
		# OHAZE as dataset 1
		# 3D60 as dataset 2
		self.opt = opt
		self.root = opt.dataroot
		self.isTrain = opt.isTrain
		self.dir_B = os.path.join(opt.dataroot, 'hazy')
		if self.isTrain:
			self.dir_A = os.path.join(opt.dataroot, 'clear')
			self.dir_R = os.path.join(opt.dataroot, 'r')
			self.dir_S = os.path.join(opt.dataroot, 's')
		#self.dir_C = os.path.join(opt.dataroot, 'real')

		self.B_paths = make_dataset(self.dir_B)
		if self.isTrain:
			self.A_paths = make_dataset(self.dir_A)
			self.R_paths = make_dataset(self.dir_R)
			self.S_paths = make_dataset(self.dir_S)
		#self.C_paths = make_dataset(self.dir_C)

		self.B_paths = sorted(self.B_paths)
		if self.isTrain:
			self.A_paths = sorted(self.A_paths)
			self.R_paths = sorted(self.R_paths)
			self.S_paths = sorted(self.S_paths)
		#self.C_paths = sorted(self.C_paths)

		self.B_size = len(self.B_paths)
		if self.isTrain:
			self.A_size = len(self.A_paths)
			self.R_size = len(self.R_paths)
			self.S_size = len(self.S_paths)
		#self.C_size = len(self.C_paths)

		self.transform = get_transform(opt)

	
	def __getitem__(self, index):
		
		if self.opt.sb:
			B_path = self.B_paths[index % self.B_size]
			if self.isTrain:
				A_path = self.A_paths[index % self.A_size]
				R_path = self.R_paths[index % self.R_size]
				S_path = self.S_paths[index % self.S_size]
		else:
			self.index_rand = random.randint(0, self.B_size - 1)
			B_path = self.B_paths[self.index_rand]
			if self.isTrain:
				A_path = self.A_paths[self.index_rand]
				
				R_path = self.R_paths[self.index_rand]
				S_path = self.S_paths[self.index_rand]

		
		#B_path = self.B_paths[index_B]
		#if self.isTrain:
			#index_C = random.randint(0, self.C_size - 1)
			#C_path = self.C_paths[index_C]
		#else:
			#C_path = self.C_paths[index % self.C_size]

		

		B_img = Image.open(B_path).convert('RGB')

		if self.isTrain:
			A_img = Image.open(A_path).convert('RGB')
			R_img = Image.open(R_path).convert('RGB')
			S_img = Image.open(S_path).convert('RGB')
		#C_img = Image.open(C_path).convert('RGB')
		B = self.transform(B_img)
		if self.isTrain:
			A = self.transform(A_img)
			R = self.transform(R_img)
			S = self.transform(S_img)
		#C = self.transform(C_img)

		if self.opt.which_direction == 'BtoA':
			input_nc = self.opt.output_nc
			output_nc = self.opt.input_nc
		else:
			input_nc = self.opt.input_nc
			output_nc = self.opt.output_nc
		
		


		if output_nc == 1:  # RGB to gray
			tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
			B = tmp.unsqueeze(0)
		if self.isTrain:
			if input_nc == 1:  # RGB to gray
				tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
				A = tmp.unsqueeze(0)
			tmp = S[0, ...] * 0.299 + S[1, ...] * 0.587 + S[2, ...] * 0.114
			S = tmp.unsqueeze(0)
		
		if self.isTrain:
			return {'A': A, 'B': B, 'R': R, 'S': S, 'A_paths': A_path, 'B_paths': B_path}
		else:
			return {'B': B, 'B_paths': B_path}
		

	def __len__(self):
		return self.B_size
	
	def name(self):
		return 'DEC_ViT'
