### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
class AlignedDataset(BaseDataset):
	def initialize(self, opt):
		self.opt = opt
		self.root = opt.dataroot    

		### input A (label maps)
		dir_A = '_A' if self.opt.label_nc == 0 else '_label'
		self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
		self.A_paths = sorted(make_dataset(self.dir_A))
		print(len(self.A_paths))

		### input B (real images)
		if opt.isTrain:
			dir_B = '_B' if self.opt.label_nc == 0 else '_img'
			self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
			self.B_paths = sorted(make_dataset(self.dir_B))

		### instance maps
		if not opt.no_instance:
			self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
			self.inst_paths = sorted(make_dataset(self.dir_inst))

		### load precomputed instance-wise encoded features
		if opt.load_features:                              
			self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
			print('----------- loading features from %s ----------' % self.dir_feat)
			self.feat_paths = sorted(make_dataset(self.dir_feat))

		self.dataset_size = len(self.A_paths) 
	  
	def work(self, index):        
		### input A (label maps)
		A_path = self.A_paths[index]              
		A_tensor = torch.load(A_path).permute((2,0,1))
		# A = Image.open(A_path)        
		# params = get_params(self.opt, A.size)
		# if self.opt.label_nc == 0:
		#     transform_A = get_transform(self.opt, params)
		#     A_tensor = transform_A(A.convert('RGB'))
		# else:
		#     transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
		#     A_tensor = transform_A(A) * 255.0

		B_tensor = inst_tensor = feat_tensor = 0
		### input B (real images)
		if self.opt.isTrain:
			B_path = self.B_paths[index]   
			B = Image.open(B_path).convert('RGB')
			# transform_B = get_transform(self.opt, params)
			# B_tensor = transform_B(B)
			B = np.array(B, dtype = float) / 255.
			B_tensor = torch.tensor(B)[:,:,:3].permute((2,0,1)).float()

			# fig = plt.figure(1)
			# ax = fig.add_subplot(111)
			# ax.imshow(B_tensor[:,:1024,:].permute((1,2,0)))
			# plt.show()

		### if using instance maps        
		if not self.opt.no_instance:
			inst_path = self.inst_paths[index]
			inst = Image.open(inst_path)
			inst_tensor = transform_A(inst)

			if self.opt.load_features:
				feat_path = self.feat_paths[index]            
				feat = Image.open(feat_path).convert('RGB')
				norm = normalize()
				feat_tensor = norm(transform_A(feat))                            

		input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
					  'feat': feat_tensor, 'path': A_path}

		return input_dict
	def __getitem__(self, index):
		x1 = self.work(index)
		x2 = self.work(index + 1)
		# print (x1['label'].shape, x2['label'].shape)
		x2['label'] = torch.cat((x1['label'], x2['label']), dim = 0)
		if self.opt.isTrain:
			x2['image'] = torch.cat((x1['image'], x2['image']), dim = 0)

		return x2

	def __len__(self):
		# return 10
		return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize - 1

	def name(self):
		return 'AlignedDataset'