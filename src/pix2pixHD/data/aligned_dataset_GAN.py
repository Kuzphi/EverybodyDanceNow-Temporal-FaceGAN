### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2
class AlignedDataset(BaseDataset):
	def initialize(self, opt):
		self.opt = opt

		### input A (label maps)
		self.label_paths = sorted(make_dataset(opt.label_root))
		# self.simage_paths = sorted(make_dataset(opt.input_image_root))
		### input B (real images)
		if opt.isTrain:
			self.rimage_paths = sorted(make_dataset(opt.real_image_root))

		### instance maps
		if not opt.no_instance:
			self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
			self.inst_paths = sorted(make_dataset(self.dir_inst))

		### load precomputed instance-wise encoded features
		if opt.load_features:                              
			self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
			print('----------- loading features from %s ----------' % self.dir_feat)
			self.feat_paths = sorted(make_dataset(self.dir_feat))

		x = 'train' if opt.isTrain else 'test'
		self.crop_coor = torch.load('../data/%s/%s/face_crop_coor.torch'% (opt.dataset_name, x))
		self.dataset_size = len(self.label_paths) 
	  
	def __getitem__(self, index):        
		### input A (label maps)
		lpath = self.label_paths[index]
		A_tensor_0 = torch.load(lpath).permute((2,0,1)).float()

		idx_ = lpath.split('/')[-1][:12]
		spath = self.opt.input_image_root + '%s_synthesized_image.jpg'%idx_
		A = Image.open(spath).convert('RGB')
		A = np.array(A, dtype = float) / 255.
		A = A[:,:,:3]
		idx = lpath.split('/')[-1].split('.')[0]

		minx, maxx, miny, maxy = list(self.crop_coor[int(idx), :])
		A = A[minx: maxx + 1, miny: maxy + 1, :]
		A  = cv2.resize(A, (128, 128))
		A_tensor_1 = torch.tensor(A).permute((2,0,1)).float()
		A_tensor = torch.cat((A_tensor_0, A_tensor_1), dim = 0)

		B_tensor = inst_tensor = feat_tensor = 0


		lidx = lpath.split('/')[-1][:12]
		sidx = spath.split('/')[-1][:12]
		### input B (real images)
		if self.opt.isTrain:
			B_path = self.rimage_paths[index]   
			B = Image.open(B_path).convert('RGB')
			B = np.array(B, dtype = float) / 255.
			B_tensor = torch.tensor(B)[:,:,:3].permute((2,0,1)).float()

			# fig = plt.figure(1)
			# ax = fig.add_subplot(111)
			# ax.imshow(B_tensor[:,:1024,:].permute((1,2,0)))
			# plt.show()
			ridx = B_path.split('/')[-1][:12]
			assert lidx == ridx , "Wrong match"

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

		# print(lpath, spath, B_path)
		# print(lidx, sidx )
		assert lidx == sidx , "Wrong match"
		# fig = plt.figure(1)
		# ax = fig.add_subplot(111)
		# ax.imshow(A)
		# plt.show()

		input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
					  'feat': feat_tensor, 'path': lpath}

		return input_dict

	def __len__(self):
		return self.dataset_size

	def name(self):
		return 'AlignedDataset'