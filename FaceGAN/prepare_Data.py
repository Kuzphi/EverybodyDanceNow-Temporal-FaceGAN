import os
import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
import argparse
import config.prepare_opt as opt
pix2pixhd_dir = Path('../src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

from data.data_loader import CreateDataLoader
from data.image_folder import make_dataset
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
parser = argparse.ArgumentParser()
parser.add_argument('--name', metavar = '-n', type = str, help = 'name of the datset')
args = parser.parse_args()

face_data = make_dataset('../data/%s/train/face_label/' %args.name)
save_dir = './data/%s_%s_%s/' %(args.name, opt.name, opt.which_epoch)
os.makedirs(save_dir, exist_ok = True)

model = create_model(opt)
for path in sorted(face_data):
	path = path.replace('face_label', 'train_label')
	# print(path)
	label_ = torch.load(path).unsqueeze(0)
	# print(label_.shape)
	label = torch.cat((label_, label_), 3)
	label = label.permute((0,3,1,2))
	
	with torch.no_grad():
		generated = model.inference(label, None)

	image = np.array(generated.cpu()[0])
	idx = path.split('/')[-1].split('.')[0]
	img = image.transpose((1,2,0)) * 255
	img = img.astype(np.uint8)
	print(save_dir + "%s.png" % idx)

	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)
	# ax.imshow(img)
	# plt.show()

	cv2.imwrite(save_dir + "%s.png" % idx, img[:,:,::-1])
