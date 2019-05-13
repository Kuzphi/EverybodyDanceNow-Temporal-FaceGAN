import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import os
import warnings
import argparse
import torch
import sys 
import json
sys.path.append('./src/utils')
from openpose_utils import create_label_full, create_face_label
from matplotlib import pyplot as plt
from functools import cmp_to_key

def gao(idx):
	# if os.path.exists(test_label_dir.joinpath('%s.torch' % idx)):
	# 	return
	anno_path = anno_dir.joinpath('%s_%s_keypoints.json' % (args.name, idx))
	anno = json.load(open(str(anno_path)))['people']
	if len(anno) == 0:
		print("warnings: %s no people"% idx)
		return [-1, -1, -1, -1]

	anno = anno[0]
	for key in anno:
		anno[key] = np.array(anno[key]).reshape(-1, 3).astype(np.float)
		anno[key][:,  0] -= 420
		anno[key][:, :2]  = (anno[key][:, :2] / 1080. * 512).clip(min=0)
	x = anno['pose_keypoints_2d'][:,:2]

	s = np.linalg.norm(x[1,:] - x[8,:])
	y = max(x[21,1], x[24,1])
	if y > 10 and x[1,:].min() > 5 and x[8, :].min() > 5:
		w = x[:,0].max() - x[:,0].min()
		b = (y - ymin)/(ymax-ymin) * (fmax - fmin) + fmin
		l = (y - ymin)/(ymax-ymin) * (tmax / smax - tmin / smin) + tmin / smin
		
		left = w * (1 - l) *  x[:,1].min() / (512 - w)
		for key in anno:
			anno[key] *= l

		d = max(x[21,1], x[24,1]) - b
		# print(l, d)
		for key in anno:
			anno[key][:, 0] += left
			anno[key][:, 1] -= d
		# print(max(x[21,1], x[24,1]) - b)

	# img_path = img_dir.joinpath('%s.png'%idx)
	# img = cv2.imread(str(img_path))[:, 420: -420]
	# img = cv2.resize(img, (512, 512))
	# cv2.imwrite(str(test_image_dir.joinpath('%s.png'%idx)), img)
	# label = create_label_full((512, 512), anno)

	# s = label.max(axis = 2)[:,:, np.newaxis]
	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)
	# ax.imshow((img * .8 + s * 255 * .2 ).astype(np.uint8))
	# ax.imshow((s[:,:, 0] * 255).astype(np.uint8))
	# plt.show()

	# label = torch.tensor(label).byte()
	# label_path = test_label_dir.joinpath('%s.torch'% idx)
	# torch.save(label, str(label_path))
	# print(str(test_image_dir.joinpath('%s.png'%idx)))

	# ================ Crop Face=====================
	face = anno['face_keypoints_2d']
	if face[:, 2].min() < 0.001:
		print(face[:, 2].min())
		return [-1, -1, -1, -1]
	minx, maxx = int(max(face[:, 1].min() - 20, 0)), int(min(face[:, 1].max() + 10, 512))
	miny, maxy = int(max(face[:, 0].min() - 15, 0)), int(min(face[:, 0].max() + 15, 512))

	face[:, 0] = (face[:, 0] - miny) / (maxy - miny + 1.) * 128.
	face[:, 1] = (face[:, 1] - minx) / (maxx - minx + 1.) * 128.
	face_label = create_face_label((128, 128), face)

	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)
	# s = face_label.max(axis = 2, keepdims = False)
	# print(s.shape)
	# ax.imshow(s)
	# plt.show()

	face_label = torch.tensor(face_label).byte()
	face_label_dir = test_face_label_dir.joinpath('%s.torch'% idx)
	torch.save(face_label, str(face_label_dir))
	return [minx, maxx, miny, maxy]

parser = argparse.ArgumentParser()
parser.add_argument('--name', metavar = '-n', type = str, help = 'name of the datset')
parser.add_argument('--which_train', metavar = '-a', type = str, help = 'name of the corresponding training set')

args = parser.parse_args()
save_dir = Path('./data/%s/'%(args.name))
anno_dir = save_dir.joinpath('anno')

save_dir.mkdir(exist_ok=True)
img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)
test_dir = save_dir.joinpath('test')
test_dir.mkdir(exist_ok=True)
test_label_dir = test_dir.joinpath('test_label')
test_label_dir.mkdir(exist_ok=True)
test_image_dir = test_dir.joinpath('test_img')
test_image_dir.mkdir(exist_ok=True)

test_face_label_dir = test_dir.joinpath('test_face_label')
test_face_label_dir.mkdir(exist_ok=True)
all_index = []
scale = []

for anno_name in sorted(os.listdir(anno_dir))[: 1800]:
	all_index.append(anno_name.split('_')[1])
	x = json.load(open(anno_dir.joinpath(anno_name)))['people']
	if len(x) == 0:
		continue
	x = x[0]
	x = np.array(x['pose_keypoints_2d']).reshape(-1, 3)[:,:2]

	x[:,  0] -= 420
	x = (x / 1080. * 512).clip(min=0)
	y = max(x[21,1], x[24,1])
	s = np.linalg.norm(x[1,:] - x[8,:])
	if x[1,:].min() < 5 or x[8, :].min() < 5:
		continue
	if y < 10:
		continue
	scale.append([y, s, int(all_index[-1])])

def xcmp(x, y):
	return x[0] - y[0]

scale = sorted(scale, key = cmp_to_key(xcmp))
scale = np.array(scale)
median = np.median(scale[:, 0])
xlen = int(scale.shape[0] * 0.05)
d = (scale[-1, 0] - scale[0,0]) * 0.1
print(scale.shape, d)


idx = np.searchsorted(scale[:, 0], scale[-1, 0] - d)
smax = scale[-idx:, 1].max()
midx = scale[-idx:, 1].argmax()
print (scale[-idx:, -1][midx])

idx = np.searchsorted(scale[:, 0], scale[0, 0] + d, side = 'right')
smin = scale[:idx, 1].max()
midx = scale[:idx, 1].argmax()
print (scale[:idx, -1][midx])

ymin, ymax = scale[0,0], scale[-1,0]
print(smin, smax, ymin, ymax)

f = open("./data/%s/train/scale.txt"%args.which_train,'r')
fmin, fmax, tmin, tmax = list(map(float,f.readline().split(' ')))


# for index in sorted(all_index)[:2000]:
# 	gao(index)

from multiprocessing import Pool
pool = Pool(10)
head_coor = pool.map(gao, sorted(all_index)[:2000])
head_coor = torch.tensor(head_coor)
print (head_coor)
torch.save(head_coor, str(test_dir.joinpath('face_crop_coor.torch')))
