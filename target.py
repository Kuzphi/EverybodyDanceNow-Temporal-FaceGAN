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
from functools import cmp_to_key
sys.path.append('./src/utils')
from openpose_utils import create_label_full, create_face_label
from matplotlib import pyplot as plt

def gao(idx):
	# if os.path.exists(train_head_dir.joinpath('%s.png' % idx)):
	# 	return
	print(idx)
	anno_path = anno_dir.joinpath('%s_%s_keypoints.json' % (args.name, idx))
	anno = json.load(open(str(anno_path)))['people']
	if len(anno) == 0:
		print("warnings: %s no people"% idx)
		return [-1, -1, -1, -1]
	anno = anno[0]
	# print(np.array(anno['pose_keypoints_2d']).reshape(-1, 3))
	for key in anno:
		anno[key] = np.array(anno[key]).reshape(-1, 3)
		anno[key][:,  0] -= 420
		# print(anno[key][:,  0].min())
		anno[key][:, :2]  = (anno[key][:, :2] / 1080. * 512).clip(min=0)

	# print(anno['pose_keypoints_2d'][:, :2][:, ::-1])
	
	img_path = img_dir.joinpath('%s.png'%idx)
	img = cv2.imread(str(img_path))[:, 420: -420]
	img = cv2.resize(img, (512, 512))

	# cv2.imwrite(str(train_img_dir.joinpath(idx + '.png')), img)
	# label = create_label_full((512, 512), anno)

	# s = label.max(axis = 2)[:,:, np.newaxis]
	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)
	# ax.imshow((img * .8 + s * 255 * .2 ).astype(np.uint8))
	# ax.imshow((s[:,:, 0] * 255).astype(np.uint8))
	# plt.show()


	# label = torch.tensor(label).byte()
	# label_path = train_label_dir.joinpath('%s.torch'% idx)
	# torch.save(label, str(label_path))
	# ===================== Crop Face ==============================
	
	face = anno['face_keypoints_2d']
	if face[:, 2].min() < 0.001:
		print(face[:, 2].min())
		return [-1, -1, -1, -1]
	minx, maxx = int(max(face[:, 1].min() - 20, 0)), int(min(face[:, 1].max() + 10, 512))
	miny, maxy = int(max(face[:, 0].min() - 15, 0)), int(min(face[:, 0].max() + 15, 512))
	# print(minx, maxx , miny, maxy )
	face_img = img[minx: maxx + 1 , miny: maxy + 1, :]
	face_img = cv2.resize(face_img, (128,128))
	cv2.imwrite(str(train_face_image_dir.joinpath('%s.png' % idx)), face_img)


	face[:, 0] = (face[:, 0] - miny) / (maxy - miny + 1.) * 128.
	face[:, 1] = (face[:, 1] - minx) / (maxx - minx + 1.) * 128.
	face_label = create_face_label((128, 128), face)

	# s = face_label.max(axis = 2)[:,:, np.newaxis]
	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)
	# ax.imshow((s[:,:, 0] * 255).astype(np.uint8))
	# plt.show()

	face_label = torch.tensor(face_label).byte()
	face_label_dir = train_face_label_dir.joinpath('%s.torch'% idx)
	torch.save(face_label, str(face_label_dir))
	return [minx, maxx, miny, maxy]
	# ===================== Crop Face ==============================

parser = argparse.ArgumentParser()
parser.add_argument('--name', metavar = '-n', type = str, help = 'name of the datset')
parser.add_argument('--type', metavar = '-t', type = str, help = 'target or source')

args = parser.parse_args()
save_dir = Path('./data/%s/'%(args.name))
anno_dir = save_dir.joinpath('anno')

save_dir.mkdir(exist_ok=True)
img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)
train_dir = save_dir.joinpath('train')
train_dir.mkdir(exist_ok=True)
train_img_dir = train_dir.joinpath('train_img')
train_img_dir.mkdir(exist_ok=True)
train_label_dir = train_dir.joinpath('train_label')
train_label_dir.mkdir(exist_ok=True)

train_face_image_dir = train_dir.joinpath('face_img')
train_face_image_dir.mkdir(exist_ok=True)

train_face_label_dir = train_dir.joinpath('face_label')
train_face_label_dir.mkdir(exist_ok=True)

if len(os.listdir(img_dir)) < 400:
	cap = cv2.VideoCapture(str(save_dir.joinpath('video.mp4')))
	i = 0
	while (cap.isOpened()):
		flag, frame = cap.read()
		if flag == False :
			break
		cv2.imwrite(str(img_dir.joinpath('{:012}.png'.format(i))), frame)
		if i%100 == 0:
			print('Has generated %d picetures'%i)
		i += 1

all_index = []
ymin, ymax = 2000, -1
smin, smax = 2000, -1
idmin, idmax = 0, 0
scale = []
for anno_name in sorted(os.listdir(anno_dir)):
	all_index.append(anno_name.split('_')[1])
	x = json.load(open(anno_dir.joinpath(anno_name)))['people'][0]
	x = np.array(x['pose_keypoints_2d']).reshape(-1, 3)[:,:2]

	x[:,  0] -= 420
	x = (x / 1080. * 512).clip(min=0)

	s = np.linalg.norm(x[1,:] - x[8,:])
	if x[1,:].min() < 5 or x[8, :].min() < 5:
		continue
	y = max(x[21,1], x[24,1])
	if y < 10:
		continue
	scale.append([y, s, int(all_index[-1])])

# def xcmp(x, y):
# 	return x[0] - y[0]
# scale = sorted(scale, key = cmp_to_key(xcmp))
# scale = np.array(scale)
# median = np.median(scale[:, 0])
# xlen = int(scale.shape[0] * 0.05)
# d = (scale[-1, 0] - scale[0,0]) * 0.1
# print(scale.shape, d)


# idx = np.searchsorted(scale[:, 0], scale[-1, 0] - d)
# smax = scale[-idx:, 1].max()
# midx = scale[-idx:, 1].argmax()
# print (scale[-idx:, -1][midx])

# idx = np.searchsorted(scale[:, 0], scale[0, 0] + d, side = 'right')
# smin = scale[:idx, 1].max()
# midx = scale[:idx, 1].argmax()
# print (scale[:idx, -1][midx])

# print(scale[0, 0], scale[-1, 0 ])
# print(smin, smax)
# f = open(train_dir.joinpath('scale.txt'),'w')
# f.write( ' '.join([str(scale[0, 0]), str(scale[-1, 0]), str(smin), str(smax)]) )

from multiprocessing import Pool
pool = Pool(10)
head_coor = pool.map(gao, sorted(all_index))
head_coor = torch.tensor(head_coor)
print (head_coor)
torch.save(head_coor, str(train_dir.joinpath('face_crop_coor.torch')))

# for index in all_index:
# 	gao(index)

	
