import os
import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
import cv2
pix2pixhd_dir = Path('../src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import config.inference_opt as opt


# input_image_root='../results/updated/test_latest/images'
# label_root='../data/1/test/face_label'

opt.input_image_root =  '../results/%s/test_%s/images/' % (opt.model_name, opt.which_epoch)
opt.label_root       = '../data/%s/test/test_face_label/' % opt.dataset_name

face_coor = '../data/%s/test/face_crop_coor.torch' % opt.dataset_name
face_coor = torch.load(face_coor)
# syn_face_img_dir = './data/face_gan_%s_%s/'% (opt.model_name, opt.which_epoch)
# os.makedirs(syn_face_img_dir, exist_ok = True)

#crop the face in synthestic image 
# if len(os.listdir(syn_face_img_dir)) > 300:
# 	for face_label_name in sorted(os.listdir(opt.label_root)):
# 		if not face_label_name.endswith('torch'):
# 			continue
# 		img_name = face_label_name[:12] + '_synthesized_image.jpg'
# 		idx = int(face_label_name[:12])

# 		img = cv2.imread(os.path.join(opt.synthetic_image_root, img_name))
# 		minx, maxx, miny, maxy = list(face_coor[idx,:])
# 		assert face_coor[idx,:].min() >= 0, "Wrong Match"
# 		img = img[minx: maxx + 1, miny: maxy + 1 , :]
# 		img = cv2.resize(img, (128,128))
# 		cv2.imwrite(syn_face_img_dir + '%s.png' % img_name[:12], img)

# opt.input_image_root = syn_face_img_dir

save_dir = opt.results_dir + ''
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

model = create_model(opt)
face_save_dir = './result/face_gan_%s_%s/face/'% (opt.model_name, opt.which_epoch)
whole_save_dir = './result/face_gan_%s_%s/whole/'% (opt.model_name, opt.which_epoch)
os.makedirs(save_dir, exist_ok = True)
os.makedirs(face_save_dir, exist_ok = True)
os.makedirs(whole_save_dir, exist_ok = True)

# for data in tqdm(dataset):
# 	minibatch = 1
	
# 	with torch.no_grad():
# 		generated = model.inference(data['label'], data['inst'])

# 	# fig = plt.figure(1)
# 	# ax = fig.add_subplot(111)
# 	# ax.imshow(generated[0].cpu().permute((1,2,0)))
# 	# plt.show()
# 	input_image = data['label'][0, 10:, :, :]
# 	input_label = data['label'][0, :10, :, :]
# 	refined_face = util.tensor2im(generated[0])
# 	visuals = OrderedDict([('input_label', util.tensor2label(input_label, opt.label_nc)),
# 						   ('refined_face', refined_face)])
	
# 	idx = str(data['path']).split("/")[-1][:12]
# 	cv2.imwrite(face_save_dir + '%s.png' % idx, refined_face[:,:,::-1])
# 	# print(opt.input_image_root + '%s_synthesized_image.jpg'% idx)
# 	refined_syn_img = cv2.imread(opt.input_image_root + '%s_synthesized_image.jpg'% idx)

# 	minx, maxx, miny, maxy = list(face_coor[int(idx),:])
# 	assert face_coor[int(idx),:].min() >= 0, "Wrong Match"
# 	refined_face = cv2.resize(refined_face, (maxy - miny + 1, maxx - minx + 1))
# 	refined_syn_img[minx: maxx + 1, miny: maxy + 1 , :] = refined_face[:,:,::-1]

# 	cv2.imwrite(whole_save_dir + '%s_refined.png' % idx, refined_syn_img)

from shutil import copyfile
from pathlib import Path

target_dir = Path('../results/%s/test_%s/images/' % (opt.model_name, opt.which_epoch))
refiend_dir  = Path(whole_save_dir)

target_dir_path = sorted(target_dir.glob('*synthesized*'))
refined_dir_path = sorted(refiend_dir.glob('*refined*'))

save_dir   = '../results/%s/test_%s/refined_images/' % (opt.model_name, opt.which_epoch)
os.makedirs(save_dir, exist_ok = True)

for path in target_dir_path:
	img = cv2.imread(str(path))
	idx = str(path).split('/')[-1][:12]
	cv2.imwrite('%s/%s.png' %(save_dir, idx), img)

for path in refined_dir_path:
	idx = str(path).split('/')[-1][:12]
	copyfile(path, '%s/%s.png' %(save_dir, idx) )

