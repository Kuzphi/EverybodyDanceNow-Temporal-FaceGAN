import os
import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt

pix2pixhd_dir = Path('./src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import src.config.test_opt_temporal as opt

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

model = create_model(opt)

for data in tqdm(dataset):
	minibatch = 1
	
	with torch.no_grad():
		generated = model.inference(data['label'], data['inst'])

	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)
	# ax.imshow(generated[0].cpu().permute((1,2,0)))
	# plt.show()

	visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
						   ('synthesized_image', util.tensor2im(generated.data[0]))])
	img_path = data['path']
	visualizer.save_images(webpage, visuals, img_path)
webpage.save()
torch.cuda.empty_cache()
