import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random

class SRDataset(data.Dataset):
	def __init__(self, image_dir, list_path, input_transform = None, full_im_transform = None):
		super(SRDataset, self).__init__()

		self.image_dir = image_dir
		self.input_transform = input_transform
		self.full_im_transform = full_im_transform
		self.names = []
		self.box1s = []
		self.box2s = []
		self.labels = []

		list_file = open(list_path)
		lines = list_file.readlines()
		for line in lines:
			box1 = [] # x1, y1, x2, y2
			box2 = []
			line = line.strip().split()
			assert (len(line) == 10), 'The len of the row in list file is {%d}, does not equal to 10'.format(len(line))
			self.names.append(line[0])

			box1.append(int(line[1]))
			box1.append(int(line[2]))
			box1.append(int(line[3]))
			box1.append(int(line[4]))
			self.box1s.append(box1)

			box2.append(int(line[5]))
			box2.append(int(line[6]))
			box2.append(int(line[7]))
			box2.append(int(line[8]))
			self.box2s.append(box2)

			self.labels.append(int(line[9]))

		list_file.close()

	def __getitem__(self, index):
		# For normalize

		# PISC
		bbox_min = 0
		# bbox_max = 1497
		bbox_m = 1497.

		area_min = 198
		# area_max = 939736
		area_m = 939538.

		img = Image.open(os.path.join(self.image_dir, self.names[index])).convert('RGB') # convert gray to rgb
		box1 = self.box1s[index].copy()
		obj1 = img.crop((box1[0], box1[1], box1[2], box1[3]))

		box2 = self.box2s[index].copy()
		obj2 = img.crop((box2[0], box2[1], box2[2], box2[3]))

		# union
		u_x1 = min(box1[0], box2[0])
		u_y1 = min(box1[1], box2[1])
		u_x2 = max(box1[2], box2[2])
		u_y2 = max(box1[3], box2[3])
		union = img.crop((u_x1, u_y1, u_x2, u_y2))

		if self.input_transform:
			obj1 = self.input_transform(obj1)
			obj2 = self.input_transform(obj2)
			union = self.input_transform(union)

		area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
		area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)


		box1.append(area1)
		box2.append(area2)


		bpos =  box1 + box2
		
		bpos = np.array(bpos, dtype=np.float32)

		# normalize
		# can be changed later
		bpos[0:4] = 2 * (bpos[0:4] - bbox_min) / bbox_m - 1
		bpos[4] = 2 * (bpos[4] - area_min) / area_m - 1
		bpos[5:9] = 2 * (bpos[5:9] - bbox_min) / bbox_m - 1
		bpos[9] = 2 * (bpos[9] - area_min) / area_m - 1

       

		bpos = torch.from_numpy(bpos)


		target = self.labels[index]

		if self.full_im_transform:
			full_im = self.full_im_transform(img)
		else:
			full_im = img

		return union, obj1, obj2, bpos, target


	def __len__(self):
		return len(self.names)

	def weighted_sampler(self,class_num):
		np_labels = np.array(self.labels)
		clss_sample_count = np.array([ len(np.where(np_labels == t)[0]) for t in range(class_num)])
		weight = 1. / (clss_sample_count)
		sample_weight = np.array([weight[t] for t in np_labels])
		sample_weight = torch.from_numpy(sample_weight)
		sample_weight = sample_weight.double()
		sampler = data.sampler.WeightedRandomSampler(sample_weight,len(sample_weight),replacement=True)
		return sampler

