import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from torchvision import datasets, transforms,models

class SRDataset(data.Dataset):
    def __init__(self, max_person,image_dir, images_list,bboxes_list,relations_list, image_size,input_transform=None):
        super(SRDataset, self).__init__()
        self.max_person = max_person
        self.image_dir = image_dir
        self.image_size = image_size
        self.input_transform = input_transform		
        self.names = []

        with open(images_list,'r') as fin:
            for line in fin:
                self.names.append(line.split()[0])
        with open(bboxes_list,'r') as fin:
            self.bboxes = json.load(fin)

        with open(relations_list,'r') as fin:
            self.relations = json.load(fin)


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_dir, self.names[index])).convert('RGB') # convert gray to rgb
        (w, h) = img.size

        bbox_num = len(self.bboxes[index])
        image_bboxes = np.zeros((self.max_person,4),dtype=np.float32)
        bbox_np = np.array(self.bboxes[index])
        
        image_bboxes[:,0] = 0
        image_bboxes[:,1] = 0
        image_bboxes[:,2] = w -1 
        image_bboxes[:,3] = h -1

        image_bboxes[0:bbox_num,:] = bbox_np[:,:]
        image_bboxes = torch.from_numpy(image_bboxes)

        if self.input_transform:
            img ,image_bboxes= self.input_transform(img,image_bboxes)


        relation_mask = np.zeros((self.max_person,self.max_person),dtype=np.int32)
        full_mask = np.zeros((self.max_person,self.max_person),dtype=np.int32)
        relation_id = np.zeros((self.max_person,self.max_person),dtype=np.int32)
        image_relations = self.relations[index]
        for i in range(len(image_relations)):
            image_relation = image_relations[i]
            relation_mask[image_relation[0]][image_relation[1]] = 1
            #relation_mask[image_relation[1]][image_relation[0]] = 1
            full_mask[image_relation[0]][image_relation[1]] = 1
            full_mask[image_relation[1]][image_relation[0]] = 1


            relation_id[image_relation[0]][image_relation[1]] = image_relation[2]
            relation_id[image_relation[1]][image_relation[0]] = image_relation[2]

        full_mask = torch.from_numpy(full_mask).long()
        relation_mask = torch.from_numpy(relation_mask).long()
        relation_id = torch.from_numpy(relation_id).long()


        #img: [3,image_size,image_size]
        #bbox_num: single number for the bbox number in this image
        #image_bboxes: [max_person, 4], the first bbox_num has real value, others are zero
        #relation_mask: [max_person,max_person] mask 
        #relation_id: [max_person,max_person] labels
        return img, image_bboxes, relation_mask,relation_id,full_mask



    def __len__(self):
        return len(self.names)


