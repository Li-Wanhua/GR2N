import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 

import resnet_roi
import GRRN

class RIG(nn.Module):
    def __init__(self,num_class=5,hidden_dim=2048,time_step=3,node_num=5):
        super(RIG,self).__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.time_step = time_step
        self.node_num = node_num

        self.roi_net = resnet_roi.resnet101_roi()
        self.grrn=GRRN.GRRN(state_dim=hidden_dim, edge_types = num_class,time_step = time_step)
    
    # batch_bboxes are transformed with dim [batch,node_num, 4]
    # return with [batch, node_num, node_num, num_class]
    def forward(self, imgs, batch_bboxes,full_mask):

        #[batcn, node_num, 2048]
        rois_feature = self.roi_net(imgs, batch_bboxes)

        all_scores = self.grrn(rois_feature,full_mask)
       
        return all_scores

