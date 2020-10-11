import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import os
import math
from torch.autograd import Variable
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_ROI(nn.Module):

    def __init__(self, block, layers,zero_init_residual=False):
        super(ResNet_ROI, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    

    # botttom with a feature dim of [batch,C,H/16,W/16]
    # batch_bboxes with [batch , node_num , 4]
    def crop_pool_layer(self,bottom,batch_bboxes):
        # implement it using stn
        # box to affine
        # input (x1,y1,x2,y2)
        """
        [  x2-x1             x1 + x2 - W + 1  ]
        [  -----      0      ---------------  ]
        [  W - 1                  W - 1       ]
        [                                     ]
        [           y2-y1    y1 + y2 - H + 1  ]
        [    0      -----    ---------------  ]
        [           H - 1         H - 1       ]
        """
        batch_bboxes = batch_bboxes.detach()
        node_num = batch_bboxes.size(1)
        batch_bboxes = batch_bboxes.view(-1,4)

        x1 = batch_bboxes[:, 0] / 32.0
        y1 = batch_bboxes[:, 1] / 32.0
        x2 = batch_bboxes[:, 2] / 32.0
        y2 = batch_bboxes[:, 3] / 32.0

        batch_size = bottom.size(0)
        channel_num = bottom.size(1)
        height = bottom.size(2)
        width = bottom.size(3)

        # affine theta
        theta = Variable(batch_bboxes.data.new(batch_bboxes.size(0), 2, 3).zero_())
        theta[:, 0, 0] = (x2 - x1) / (width - 1)
        theta[:, 0, 2] = (x1 + x2 - width + 1) / (width - 1)
        theta[:, 1, 1] = (y2 - y1) / (height - 1)
        theta[:, 1, 2] = (y1 + y2 - height + 1) / (height - 1)

        POOLING_SIZE = height * 2

        grid = F.affine_grid(theta, torch.Size((batch_bboxes.size(0), 1, POOLING_SIZE, POOLING_SIZE)))
        crops = F.grid_sample(bottom.view(-1,1,channel_num,height,width).repeat(1,node_num,1,1,1).view(-1,channel_num,height,width), grid) 
        crops = F.max_pool2d(crops,2,2)
        return crops

    # return with fature dim [batch, node_num, 2048]
    def forward(self, x, batch_bboxes):
        
        batch_size = x.size(0)
        node_num = batch_bboxes.size(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
    
        x = self.layer1(x)
        
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.crop_pool_layer(x,batch_bboxes)
         
        x = self.avgpool(x)
        x = x.view(batch_size,node_num,-1)
        return x


def resnet101_roi(pretrained=True):
    model = ResNet_ROI(Bottleneck, [3, 4, 23, 3])
    
    if pretrained:
        unload_model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join('./resnet_model/resnet101-5d3b4d8f.pth'))
        print(len(pretrained_dict))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in unload_model_dict and pretrained_dict[k].shape == unload_model_dict[k].shape )}           
        print(len(pretrained_dict))
        # for dict_inx, (k,v) in enumerate(pretrained_dict.items()):
        #     print(dict_inx,k,v.shape)
        unload_model_dict.update(pretrained_dict) 
        model.load_state_dict(unload_model_dict) 
    return model