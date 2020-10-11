import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import resnet

class pair_cnn(nn.Module):
    def __init__(self, num_classes = 3):
        super(pair_cnn, self).__init__()
        #self.resnet101_union = resnet.resnet101()
        self.resnet101_a = resnet.resnet101()
        #self.resnet101_b = self.resnet101_a

        #self.bboxes = nn.Linear(10, 256)
        self.fc6 = nn.Linear(2048+2048, 4096)
        self.fc7 = nn.Linear(4096, num_classes)
        self.ReLU = nn.ReLU(False)
        self.Dropout = nn.Dropout()

        self._initialize_weights()

    # x1 = union, x2 = object1, x3 = object2, x4 = bbox geometric info
    def forward(self, x1, x2, x3, x4): 
        #x1 = self.resnet101_union(x1)
        x2 = self.resnet101_a(x2)
        x3 = self.resnet101_a(x3)
        #x4 = self.bboxes(x4)

        x = torch.cat((x2, x3), 1)
        x = self.Dropout(x)
        fc6 = self.fc6(x)
        x = self.ReLU(fc6)
        x = self.Dropout(x)
        x = self.fc7(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()   

if __name__ == '__main__':
    a = pair_cnn()    