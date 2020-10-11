import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,models
from utils import transforms
import numpy as np
import random
import os
import json
import time
import AverageMeter
import SRDataset
import RIG
import pdb

#super-parameters
parser = argparse.ArgumentParser(description='PyTorch Social Relation')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--max-person', type=int, default=5, metavar='N',
                    help=' the maximum person in an image (default: 5)')
parser.add_argument('--image-size', type=int, default=224, metavar='N',
                    help='the size of image (default: 224)')
parser.add_argument('--images-root', type=str,
                        help='images root for dataset')
parser.add_argument('--train-file-pre', type=str, default='../domain_split/domain_trainidx.txt',
                        help='data file for train dataset')
parser.add_argument('--valid-file-pre', type=str, default='../domain_split/domain_valididx.txt',
                        help='data file for valid dataset')
parser.add_argument('--test-file-pre', type=str, default='../domain_split/domain_testidx.txt',
                        help='data file for test dataset')
parser.add_argument('--num-workers',  default=2, type=int,
                   help='number of load data workers (default: 2)')
parser.add_argument('--num-classes',  default=6, type=int,
                   help='number of classes (default: 6)')
parser.add_argument('--save-model', type=str, default='../Saved_Model/',
                        help='where you save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--fc-lr', type=float, default=0.01, 
                    help='fc layer learning rate (default: 0.01)')
parser.add_argument('--max-epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--print-freq',  default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--manualSeed',  type=int,
                     help='manual seed')
parser.add_argument('--time-steps', type=int, default=0, metavar='N',
                    help='the time steps (default: 0)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.manualSeed is None or args.manualSeed < 0:
    args.manualSeed = random.randint(1,10000)
    
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
setup_seed(args.manualSeed)

class edge_loss(nn.Module):
    def __init__(self):
        super(edge_loss,self).__init__()
        self.criterion = nn.BCELoss(reduction='sum')
    def forward(self,all_scores,labels,masks):
        total_loss = 0
        masks = masks.view(-1,1).repeat(1,args.num_classes).float()
        masks = masks.detach()
        labels = labels.view((-1,1))
        one_hot_labels = Variable(torch.zeros((labels.size(0),args.num_classes)).float()).cuda()
        one_hot_labels.scatter_(1,labels,1.0)
        loss_num = len(all_scores)
        

        
        for scores in all_scores[-1:]:
            scores = scores.view(-1,args.num_classes)
            raw_losses = self.criterion(scores * masks ,one_hot_labels * masks)
            losses = raw_losses  / torch.sum(masks)
            total_loss += losses 

        return total_loss

def cal_acc(all_logits,labels,masks):

    labels_np = labels.data.cpu().long().numpy()
    masks_np = masks.data.cpu().long().numpy()
    count = np.sum(masks_np)
    acc_list = []
    all_logits_np = []
    for logits in all_logits:
        logits_np = logits.data.cpu().numpy()
        all_logits_np.append(logits_np)
        pred = np.argmax(logits_np,axis=3)
        res = (pred == labels_np)
        res = res * masks_np
        right_num = np.sum(res)
        acc_list.append(right_num * 1.0 / count)
        
    if(len(acc_list) > 1):
        all_logits_np = np.array(all_logits_np)
        all_logits_np_mean = np.mean(all_logits_np,axis=0)
        pred = np.argmax(all_logits_np_mean,axis=3)
        res = (pred == labels_np)
        res = res * masks_np
        right_num = np.sum(res)
        acc_list.append(right_num * 1.0 / count)

        all_logits_np_max = np.max(all_logits_np,axis=0)
        pred = np.argmax(all_logits_np_max,axis=3)
        res = (pred == labels_np)
        res = res * masks_np
        right_num = np.sum(res)
        acc_list.append(right_num * 1.0 / count)


    return acc_list, count

  
    
def load_model(unload_model):
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
        print(args.save_model,'is created!')
    if not os.path.exists(os.path.join(args.save_model,'checkpoint.txt')):
        f = open(os.path.join(args.save_model,'checkpoint.txt'),'w')
        print('checkpoint','is created!')

    start_index = 0
    with open(os.path.join(args.save_model,'checkpoint.txt'),'r') as fin:
        lines = fin.readlines()
        if len(lines) > 0:
            model_path,model_index = lines[0].split()
            print('Resuming from',model_path)
            if int(model_index) == 0:
                unload_model_dict = unload_model.state_dict()
                # print(len(unload_model_dict))
                # for dict_inx, (k,v) in enumerate(unload_model_dict.items()):
                #     print(dict_inx,k,v.shape)

                pretrained_dict = torch.load(os.path.join(args.save_model,model_path))

                pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in unload_model_dict and pretrained_dict[k].shape == unload_model_dict[k].shape )}           
                print(len(pretrained_dict))
                for dict_inx, (k,v) in enumerate(pretrained_dict.items()):
                    print(dict_inx,k,v.shape)
                unload_model_dict.update(pretrained_dict) 
                unload_model.load_state_dict(unload_model_dict)       
            else:
                unload_model.load_state_dict(torch.load(os.path.join(args.save_model,model_path)))
            
            start_index = int(model_index) + 1
    return start_index
 

def save_model(tosave_model,epoch):
    model_epoch = '%04d' % (epoch)
    model_path = 'model-' + model_epoch + '.pth'
    save_path = os.path.join(args.save_model,model_path)
    torch.save(tosave_model.module.state_dict(), save_path)
    with open(os.path.join(args.save_model,'checkpoint.txt'),'w') as fin:
        fin.write(model_path + ' ' + str(epoch) + '\n')


#dataset prepare
#---------------------------------
print('Loading dataset...')
cache_size = 256
if args.image_size == 448:
    cache_size = 256 * 2
if args.image_size == 352:
    cache_size = 402
transform_train = transforms.Compose([
    transforms.Resize((cache_size,cache_size)),
    #transforms.Resize((args.image_size,args.image_size)),
    #transforms.RandomRotation(10),
    transforms.RandomCrop(args.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((cache_size,cache_size)),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])


print("Dataset Initializing...")
trainset = SRDataset.SRDataset(max_person=args.max_person,image_dir=args.images_root, \
    images_list=args.train_file_pre + '_images.txt',bboxes_list=args.train_file_pre + '_bbox.json', \
    relations_list=args.train_file_pre + '_relation.json', image_size=args.image_size,input_transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,worker_init_fn=np.random.seed(args.manualSeed))

validset = SRDataset.SRDataset(max_person=args.max_person,image_dir=args.images_root, \
    images_list=args.valid_file_pre + '_images.txt',bboxes_list=args.valid_file_pre + '_bbox.json', \
    relations_list=args.valid_file_pre + '_relation.json', image_size=args.image_size,input_transform=transform_test)
validloader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,worker_init_fn=np.random.seed(args.manualSeed))


testset = SRDataset.SRDataset(max_person=args.max_person,image_dir=args.images_root, \
    images_list=args.test_file_pre + '_images.txt',bboxes_list=args.test_file_pre + '_bbox.json', \
    relations_list=args.test_file_pre + '_relation.json', image_size=args.image_size,input_transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,worker_init_fn=np.random.seed(args.manualSeed))


##Model prepare
print("Loading model...")
SRModel = RIG.RIG(num_class=args.num_classes,hidden_dim=2048,time_step=args.time_steps,node_num=args.max_person)

#start_epoch = load_model(SRModel)
start_epoch =  1

SRModel.cuda()


SRModel = torch.nn.DataParallel(SRModel)


criterion = edge_loss()
#criterion.cuda()

def is_fc(para_name):
    split_name = para_name.split('.')
    if (split_name[1] == 'classifier'):
        return True
    else:
        return False

params = []
print("FC layers:")
for keys,param_value in SRModel.named_parameters():
    #print(keys)
    if (is_fc(keys)):
        params += [{'params':[param_value],'lr':args.fc_lr}]
        print(keys)
    else:
        params += [{'params':[param_value],'lr':args.lr}]


#optimizer = optim.Adam(SRModel.parameters(), lr=args.lr)
optimizer = optim.Adam(params, lr=args.lr)
#optimizer = optim.SGD(params, momentum=0.9, weight_decay=5e-4)

def train_epoch(epoch,log_file):
    
    batch_time = AverageMeter.AverageMeter()
    data_time = AverageMeter.AverageMeter()
    losses = AverageMeter.AverageMeter()
    acces = []
    for i in range(1 + args.time_steps):
        acces.append(AverageMeter.AverageMeter())

    SRModel.train()

    end_time = time.time()
    for batch_idx, (img, image_bboxes, relation_half_mask,relation_id,full_mask) in enumerate(trainloader):
        data_time.update(time.time() - end_time)

        img, image_bboxes, relation_half_mask,relation_id,full_mask = img.cuda(), image_bboxes.cuda(),  relation_half_mask.cuda(), relation_id.cuda(),full_mask.cuda()
        img, image_bboxes, relation_half_mask,relation_id,full_mask = Variable(img), Variable(image_bboxes),Variable(relation_half_mask), Variable(relation_id), Variable(full_mask)

        optimizer.zero_grad()
        #pdb.set_trace()

        logits = SRModel(img, image_bboxes,full_mask)
        #pdb.set_trace()
        loss = criterion(logits, relation_id,relation_half_mask )

        loss.backward()
        optimizer.step()
        losses.update(loss.cpu().data.numpy())

        #calcalate accuracy
        acc_list,count = cal_acc(logits, relation_id,relation_half_mask)
        for i in range(1 + args.time_steps):
            acces[i].update(acc_list[i],count)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        acc_str = ''
        for i in range(1 + args.time_steps):
            acc_str += ('%.3f ' % acces[i].avg)
        
        if batch_idx % args.print_freq == 0:
            print('Epoch: [%d][%d/%d]  ' 
                 'Time %.3f (%.3f)\t'
                 'Data %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                 'pair %.1f\t'
                  % (epoch, batch_idx, len(trainloader),
                    batch_time.val,batch_time.avg, data_time.val,data_time.avg,
                    losses.val,losses.avg,count * 1.0) + acc_str)
            log_file.write('Epoch: [%d][%d/%d]\t' 
                 'Time %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                 % (epoch, batch_idx, len(trainloader),
                    batch_time.val,batch_time.avg,
                    losses.val,losses.avg) + acc_str + '\n')

def valid_epoch(epoch,log_file):

    batch_time = AverageMeter.AverageMeter()

    losses = AverageMeter.AverageMeter()
    acces = []
    for i  in range(1 + args.time_steps):
        acces.append(AverageMeter.AverageMeter())

    SRModel.eval()

    end_time = time.time()
    for batch_idx, (img, image_bboxes, relation_half_mask,relation_id,full_mask) in enumerate(validloader):

        img, image_bboxes, relation_half_mask,relation_id,full_mask = img.cuda(), image_bboxes.cuda(),  relation_half_mask.cuda(), relation_id.cuda(), full_mask.cuda()
        img, image_bboxes, relation_half_mask,relation_id,full_mask = Variable(img), Variable(image_bboxes),Variable(relation_half_mask), Variable(relation_id), Variable(full_mask)

        
        logits = SRModel(img, image_bboxes,full_mask)
        
        loss = criterion(logits, relation_id,relation_half_mask)

        losses.update(loss.cpu().data.numpy())

        #calcalate accuracy
        acc_list,count = cal_acc(logits, relation_id,relation_half_mask)
        for i in range(1 + args.time_steps):
            acces[i].update(acc_list[i],count)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        acc_str = ''
        for i in range(1 + args.time_steps):
            acc_str += ('%.3f ' % acces[i].avg)
        
        if batch_idx % args.print_freq == 0:
            print('Epoch: [%d][%d/%d]  ' 
                 'Time %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                  % (epoch, batch_idx, len(validloader),
                    batch_time.val,batch_time.avg, 
                    losses.val,losses.avg) + acc_str)
            log_file.write('Epoch: [%d][%d/%d]\t' 
                 'Time %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                 % (epoch, batch_idx, len(validloader),
                    batch_time.val,batch_time.avg,
                    losses.val,losses.avg) +acc_str + '\n')
    
    acc_str = ''
    for i in range(1 + args.time_steps):
        acc_str += ('%.3f ' % acces[i].avg)
    print("Valid: Acc " + acc_str + '\n')
    log_file.write("Valid: Acc " + acc_str + '\n' )
    return acces

def test_epoch(epoch,log_file):

    batch_time = AverageMeter.AverageMeter()

    losses = AverageMeter.AverageMeter()
    acces = []
    for i  in range(1 + args.time_steps):
        acces.append(AverageMeter.AverageMeter())

    SRModel.eval()

    end_time = time.time()
    for batch_idx, (img, image_bboxes, relation_half_mask,relation_id,full_mask) in enumerate(testloader):

        img, image_bboxes, relation_half_mask,relation_id,full_mask = img.cuda(), image_bboxes.cuda(),  relation_half_mask.cuda(), relation_id.cuda(), full_mask.cuda()
        img, image_bboxes, relation_half_mask,relation_id,full_mask = Variable(img), Variable(image_bboxes),Variable(relation_half_mask), Variable(relation_id), Variable(full_mask)

        
        logits = SRModel(img, image_bboxes,full_mask)
        
        loss = criterion(logits, relation_id,relation_half_mask)

        losses.update(loss.cpu().data.numpy())

        #calcalate accuracy
        acc_list,count = cal_acc(logits, relation_id,relation_half_mask)
        for i in range(1 + args.time_steps):
            acces[i].update(acc_list[i],count)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        acc_str = ''
        for i in range(1 + args.time_steps):
            acc_str += ('%.3f ' % acces[i].avg)
        
        if batch_idx % args.print_freq == 0:
            print('Epoch: [%d][%d/%d]  ' 
                 'Time %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                  % (epoch, batch_idx, len(testloader),
                    batch_time.val,batch_time.avg, 
                    losses.val,losses.avg) + acc_str)
            log_file.write('Epoch: [%d][%d/%d]\t' 
                 'Time %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                 % (epoch, batch_idx, len(testloader),
                    batch_time.val,batch_time.avg,
                    losses.val,losses.avg) +acc_str + '\n')
    
    acc_str = ''
    for i in range(1 + args.time_steps):
        acc_str += ('%.3f ' % acces[i].avg)
    print("Test: Acc " + acc_str + '\n')
    log_file.write("Test: Acc " + acc_str + '\n' )
    return acces


print('Start training...')
fout = open('log_info.txt','a')
print("Random Seed is",args.manualSeed)
fout.write('Random Seed is %d\n' % args.manualSeed)

for epoch in range(start_epoch,args.max_epochs):
    print('Epoch: %d start!' % epoch)
    fout.write('Epoch: %d start!\n' % epoch)

    train_epoch(epoch,fout)
    valid_epoch(epoch,fout)
    test_epoch(epoch,fout)
    save_model(SRModel,epoch)