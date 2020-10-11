import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms,models
import numpy as np 
import os
import json
import time
import AverageMeter
import SRDataset
import pair_cnn

#super-parameters
parser = argparse.ArgumentParser(description='PyTorch Social Relation')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--images-root', type=str,
                        help='images root for dataset')
parser.add_argument('--train-data_file', type=str, default='../domain_split/domain_trainidx.txt',
                        help='data file for train dataset')
parser.add_argument('--valid-data_file', type=str, default='../domain_split/domain_valididx.txt',
                        help='data file for valid dataset')
parser.add_argument('--test-data_file', type=str, default='../domain_split/domain_testidx.txt',
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

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
transform_train = transforms.Compose([
    transforms.Resize((256,256)),
    #transforms.RandomRotation(10),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
transform_valid = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
transform_test = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

trainset = SRDataset.SRDataset(image_dir=args.images_root,list_path=args.train_data_file,input_transform=transform_train)
#sampler = trainset.weighted_sampler(args.num_classes)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

validset = SRDataset.SRDataset(image_dir=args.images_root,list_path=args.valid_data_file,input_transform=transform_valid)
validloader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

testset = SRDataset.SRDataset(image_dir=args.images_root,list_path=args.test_data_file,input_transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

##Model prepare
print("Loading model...")
SRModel = pair_cnn.pair_cnn(num_classes=args.num_classes)

SRModel.cuda()
start_epoch = load_model(SRModel)
#start_epoch =  1

SRModel = torch.nn.DataParallel(SRModel)

criterion = nn.CrossEntropyLoss()


def is_fc(para_name):
    split_name = para_name.split('.')
    if (split_name[1] == 'resnet101_union' or split_name[1] == 'resnet101_a'):
        return False
    else:
        return True

params = []
print("FC layers:")
for keys,param_value in SRModel.named_parameters():
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
    acces = AverageMeter.AverageMeter()


    SRModel.train()

    end_time = time.time()
    for batch_idx, (unions, obj1s, obj2s, bposs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end_time)
        unions, obj1s, obj2s, bposs, targets = unions.cuda(), obj1s.cuda(), obj2s.cuda(), bposs.cuda(), targets.cuda()
        unions, obj1s, obj2s, bposs, targets = Variable(unions), Variable(obj1s),Variable(obj2s),Variable(bposs), Variable(targets)

        optimizer.zero_grad()

        logits = SRModel(unions, obj1s, obj2s, bposs)
        
        loss = criterion(logits, targets).cuda()

        loss.backward()
        optimizer.step()
        losses.update(loss.cpu().data.numpy())

        #calcalate accuracy
        t = targets.data.cpu().long().numpy()
        output_f = F.softmax(logits, dim=1)
        output_np = output_f.data.cpu().numpy()
        pred = np.argmax(output_np,axis=1)
        acc = sum(pred == t) / len(pred)
        acces.update(acc)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        if batch_idx % args.print_freq == 0:
            print('Epoch: [%d][%d/%d]  ' 
                 'Time %.3f (%.3f)\t'
                 'Data %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                 'Acc %.3f (%.3f)' % (epoch, batch_idx, len(trainloader),
                    batch_time.val,batch_time.avg, data_time.val,data_time.avg,
                    losses.val,losses.avg,acces.val,acces.avg))
            log_file.write('Epoch: [%d][%d/%d]\t' 
                 'Time %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                 'Acc %.3f (%.3f)\n' % (epoch, batch_idx, len(trainloader),
                    batch_time.val,batch_time.avg,
                    losses.val,losses.avg,acces.val,acces.avg) )


def valid_epoch(epoch,log_file):  
    batch_time = AverageMeter.AverageMeter()
    losses = AverageMeter.AverageMeter()
    acces = AverageMeter.AverageMeter()

    SRModel.eval()

    total = 0
    acc_num = 0
    end_time = time.time()
    for batch_idx, (unions, obj1s, obj2s, bposs, targets) in enumerate(validloader):
        
        unions, obj1s, obj2s, bposs, targets = unions.cuda(), obj1s.cuda(), obj2s.cuda(), bposs.cuda(), targets.cuda()
        unions, obj1s, obj2s, bposs, targets = Variable(unions), Variable(obj1s),Variable(obj2s),Variable(bposs), Variable(targets)

        

        logits = SRModel(unions, obj1s, obj2s, bposs)
        
        loss = criterion(logits, targets).cuda()

        losses.update(loss.cpu().data.numpy())

        #calcalate accuracy
        t = targets.data.cpu().long().numpy()
        output_f = F.softmax(logits, dim=1)
        output_np = output_f.data.cpu().numpy()
        pred = np.argmax(output_np,axis=1)
        acc = sum(pred == t) / len(pred)
        acc_num = acc_num +  sum(pred == t)
        total = total + len(pred)
        acces.update(acc)
        

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        if batch_idx % args.print_freq == 0:
            print('Epoch: [%d][%d/%d]  ' 
                 'Time %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                 'Acc %.3f (%.3f)'  % (epoch, batch_idx, len(validloader),
                    batch_time.val,batch_time.avg,
                    losses.val,losses.avg,acces.val,acces.avg))
            log_file.write('Epoch: [%d][%d/%d]\t' 
                 'Time %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                 'Acc %.3f (%.3f)\n' % (epoch, batch_idx, len(validloader),
                    batch_time.val,batch_time.avg,
                    losses.val,losses.avg,acces.val,acces.avg) )
    total_acc =  acc_num * 1.0 / total

    
    print("Valid: Acc %.3f  " % (total_acc) )
    log_file.write("Valid: Acc %.3f\n" % (total_acc) )



def test_epoch(epoch,log_file):
    
    batch_time = AverageMeter.AverageMeter()
    losses = AverageMeter.AverageMeter()
    acces = AverageMeter.AverageMeter()

    SRModel.eval()

    total = 0
    acc_num = 0
    end_time = time.time()
    for batch_idx, (unions, obj1s, obj2s, bposs, targets) in enumerate(testloader):
        
        unions, obj1s, obj2s, bposs, targets = unions.cuda(), obj1s.cuda(), obj2s.cuda(), bposs.cuda(), targets.cuda()
        unions, obj1s, obj2s, bposs, targets = Variable(unions), Variable(obj1s),Variable(obj2s),Variable(bposs), Variable(targets)

        

        logits = SRModel(unions, obj1s, obj2s, bposs)
        
        loss = criterion(logits, targets).cuda()

        losses.update(loss.cpu().data.numpy())

        #calcalate accuracy
        t = targets.data.cpu().long().numpy()
        output_f = F.softmax(logits, dim=1)
        output_np = output_f.data.cpu().numpy()
        pred = np.argmax(output_np,axis=1)
        acc = sum(pred == t) / len(pred)
        acc_num = acc_num +  sum(pred == t)
        total = total + len(pred)
        acces.update(acc)
  

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        if batch_idx % args.print_freq == 0:
            print('Epoch: [%d][%d/%d]  ' 
                 'Time %.4f (%.4f)\t'
                 'Loss %.3f (%.3f)\t'
                 'Acc %.3f (%.3f)'% (epoch, batch_idx, len(testloader),
                    batch_time.val,batch_time.avg,
                    losses.val,losses.avg,acces.val,acces.avg))
            log_file.write('Epoch: [%d][%d/%d]\t' 
                 'Time %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                 'Acc %.3f (%.3f)\n' % (epoch, batch_idx, len(testloader),
                    batch_time.val,batch_time.avg,
                    losses.val,losses.avg,acces.val,acces.avg) )
    total_acc =  acc_num * 1.0 / total
   
    print("Test: Acc %.3f" % (total_acc) )
    log_file.write("Test: Acc %.3f\n" % (total_acc))


#training
print('Start training...')
fout = open('log_info.txt','a')
for epoch in range(start_epoch,args.max_epochs):
    print('Epoch: %d start!' % epoch)
    fout.write('Epoch: %d start!\n' % epoch)

    #scheduler.step()
    #display_lr(optimizer)
    train_epoch(epoch,fout)
    valid_epoch(epoch,fout)
    test_epoch(epoch,fout)
    save_model(SRModel,epoch)