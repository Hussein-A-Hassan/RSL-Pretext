
"""
parts of the code are based on the following github codebases, we thank the authors, aknowledge their work, and give them credit for their open source code. 
https://github.com/xudejing/video-clip-order-prediction
https://github.com/xudejing/video-clip-order-prediction/blob/master/train_classify.py
"""

"""Train 3D ConvNets to action classification."""
import os
import argparse
import time
import random
import copy
import numpy as np
import pandas as pd
import torch
import torchvision
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim

import tqdm
from tqdm import tqdm

from datasets.ucf101 import UCF101Dataset
from datasets.hmdb51 import HMDB51Dataset
from models.r2plus1d_18 import r2plus1d_18
from models.s3d import S3D

from utilities.model_saver import Model_Saver
from utilities.model_loader import Model_Loader
from utilities.logger import Logger
import utilities.augmentations as A
import utilities.transforms as T


    
def train_amp(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, scheduler, scaler):
    torch.set_grad_enabled(True)
    model.train()

    epoch_running_loss=0.0
    epoch_running_corrects=0
    
    running_loss = 0.0
    correct = 0
    i, train_bar = 1, tqdm(train_dataloader)
    for clips, idxs in train_bar:
    #for i, data in (enumerate(train_dataloader, 1)):
        # get inputs
        #clips, idxs = data
        inputs = clips.to(device)
        targets = idxs.to(device)
        
        if args.grad_accum==False:
            # zero the parameter gradients
            optimizer.zero_grad()
        
        # forward and backward
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs) # return logits here
            #print (outputs.shape)
            assert outputs.dtype is torch.float16
            loss = criterion(outputs, targets)
            
            ###if args.grad_accum==True:
            ###    loss = loss / args.accum_iter
        
        scaler.scale(loss).backward()
        
        if args.grad_accum==False: 
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_accum==True:
                if ((i + 1) % args.accum_iter == 0) or (i + 1 == len(train_dataloader)):
                    scaler.step(optimizer)
                    scaler.update()
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    if args.sch_mode=='one_cycle' or args.sch_mode=='Cosine' :
                        scheduler.step()

        
        #loss.backward()
        #optimizer.step()
        
        # compute loss and acc
        # running_loss += loss.item()  # I think its not accurate
        running_loss += loss.item()* inputs.size(0)
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        
        
        
        # stats for the complete epoch
        epoch_running_loss += loss.item() * inputs.size(0)
        epoch_running_corrects+=torch.sum(targets == pts).item()
        
        
        
        if (args.sch_mode=='one_cycle' or args.sch_mode=='Cosine') and args.grad_accum==False :
            scheduler.step()
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            # avg_loss = running_loss / pf # I think its not accurate
            avg_loss = running_loss / (args.pf * args.bs)
            avg_acc = correct / (args.pf * args.bs)
            #print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
            print('\n')
            train_bar.set_description('[TRAIN]: [{}/{}], lr: {:.10f}, loss: {:.4f}, , acc: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], avg_loss, avg_acc))
            #step = (epoch-1)*len(train_dataloader) + i
            ###writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            ###writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0
            
        i += 1
        torch.cuda.empty_cache()
            
    epoch_loss = epoch_running_loss / len(train_dataloader.dataset)
    epoch_acc  = epoch_running_corrects / len(train_dataloader.dataset)
    ###print('epoch_loss :', epoch_loss)
    return epoch_acc, epoch_loss
    
    


def test(model, criterion, device, test_dataloader, use_amp):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    i, test_bar = 1, tqdm(test_dataloader)
    for sampled_clips, idxs in test_bar:
    #for i, data in (enumerate(test_dataloader, 1)):
        # get inputs
        #sampled_clips, idxs = data
        targets = idxs.to(device)
        outputs = []
        for clips in sampled_clips:
            inputs = clips.to(device)
            # forward
            if (use_amp==True):
                with torch.cuda.amp.autocast():
                    o = model(inputs)
            else:
                o = model(inputs)
                
            # print(o.shape)
            o = torch.mean(o, dim=0)
            # print(o.shape)
            # exit()
            outputs.append(o)
        outputs = torch.stack(outputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        #total_loss += loss.item()  # I think this is not accurate
        
        total_loss += loss.item()*sampled_clips.size(0)
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        
        
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
        #if i % args.pf == 0:
            #print('[TEST] batch-{}, running loss: {:.3f}, corrects: {:.3f}'.format(i, total_loss/(i*args.bs), correct))
        
        test_bar.set_description('[TEST]: loss: {:.4f}, corrects {}, acc: {:.4f}'.format( total_loss/(i*args.bs), correct,  correct/(i*args.bs)))
        i += 1
        torch.cuda.empty_cache()
        
        
    #avg_loss = total_loss / len(test_dataloader) # I think this is not accurate
    avg_loss = total_loss / len(test_dataloader.dataset)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}, total corrects: {:.3f}'.format(avg_loss, avg_acc, correct))
    return avg_acc ,avg_loss




# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    wd = args.wd
    #if args.cos:  # cosine lr schedule
    #    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    #else:  # stepwise lr schedule
    #    for milestone in args.schedule:
    #        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        #param_group['lr'] = lr
        #param_group['initial_lr'] = lr
        param_group['weight_decay'] = wd 
        #param_group['momentum'] = args.momentum

        
def parse_args():
    
 
    experiment='test' 
    parser = argparse.ArgumentParser(description='Video Classification')
    parser.add_argument('--mode', type=str, default='train', help='train/test/')
    parser.add_argument('--model', type=str, default='s3d', help='/s3d/r2plus1d_18/')
    parser.add_argument('--dataset', type=str, default='hmdb51', help='ucf101/hmdb51')
    parser.add_argument('--split', type=str, default='1', help='dataset split 1,2,3')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')  #1e-3 #3e-7 6e-7 1e-6 3e-6 [Good 3e-5]
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum') #default=9e-1   #99e-2
    parser.add_argument('--wd', type=float, default=1e-3 , help='weight decay')  #5e-4 #1e-5 #1e-4 #1e-3 #1e-2  #0  4e-5 #2e-3
    

    parser.add_argument('--epochs', type=int, default=401, help='number of total epochs to run')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=16, help='mini-batch size')                    
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')

    
   
    parser.add_argument('--sch_mode', type=str, default='one_cycle', help='one_cycle/reduce_lr/None/Cosine. .')
   
    parser.add_argument('--log', type=str, help='log directory')
    parser.add_argument('--use_amp', type=bool, default=True, help='True/False.')
    parser.add_argument('--run_testing', type=bool, default=True, help='True/False.')  
    
    parser.add_argument('--exp_name', type=str, default=experiment, help='experiment name.')
    parser.add_argument('--pretrained_model', type=str, default='rsl_s3d_exp1_e200.tar', help='the name of the pretrained model.') 
    parser.add_argument('--finetuning_class_num', type=int, default=4, help='number of classes during finetuning.')
    
    parser.add_argument('--sampling_mode', type=str, default='fixed_skip', help='random_skip/fixed_skip/sequential.')
    parser.add_argument('--skip_rate', type=int, default=4, help='1..4.')
    parser.add_argument('--init_mode', type=str, default='kaiming', help='kaiming/None')
    parser.add_argument('--test_mode', type=str, default='ten_clips', help='ten_clips/')
    parser.add_argument('--dual_testing', type=bool, default=False, help='True/False.')
    parser.add_argument('--auto_testing', type=bool, default=False, help='True/False.')
    
    
    
    #https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
    parser.add_argument('--grad_accum', type=bool, default=False, help='')
    parser.add_argument('--accum_iter', type=int, default = 8, help='1...')   #The total batchs has to be divisible by this number

    
    

    parser.add_argument('--ssl_class_num', type=int, default=4, help='ssl,psl class number')
    parser.add_argument('--rsl_class_num', type=int, default=4, help='the rsl_class_num.')
    
    parser.add_argument('--tuple_len', type=int, default=3, help='the number of clips sampled in each tuple.')
    parser.add_argument('--pretext_mode', type=str, default='ssl-forward', help='ssl-forward/ssl-backward/ssl-mixed/sspl/ssl-rsl/None')
    
    parser.add_argument('--device', type=str, default='1', help='Device')
    
    args = parser.parse_args()
    
 
    return args


def main_program(args):
    

    
    print_sep = '============================================================='
    
    ###torch.autograd.detect_anomaly()
    current_dir=os.getcwd().replace('C:','')

    
    ucf_dir=r''
    hmdb_dir=r''
    
    
    args.log=os.path.join(current_dir, 'experiments', args.exp_name, 'Run')
    
    if args.pretrained_model!=None:
       finetuning_model=os.path.join(args.log,args.pretrained_model)
    

    ###torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    ###os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    

    ########### model ##############
    if args.dataset == 'ucf101':
        class_num = 101
    elif args.dataset == 'hmdb51':
        class_num = 51

    if (args.pretrained_model == 'None'):
 
        if  args.model == 'r2plus1d_18':
            model = r2plus1d_18( num_classes = class_num, return_conv = False).to(device)
            
        elif  args.model == 's3d':
            model = S3D(class_num).to(device)
        
            def kaiming_init(m):
        
               if isinstance(m, nn.Conv3d):
                   nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                   if m.bias is not None:
                       nn.init.constant_(m.bias, 0)
               elif isinstance(m, nn.BatchNorm3d):
                   nn.init.constant_(m.weight, 1)
                   nn.init.constant_(m.bias, 0)
               elif isinstance(m, nn.Linear):
                   nn.init.normal_(m.weight, 0, 0.01)
                   nn.init.constant_(m.bias, 0)
            if (args.init_mode=='kaiming') and (args.model=='s3d'):
                model.apply(kaiming_init)
                print('applying kaiming normal init...')
                
        elif (args.model == 'resnet_18'):
            print('Using resnet_18 Model !!!!')
   #         model = ResNGen.generate_model(18,features_output='False', with_classifier='True').to(device)
        
            
    else:
        if os.path.exists(finetuning_model):
                
            if (args.model == 's3d'):
                model = S3D(args.finetuning_class_num).to(device)
                print("Loading From Self-Supervised Trained Model-Dropout3d is used !!! ")
                checkpoint = torch.load(finetuning_model)
                model.load_state_dict(checkpoint['model_state_dict'])
                print ('epoch :', checkpoint['epoch'])
                in_channels=model.fc[0].in_channels
                #model.fc = nn.Sequential(nn.Conv3d(in_channels, class_num, kernel_size=1, stride=1, bias=True),)
                model.fc = nn.Sequential(nn.Conv3d(in_channels, 512, kernel_size=1, stride=1, bias=True),
                                         nn.Dropout3d(0.8),
                                         nn.Conv3d(512, class_num, kernel_size=1, stride=1, bias=True)
                                         )
                
                if isinstance(model.fc[0], nn.Conv3d):
                    print('applying kaiming normal init to the last layer...')
                    nn.init.kaiming_normal_(model.fc[0].weight, mode='fan_out', nonlinearity='relu')
                    if model.fc[0].bias is not None:
                        nn.init.constant_(model.fc[0].bias, 0)
                        
                model=model.to(device)
                print('Model Last Layer::', model.fc)
                
            elif  (args.model == 'r2plus1d_18'):
                model = r2plus1d_18( num_classes =args.finetuning_class_num, return_conv = False)
                print(print_sep)
                print("Loading From Previously Self-Supervised Trained Model-Dropout3d is used !!!")
                checkpoint = torch.load(finetuning_model)
                print ('epoch :', checkpoint['epoch'])
                ###print (model)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                
                in_channels=model.fc[0].in_features
                print('in_channels',in_channels)
                model.fc = nn.Sequential(nn.Linear(in_channels, 256),
                                         nn.Dropout(0.9),
                                         nn.Linear(256, class_num)
                                         )
                model=model.to(device)
                print('Model Last Layer::', model.fc)
            
                
        else:
            print(print_sep)
            print('Loading New Model !!!!')
                
            if (args.model == 's3d'):
                model = S3D(class_num).to(device)
                in_channels=model.fc[0].in_channels
                #model.fc = nn.Sequential(nn.Conv3d(in_channels, class_num, kernel_size=1, stride=1, bias=True),)
                model.fc = nn.Sequential(nn.Conv3d(in_channels, 512, kernel_size=1, stride=1, bias=True),
                                         nn.Dropout3d(0.8),
                                         nn.Conv3d(512, class_num, kernel_size=1, stride=1, bias=True)
                                         )
                
                if isinstance(model.fc[0], nn.Conv3d):
                    print('applying kaiming normal init to the last layer...')
                    nn.init.kaiming_normal_(model.fc[0].weight, mode='fan_out', nonlinearity='relu')
                    if model.fc[0].bias is not None:
                        nn.init.constant_(model.fc[0].bias, 0)
                        
                model=model.to(device)
                print('Model Last Layer::', model.fc)
                
            if (args.model == 'r2plus1d_18'):
                 model = r2plus1d_18( num_classes =class_num, return_conv = False)
                 in_channels=model.fc[0].in_features
                 model.fc = nn.Sequential(nn.Linear(in_channels, 256),
                                         nn.Dropout(0.9),
                                         nn.Linear(256, class_num)
                                         )
                 model=model.to(device)
                 print('Model Last Layer::', model.fc)
          
            

    if args.mode == 'train':  ########### Train #############
        
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()
        ###optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        
        #torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        ###optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
             
        ### resume training # check if there was a previously saved checkpoint
        loader = Model_Loader(args, model, optimizer)
        model, optimizer, epoch_resume, starting_epoch, best_video_acc, best_video_loss = loader.load()
        ### resume training # check if there was a previously saved checkpoint
        
     


        train_transforms =torchvision.transforms.Compose([T.ToTensorVideo(),
                                               
                                               T.Resize((128,171)),                                              
                                              
                                               transforms.RandomApply([A.ColorJitter([0.6,1.6],[0.7,1.3],[0.7,1.3],[0,0.2], p=1, consistent=False, seq_len=args.cl)], p=0.5),
                                               T.RandomHorizontalFlipVideo(),
                                               T.RandomCropVideo((112,112)),
                              
        ])
        
        val_transforms = torchvision.transforms.Compose([T.ToTensorVideo(),
                                         
                                               T.Resize((128, 171)),
                                               T.CenterCropVideo((112,112)),
                                            
        ])

        if args.dataset == 'ucf101':
            train_dataset = UCF101Dataset(args, ucf_dir, True, train_transforms)

            val_dataset=UCF101Dataset(args, ucf_dir, False, val_transforms, 10)
            
        elif args.dataset == 'hmdb51':
             train_dataset = HMDB51Dataset(args, hmdb_dir, True, train_transforms)
             val_dataset = HMDB51Dataset(args, hmdb_dir, False, val_transforms, 10)

        
        print(print_sep)
        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        print(print_sep)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                    num_workers=args.workers, pin_memory=True, drop_last=True)
        if (args.cl==64):
            effective_bs = 1
        else:
            effective_bs = args.bs
            
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)


        total_batches=0
        if (epoch_resume>1):
            total_batches=(epoch_resume-1)*len(train_dataloader)
        print('Scheduler Batch :',total_batches)
        if (args.sch_mode=='one_cycle'):
            if (total_batches!=0):
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, last_epoch= total_batches, verbose=False)
            else:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, last_epoch=-1, verbose=False)
        
        elif(args.sch_mode=='reduce_lr'):
            #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-8, patience=8, factor=0.1)
        
        elif(args.sch_mode=='Cosine'):
            
            #adjust_learning_rate(optimizer, 1, args)
            #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2 * int((len(train_dataloader)/(args.accum_iter))))
            #print('optimizer.param_groups[0][]',optimizer.param_groups[0]['lr'])
            #print(scheduler.T_max)
    
        elif(args.sch_mode=='None'):
            print('scheduler is deleted')
            #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
            scheduler=None
           
            
        ###adjust_learning_rate(optimizer, None, args)
        print(print_sep)
        print ('Using Scheduler ::', scheduler)
        print ('Using Optim     ::', optimizer)
        print(print_sep)
        
        writer=None
        for epoch in range(starting_epoch, args.start_epoch+args.epochs):
            print('epoch::',epoch)
            
            time_start = time.time()
            
            if (args.use_amp == True):
                train_acc, train_loss=train_amp(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, scheduler, scaler)
            # else:
                # train_acc, train_loss=train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, scheduler)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            #val_acc, val_loss = validate(model, criterion, device, val_dataloader, writer, epoch)
            
            if train_acc >= 0.94 and args.auto_testing ==True and  args.run_testing == False :
                args.run_testing = True

            
            
            if (args.run_testing==True and epoch % 1 == 0):
                time_start = time.time()
                if (args.test_mode == 'ten_clips'):
                    val_acc, val_loss = test(model, criterion, device, val_dataloader, args.use_amp)

                print('Testing time: {:.2f} s.'.format(time.time() - time_start))
                
                if (args.dual_testing==True):
                    args.test_mode = 'all_seq'
                    #args.sampling_mode = 'fixed_skip'
                    if args.dataset == 'ucf101':
                        dual_val_dataset=UCF101Dataset(args, ucf_dir, False, val_transforms, 10)
                    elif args.dataset == 'hmdb51':
                        dual_val_dataset = HMDB51Dataset(args, hmdb_dir, False, val_transforms, 10)
                    dual_val_dataloader = DataLoader(dual_val_dataset, batch_size=1, shuffle=False,num_workers=args.workers, pin_memory=True)
                    print('Dual Testing is On ...')
                    time_start = time.time()
                    dual_val_acc, dual_val_loss = test(model, criterion, device, dual_val_dataloader, args.use_amp)
                    print('Dual Testing time: {:.2f} s.'.format(time.time() - time_start))
                    args.test_mode = 'ten_clips'
                    #args.sampling_mode = 'sequential'
                else:
                    if (args.test_mode != 'ten_and_1010'):
                        dual_val_acc = 0
                        dual_val_loss = 0
                         
            else:
                val_acc=0
                val_loss=0
                dual_val_acc = 0
                dual_val_loss = 0
                
                
            
            if (args.sch_mode=='reduce_lr'):
                scheduler.step(train_loss)
            
            ###writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            
            # save model every 20 epoches
            if epoch % 1 == 0:

                saver = Model_Saver(args, model, optimizer, epoch, train_acc, train_loss, batch_count=None, best_mode=1)
                saver.save()

                logger = Logger(args, epoch, train_acc, val_acc, train_loss, val_loss, None)
                logger.acc_log()

                
            # save model for the best val
            if (val_acc > best_video_acc) or (dual_val_acc > best_video_acc):

                if (val_acc >= dual_val_acc ):
                    top_acc = val_acc
                    best_video_acc = val_acc
                    
                    saver = Model_Saver(args, model, optimizer, epoch, top_acc, val_loss, batch_count=None, best_mode=3)
                    saver.save()
                
                elif (dual_val_acc > val_acc  ):
                     top_acc = dual_val_acc
                     best_video_acc = dual_val_acc
                     
                     saver = Model_Saver(args, model, optimizer, epoch, top_acc, val_loss, batch_count=None, best_mode=4)
                     saver.save()
               
                     

            if epoch % 2 ==0 and args.sch_mode =='Cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2 * int((len(train_dataloader)/(args.accum_iter))))

    elif args.mode == 'test':  ########### Test #############
        #
        ckpt=os.path.join(args.log, ('Best-Video-'+args.exp_name))+'.tar'
        ckpt = r''  
        checkpoint=torch.load(ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        
        print('Loading The Best Clip Model From Epoch::', checkpoint['epoch'], ' With Val Acc:',checkpoint['acc'])
        
        val_transforms = torchvision.transforms.Compose([T.ToTensorVideo(),
                                                         T.Resize((256, 256)),
                                                         T.CenterCropVideo((224,224)),
                                               

        ])

        if args.dataset == 'ucf101':
            test_dataset = UCF101Dataset(args, ucf_dir, False, val_transforms, 10)
        elif args.dataset == 'hmdb51':
            test_dataset = HMDB51Dataset(args, hmdb_dir, False, val_transforms, 10)

        if (args.cl==64):
            effective_bs = 1
        else:
            effective_bs = args.bs
            
        test_dataloader = DataLoader(test_dataset, batch_size=effective_bs, shuffle=False, num_workers=args.workers, pin_memory=True)
        #test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        val_acc, val_loss = test(model, criterion, device, test_dataloader, args.use_amp) 
        print(f'Testing is Done : Acc {val_acc} : Loss {val_loss}')

        
if __name__ == '__main__':
    
    torch.set_warn_always(False)
    
    args = parse_args()
    print(vars(args))
    main_program(args)