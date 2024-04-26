
"""
parts of the code are based on the following github codebases, we thank the authors, aknowledge their work, and give them credit for their open source code.

This code is taken from the following sites, modifications were made according to our objectives: 
https://github.com/xudejing/video-clip-order-prediction
https://github.com/xudejing/video-clip-order-prediction/blob/master/train_classify.py
"""

"""Train 3D ConvNets to action classification."""
import os
import argparse
import time

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
from PIL import ImageOps, Image, ImageFilter
from datasets.ds_ucf101 import DSUCF101Dataset

from models.r2plus1d_18 import r2plus1d_18

from utilities.model_saver import Model_Saver
from utilities.model_loader import Model_Loader
from utilities.logger import Logger
from models.s3d import S3D
import utilities.augmentations as A
import utilities.transforms as T

    
    
def train_amp(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, classes, scheduler, scaler):
    torch.set_grad_enabled(True)
    model.train()

    epoch_running_loss=0.0
    epoch_running_corrects=0
    
    running_loss = 0.0
    correct = 0
    
    #targets_counts=[0 for x in (classes)]
    i, train_bar = 1, tqdm(train_dataloader)
    for clips, idxs in train_bar:
    #for i, data in (enumerate(train_dataloader, 1)):
        # get inputs
        #clips, idxs = data
        inputs = clips.to(device)
        targets = idxs.to(device)
        #print('targets.shape:',targets.shape)
        
        #targets_counts=targets_info(targets, targets_counts, None, None, classes)
        
        # zero the parameter gradients
        optimizer.zero_grad(set_to_none=True)
        
        # forward and backward
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs) # return logits here
            #print('outputs.shape:',outputs.shape)
            assert outputs.dtype is torch.float16
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # compute loss and acc
        # running_loss += loss.item()  # I think its not accurate
        running_loss += loss.item()* inputs.size(0)
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        #print(pts,':::',targets)
        
        
        # stats for the complete epoch
        epoch_running_loss += loss.item() * inputs.size(0)
        epoch_running_corrects+=torch.sum(targets == pts).item()
        
        if args.sch_mode=='one_cycle':
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
    #epoch_targets= targets_info(None, targets_counts, 'final', len(train_dataloader.dataset), classes)
    
    return epoch_acc, epoch_loss, 0.0 #epoch_targets



def test_backup(model, criterion, device, test_dataloader, classes):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    
    #targets_counts=[0 for x in (classes)]
    i, test_bar = 1, tqdm(test_dataloader)
    for clips, idxs in test_bar:
    #for i, data in (enumerate(test_dataloader, 1)):
        # get inputs
        #clips, idxs = data
        inputs = clips.to(device)
        targets = idxs.to(device)
        
        #targets_counts=targets_info(targets, targets_counts, None, None, classes)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        # total_loss += loss.item() # I think this is not accurate
        total_loss += loss.item()* inputs.size(0)
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
        test_bar.set_description('[TEST]: loss: {:.4f}, corrects {}, acc: {:.4f}'.format( total_loss/(i*args.bs), correct,  correct/(i*args.bs)))
        i += 1
        torch.cuda.empty_cache()
        
    #avg_loss = total_loss / len(test_dataloader)   # I think this is not accurate
    avg_loss = total_loss / len(test_dataloader.dataset)
    avg_acc = correct / len(test_dataloader.dataset)
    #avg_targets=targets_info(None, targets_counts, 'final', len(test_dataloader.dataset), classes) 
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_acc, avg_loss, 0.0 #, avg_targets 


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
        param_group['lr'] = lr
        #param_group['initial_lr'] = lr
        param_group['weight_decay'] = wd 
        #param_group['momentum'] = args.momentum
       

def parse_args():
    
    
    experiment='test' 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='r2plus1d_18', help='s3d/r2plus1d_18')                          
    parser.add_argument('--dataset', type=str, default='dsucf101', help='ucf101')
    parser.add_argument('--split', type=str, default='1', help='dataset split 1,2,3')
    parser.add_argument('--cl', type=int, default=16, help='clip length')                                         
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
    
    #parser.add_argument('--ckpt', type=str, help='checkpoint path')
    #parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=400, help='number of total epochs to run')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=16, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    #parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    
   
    parser.add_argument('--sch_mode', type=str, default='reduce_lr', help='one_cycle/reduce_lr/None.')                
   
    parser.add_argument('--log', type=str, help='log directory')
    parser.add_argument('--use_amp', type=bool, default=True, help='True/False.')
    parser.add_argument('--run_testing', type=bool, default=True, help='True/False.')
    
    parser.add_argument('--exp_name', type=str, default=experiment, help='experiment name.')
    #parser.add_argument('--pretrained_model', type=str, default=None, help='the name of the pretrained model.')
    #parser.add_argument('--finetuning class num', type=int, default=4, help='number of classes during finetuning.')
    
    
    parser.add_argument('--r_frames', type=int, default=4, help='the number of frames to be repeated.')
    parser.add_argument('--n_frames', type=int, default=12, help='the number of non-repeated frames.')
    parser.add_argument('--offset', type=int, default=4, help='the number of frames between the repeated and the original scenes.')
    parser.add_argument('--select_mode', type=str, default='fixed', help='fixed/random.')
    parser.add_argument('--ins_mode', type=str, default='faraway', help='random,  offset>0 and mode="faraway", offset>0 and mode="close_by".')
    parser.add_argument('--minimum_p', type=float, default=0.25, help='minimum probablity to generate duplicats on the fly.')
    parser.add_argument('--sampling_mode', type=str, default='random_skip', help='random_skip/fixed_skip/sequential.')
    parser.add_argument('--skip_rate', type=int, default=4, help='1..4.')
    parser.add_argument('--init_mode', type=str, default='kaiming', help='kaiming/None')
    
    
    args = parser.parse_args()
    
    

   
    
 
    return args


def main(args):
    print_sep = '============================================================='
    
    if (args.select_mode=='fixed' and args.r_frames==2 and args.cl == 16):
       classes= [0,1,2,3,4,5,6,7] #[0,2,4,6,8,10,12,16] Classes:8
       
    elif (args.select_mode=='fixed' and args.r_frames==4 and args.cl == 16):
       classes= [0,1,2,3] #[0,4,8,16]                                 #16=>No Repeated Frames
       
    elif (args.select_mode=='fixed' and args.r_frames==8 and args.cl == 16):
       classes= [0,1] #[0,16] Classes:2 
    
    elif (args.select_mode=='fixed' and args.r_frames==4 and args.cl == 32):
       classes= [0,1,2,3,4,5,6,7] #[0,4,8,12,16,20,24,32] Classes:8
     
    elif (args.select_mode=='fixed' and args.r_frames==8 and args.cl == 32):
       classes= [0,1,2,3] #[0,8,16,32] Classes:4
       
    elif (args.select_mode=='fixed' and args.r_frames==16 and args.cl == 32):
       classes= [0,1] #[0,32] Classes:2    
    
    elif (args.select_mode=='fixed' and args.r_frames==8 and args.cl == 64 ):
        classes =  [0,1,2,3,4,5,6,7] #[0,8,16,24,32,40,48,64]               #64=>No Repeated Frames
        
    elif ((args.select_mode=='fixed' and args.r_frames==16 and args.cl == 64 )):
        classes = [0,1,2,3]  #[0,16,32,64]
  
        
                
                
    ###########################################################################
    
    current_dir=os.getcwd().replace('C:','')
    args.log=os.path.join(current_dir, 'experiments', args.exp_name,'full_400_epochs')
    
    ucf_dir=r''

    
   

    ###torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")



    ########### model ##############

    if args.dataset == 'dsucf101':
        class_num = len((classes))
        print('Number Of Targets:',class_num)


    if  args.model == 'r2plus1d_18':
        model = r2plus1d_18( num_classes = class_num).to(device)
        

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
            
    if (args.init_mode=='kaiming')and((args.model=='s3d')):
        print('applying kaiming normal init...')
        model.apply(kaiming_init)
      
    print (print_sep)    
#    total_params = sum(p.numel() for p in model.parameters())
#    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#    print ('Total Params', total_params, ':: ::', 'Trainable',total_trainable_params)
    print (print_sep) 
    

    if  args.mode == 'train':  ########### Train #############
        
    
    
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        #torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        #optimizer = optim.AdamW(model.parameters())
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
      
        # patience was 50 I changed it to 20
        # patience was 20 I changed it to 10
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=10, factor=0.1)

        
        # resume training # check if there was a previously saved checkpoint
        # resume training # check if there was a previously saved checkpoint
        loader = Model_Loader(args, model, optimizer)
        model, optimizer, epoch_resume, starting_epoch, best_video_acc, best_video_loss = loader.load()
        # resume training # check if there was a previously saved checkpoint
        
       
            
        #(0.4218, 0.4025, 0.3738) - (0.2337, 0.2267, 0.2240)
           
        train_transforms =torchvision.transforms.Compose([T.ToTensorVideo(),
                                               T.Resize((128, 171)),
                                               #T.RandomResizedCropVideo((112,112)),
                                               A.RandomSizedCrop(112, interpolation=Image.BICUBIC, consistent=False, p=1.0, seq_len=4, bottom_area=0.2),
                                               A.RandomHorizontalFlip(consistent=False, command=None, seq_len=6),
                                               transforms.RandomApply([A.RandomGray(consistent=False, p=0.5, dynamic=False, seq_len=4)], p=0.7),
                                               transforms.RandomApply([A.ColorJitter(brightness=0.3, contrast=1, saturation=0.3, hue=0.3, consistent=False, p=1.0, seq_len=4)], p=0.7),
                                               transforms.RandomApply([A.GaussianBlur(sigma=[.1, 2.], seq_len=4)],p=0.7),
                                               #T.NormalizeVideo([0.4218, 0.4025, 0.3738],[0.2337, 0.2267, 0.2240])
                                               
                                                          
        ])
        
        val_transforms = torchvision.transforms.Compose([T.ToTensorVideo(),
                                               T.Resize((128, 171)),
                                               A.RandomSizedCrop(112, interpolation=Image.BICUBIC, consistent=False, p=1.0, seq_len=4, bottom_area=0.2),
                                               A.RandomHorizontalFlip(consistent=False, command=None, seq_len=6),
                                               transforms.RandomApply([A.RandomGray(consistent=False, p=0.5, dynamic=False, seq_len=4)], p=0.7),
                                               transforms.RandomApply([A.ColorJitter(brightness=0.3, contrast=1, saturation=0.3, hue=0.3, consistent=False, p=1.0, seq_len=4)], p=0.7),
                                               transforms.RandomApply([A.GaussianBlur(sigma=[.1, 2.], seq_len=4)],p=0.7),
                                               ###T.CenterCropVideo((112,112)),
                                               #T.NormalizeVideo([0.4218, 0.4025, 0.3738],[0.2337, 0.2267, 0.2240])
        ])

        
        
        if args.dataset == 'dsucf101':
            train_dataset = DSUCF101Dataset(args, ucf_dir, True, train_transforms)

            val_dataset=DSUCF101Dataset(args, ucf_dir, False, val_transforms)
            


        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

        total_batches=0
        if (epoch_resume>1):
            total_batches=(epoch_resume-1)*len(train_dataloader)
        print (print_sep)     
        print('Completed Epochs     :: ', epoch_resume-1)
        print('len(train_dataloader)::', len(train_dataloader))    
        print('Scheduler Batch      ::', total_batches)
        print (print_sep) 
        
        if (args.sch_mode=='one_cycle'):
            if (total_batches!=0):
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, last_epoch= total_batches, verbose=False)
            else:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, last_epoch=-1, verbose=False)
        else:
            if (args.sch_mode=='reduce_lr'):
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-6, patience=10, factor=0.1)
            else:
                if (args.sch_mode=='None'):
                    scheduler=None
                    
        
        
        adjust_learning_rate(optimizer, None, args)
        print(print_sep)
        print ('Using Scheduler ::', scheduler)
        print ('Using Optim     ::', optimizer)
        print(print_sep)
        
        writer=None
        for epoch in range(starting_epoch, args.start_epoch + args.epochs):
            print('Epoch::',epoch)
            time_start = time.time()
            if (args.use_amp == True):
               train_acc, train_loss, epoch_targets=train_amp(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, classes, scheduler, scaler)
            else:
               print ('AMP Should Be Used !!! ') 
               #train_acc, train_loss, epoch_targets=train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, classes, scheduler) 
           
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            #val_acc, val_loss = validate(model, criterion, device, val_dataloader, writer, epoch)
            if args.run_testing == True and epoch % 10 == 0 :
                val_acc, val_loss, test_targets = test_backup(model, criterion, device, val_dataloader, classes)
            else:
                val_acc=0.0
                val_loss=0.0
                test_targets=0.0
                
                
            
            if (args.sch_mode=='reduce_lr'):
                scheduler.step(train_loss)
           
            
            # save model every 20 epoches
            if epoch % 1 == 0:
                #torch.save(model.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
                #save_model(args.log, args.exp_name, model, optimizer, epoch, train_acc, train_loss, batch_count=None,best_mode=1)
                saver = Model_Saver(args, model, optimizer, epoch, train_acc, train_loss, batch_count=None, best_mode=1)
                saver.save()
                
                
                logger = Logger(args, epoch, train_acc, val_acc, train_loss, val_loss, None)
                logger.acc_log()
             

                
            # save model for the best val
            if val_acc > best_video_acc:

                
                saver = Model_Saver(args, model, optimizer, epoch, val_acc, val_loss, batch_count=None, best_mode=3)
                saver.save()
                best_video_acc = val_acc 
                


    elif args.mode == 'test':  ########### Test #############
        #
        best_test_ckpt=os.path.join(args.log, ('Best-Video-'+args.exp_name))+'.tar'
        checkpoint=torch.load(best_test_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Loading The Best Video Model From Epoch::', checkpoint['epoch'], ' With Val Acc:',checkpoint['acc'])
        
        
        
        test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()
        ])

        if args.dataset == 'dsucf101':
            test_dataset = DSUCF101Dataset(args, ucf_dir, False, test_transforms)
        #elif dataset == 'hmdb51':
        #    test_dataset = HMDB51Dataset('data/hmdb51', cl, split, False, test_transforms, 10)

        if (args.cl==64):
            effective_bs = 1
        else:
            effective_bs = args.bs
            
        test_dataloader = DataLoader(test_dataset, batch_size=effective_bs, shuffle=False, num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test_backup(model, criterion, device, test_dataloader)
        
   

        
if __name__ == '__main__':
    args = parse_args()
    print(vars(args))
    
    main(args)