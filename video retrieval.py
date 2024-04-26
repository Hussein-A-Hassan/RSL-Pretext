
"""
parts of the code are based on the following github codebases, we thank the authors, aknowledge their work, and give them credit for their open source code.

# https://github.com/xudejing/video-clip-order-prediction/blob/master/retrieve_clips.py


"""


"""Video retrieval experiment, top-k."""
import os
import math
import itertools
import argparse
import time
import random
import json

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances



from datasets.ucf101_clip_retrieval_dataset import UCF101ClipRetrievalDataset
from datasets.hmdb51_clip_retrieval_dataset import HMDB51ClipRetrievalDataset
import utilities.augmentations as A
import utilities.transforms as T

from models.r2plus1d_18 import r2plus1d_18

from os import listdir
from os.path import isfile, join

from models.s3d import S3D

def load_model (model):
     print('===========================================================================')
     print("Loading From Previously Self-Supervised Trained Model-Dropout3d is used !!!")
     print('===========================================================================')
     checkpoint = torch.load(args.ckpt)
     model.load_state_dict(checkpoint['model_state_dict'], strict=True)
     print('Epoch:',checkpoint['epoch'] )
     print('Acc:', checkpoint['acc'])
     #for l in checkpoint.keys():
     #    print (l)
     print('===========================================================================')
     print('Model Last Layer::', model.fc)
     
     return model


def load_pretrained_weights_s3d(ckpt_path, model):
    #pretrained = torch.load(saved_model_file)
    #pretrained_kvpair = pretrained['state_dict']
    # print(pretrained_kvpair)
    # exit()
    #model_kvpair = model.state_dict()

    # for layer_name, weights in pretrained_kvpair.items():
    #     if 'module.1.' in layer_name:
    #         continue
    #     elif '1.' == layer_name[:2]:
    #         continue
    #     print(layer_name)
    #     break
    # for layer_name, weights in model_kvpair.items():
    #     print(layer_name)
    #     break
    # exit()
    
    """load pretrained weights and adjust params name."""
    print ('transfering the weights !!!')
    adjusted_weights = {}
    pretrained_weights = (torch.load(ckpt_path))['model_state_dict']
    for name, params in pretrained_weights.items():

        if 'base.14' in name or 'base.15' in name:
            print (name)
            continue
        else:
            #name = name[name.find('.')+1:]
            adjusted_weights[name] = params
            #print('Pretrained weight name: [{}]'.format(name))
    
    model.load_state_dict(adjusted_weights, strict=True)
    return adjusted_weights, model

#=========================================================================
    
def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path)
    for name, params in pretrained_weights.items():
        if 'base_network' in name:
            name = name[name.find('.')+1:]
            adjusted_weights[name] = params
            print('Pretrained weight name: [{}]'.format(name))

    return adjusted_weights


def extract_feature(args):
    """Extract and save features for train split, several clips per video."""
    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    print_sep = '============================================================='
    ########### model ##############
    if args.model == 'c3d':
        print()
        #model = C3D(with_classifier=False, return_conv=True).to(device)
    elif args.model == 'r3d':
        print()
        #model = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False, return_conv=True).to(device)
    elif args.model == 'r21d':
        print()
        #model = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False, return_conv=True).to(device)
        
    elif  args.model == 'r2plus1d_18':
            model = r2plus1d_18(num_classes =args.finetuning_class_num, return_conv = True).to(device)
            
    elif args.model == 's3d':
        model = S3D(args.finetuning_class_num, return_conv = True).to(device)
        
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
      
        

    if args.ckpt and  (args.model != 'sspl' and args.model != 'ModelMoCo' and args.model != 'ssl_rsl' and args.model != 'ssl-multi' and args.model != 'ic_resnet18' and args.model != 'enhanced_ic_resnet18' and args.model != 'multi_level_tclr' and args.model != 'tclr'):

         model = load_model(model)
         model = model.to(device)

    model.eval()
    torch.set_grad_enabled(False)
    ### Exract for train split ###
    train_transforms = transforms.Compose([
        T.ToTensorVideo(),
        
        #T.Resize((256, 256)),
        T.Resize((128, 171)),
        T.CenterCropVideo((112,112)),
        #T.CenterCropVideo((128,128)),
        #T.CenterCropVideo((224,224)),
        
    ])
    if args.dataset == 'ucf101':
        train_dataset = UCF101ClipRetrievalDataset( ucf_dir, args.cl, 10, True, train_transforms)
    elif args.dataset == 'hmdb51':
        train_dataset = HMDB51ClipRetrievalDataset( hmdb_dir, args.cl, 10, True, train_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True, drop_last=True)

    
    features = []
    classes = []
    #
    v_names = []
    #
    
    if not os.path.exists(os.path.join(args.feature_dir, 'train_feature.npy')):  
        print ('Training Len:',len(train_dataloader))
        for data in tqdm(train_dataloader):
            sampled_clips, idxs, v_name = data
            clips = sampled_clips.reshape((-1, 3, args.cl, 112, 112))
            ###clips = sampled_clips.reshape((-1, 3, args.cl, 128, 171))
            inputs = clips.to(device)
            # forward
            outputs = model(inputs)
            #print(outputs.shape)                            # torch.Size([bs * 10, 9216])
            # exit()
            features.append(outputs.cpu().numpy().tolist())  # Converts the tensor to list 
            classes.append(idxs.cpu().numpy().tolist())
            
            v_names.extend(v_name)
        
        

    
        features = np.array(features).reshape(-1, 10, outputs.shape[1])
        print('features is np.array of shape ::', features.shape )     #(batchs * batch.size,10,9216)
    
        classes = np.array(classes).reshape(-1, 10)                    # every video has 10 classes for 10 clips
        np.save(os.path.join(args.feature_dir, 'train_feature.npy'), features)
        np.save(os.path.join(args.feature_dir, 'train_class.npy'), classes)
        
      
        ###   
        with open(os.path.join(args.feature_dir, 'train_v_names.txt'), "w") as f:
            for v in v_names:
                f.write(str(v) +"\n")
        ###


    ### Exract for test split ###
    test_transforms = transforms.Compose([
        T.ToTensorVideo(),
        ###T.Resize((256, 256)),
        T.Resize((128, 171)),
        T.CenterCropVideo((112,112)),
        #T.CenterCropVideo((128,128)),
        #T.CenterCropVideo((224,224)),
        
    ])
    if args.dataset == 'ucf101':
        test_dataset = UCF101ClipRetrievalDataset(ucf_dir, args.cl, 10, False, test_transforms)
    elif args.dataset == 'hmdb51':
        test_dataset = HMDB51ClipRetrievalDataset(hmdb_dir, args.cl, 10, False, test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True, drop_last=True)

    features = []
    classes = []
    
    #
    v_names = []
    #
    
    if not os.path.exists(os.path.join(args.feature_dir, 'test_feature.npy')):  
        for data in tqdm(test_dataloader):
            sampled_clips, idxs, v_name = data
            clips = sampled_clips.reshape((-1, 3, args.cl, 112, 112))
            ###clips = sampled_clips.reshape((-1, 3, args.cl, 128, 171))
            inputs = clips.to(device)
            # forward
            outputs = model(inputs)
            #print(outputs.shape)
            features.append(outputs.cpu().numpy().tolist())
            classes.append(idxs.cpu().numpy().tolist())
            
            v_names.extend(v_name)

        features = np.array(features).reshape(-1, 10, outputs.shape[1])
        classes = np.array(classes).reshape(-1, 10)
        np.save(os.path.join(args.feature_dir, 'test_feature.npy'), features)
        np.save(os.path.join(args.feature_dir, 'test_class.npy'), classes)
        
        ###   
        with open(os.path.join(args.feature_dir, 'test_v_names.txt'), "w") as f:
            for v in v_names:
                f.write(str(v) +"\n")
        ###


def topk_retrieval(args):
    """Extract features from test split and search on train split features."""
    print('Load local .npy files.')
    X_train = np.load(os.path.join(args.feature_dir, 'train_feature.npy')) # X_train is a tuble
    y_train = np.load(os.path.join(args.feature_dir, 'train_class.npy'))   # y_train is a tuble
    
   
    
    
    X_train = np.mean(X_train,1)
   
    y_train = y_train[:,0]       
    
    
    
    X_train = X_train.reshape((-1, X_train.shape[-1])) 
    y_train = y_train.reshape(-1)                     
    
   
    

    X_test = np.load(os.path.join(args.feature_dir, 'test_feature.npy'))
    y_test = np.load(os.path.join(args.feature_dir, 'test_class.npy'))
    X_test = np.mean(X_test,1)
    y_test = y_test[:,0]
    X_test = X_test.reshape((-1, X_test.shape[-1]))
    y_test = y_test.reshape(-1)

    ks = [1, 5, 10, 20, 50]
    topk_correct = {k:0 for k in ks} # <class 'dict'> counter for the corrects
    
    

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)

    for k in ks:
        # print(k)
        top_k_indices = indices[:, :k]
        # print(top_k_indices.shape, y_test.shape)
        for ind, test_label in zip(top_k_indices, y_test):
            labels = y_train[ind]
            if test_label in labels:
                # print(test_label, labels)
                topk_correct[k] += 1

    result = ''
    for k in ks:
        correct = topk_correct[k]
        total = len(X_test)
        print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))
        result += ' ' + str(correct/total) + ' '
        
    with open(os.path.join(args.feature_dir, str(args.ckpt_name) + '_' + 'topk_correct.json'), 'w') as fp:
        json.dump(topk_correct, fp)
        
    return result


def get_info ():
    print ('Getting Info')
    
    """Extract features from test split and search on train split features."""
    print('Load local .npy files.')
    #args.feature_dir = r''
    args.feature_dir =  r''
    X_train = np.load(os.path.join(args.feature_dir, 'train_feature.npy')) # X_train is a tuble
    y_train = np.load(os.path.join(args.feature_dir, 'train_class.npy'))   # y_train is a tuble
    X_train = np.mean(X_train,1)
    y_train = y_train[:,0]
    X_train = X_train.reshape((-1, X_train.shape[-1]))
    y_train = y_train.reshape(-1)
    
    X_test = np.load(os.path.join(args.feature_dir, 'test_feature.npy'))
    y_test = np.load(os.path.join(args.feature_dir, 'test_class.npy'))
    X_test = np.mean(X_test,1)
    y_test = y_test[:,0]
    X_test = X_test.reshape((-1, X_test.shape[-1]))
    y_test = y_test.reshape(-1)
    
    
    ks = [2]
    topk_correct = {k:0 for k in ks} # <class 'dict'> counter for the corrects
    
    

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)
    
    print ('indices',indices.shape)
    print ('indices [0]', indices[0])
    
    testing_index = 0
    all_correct = {}
    not_correct = {}
    half_correct = {}
    
    for k in ks:
        # print(k)
        top_k_indices = indices[:, :k]
        # print(top_k_indices.shape, y_test.shape)
        for ind, test_label in zip(top_k_indices, y_test):
            labels = y_train[ind]
            if test_label in labels:
                # print(test_label, labels)
                topk_correct[k] += 1
            
            if test_label==labels[0] and test_label==labels[1] :
                all_correct[testing_index] = ind.tolist()
                
            if test_label==labels[0] or test_label==labels[1] :
                half_correct[testing_index] = ind.tolist()
                
            if test_label!=labels[0] and test_label!=labels[1] :
                not_correct[testing_index] = ind.tolist()
                
                
            testing_index += 1    
                

    result = ''
    for k in ks:
        correct = topk_correct[k]
        total = len(X_test)
        print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))
        result += ' ' + str(correct/total) + ' '
    
    print (len(all_correct))
    print (len(half_correct))
    print (len(not_correct))
    
    with open(os.path.join(args.feature_dir, '_' + 'all_correct.json'), 'w') as fp:
        json.dump(all_correct, fp)
        
    with open(os.path.join(args.feature_dir,  '_' + 'half_correct.json'), 'w') as fp:
        json.dump(half_correct, fp)
        
    with open(os.path.join(args.feature_dir,  '_' + 'not_correct.json'), 'w') as fp:
        json.dump(not_correct, fp)    
    

    
    
    

def logger(msg):
     import datetime
     now = datetime.datetime.now()
     now_str = now.strftime("%d-%m-%Y %H:%M:%S")
     
     # Writing to a file

     log_path = './retrival_log.txt'
     if (os.path.exists(log_path)):
         with open(log_path, "a") as file:
             file.write((f'{msg} \n'))
     else:
         with open(log_path, "w+") as file:
             file.write((f'{msg} \n')) 
             
 
 
def parse_args():
    parser = argparse.ArgumentParser(description='Video Retrieval')
    parser.add_argument('--exp_name', type=str, default='Video Retrieval', help='') #  Video Retrieval-RSL RF4-R(2+1)D16
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--model', type=str, default='r2plus1d_18', help='r2plus1d_18/s3d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--feature_dir', type=str, default='', help='dir to store feature.npy')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--ckpt', type=str, default= '' , help='')
    parser.add_argument('--ckpt_name', type=str, default= '' , help='')
    parser.add_argument('--bs', type=int, default=1, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--finetuning_class_num', type=int, default=4, help='number of classes during finetuning.')
    

    
    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')
    parser.add_argument('--bn-splits', default=2, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')
    parser.add_argument('--symmetric',  default=False ,action='store_true', help='use a symmetric loss function that backprops to both crops')
    parser.add_argument('--device', type=str, default='', help='Device')
    parser.add_argument('--init_mode', type=str, default='None', help='kaiming/None')
    
    
    parser.add_argument('--tuple_len', type=int, default=3, help='the number of clips sampled in each tuple.')
    parser.add_argument('--rsl_class_num', type=int, default=4, help='the rsl_class_num.')
    parser.add_argument('--ssl_class_num', type=int, default=4, help='ssl,psl class number')
    parser.add_argument('--pretext_mode', type=str, default='ssl-forward', help='ssl-forward/l/None')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    current_dir=os.getcwd().replace('C:','')
    ucf_dir=r''
    hmdb_dir=r''
    
    ucf_dir=r''
    
    args.feature_dir = os.path.join(current_dir, 'experiments', args.exp_name)
    args.ckpt = os.path.join(current_dir, 'experiments', args.exp_name, args.ckpt)
    print('===========================================================================')
    
    print(vars(args))
    
    print('===========================================================================')
 
    
    files_path = r''
    onlyfiles = [f for f in listdir(files_path) if isfile(join(files_path, f))]
    logger ('Testing Start')
    i = 1
  
    print (onlyfiles)
    #data_list = []
    i = 1
    for x in onlyfiles:
        args.ckpt = os.path.join(files_path,x)
        args.ckpt_name = x
        
        #get_info ()
        #break
        
        extract_feature(args)
        result = topk_retrieval(args)
        
        logger (str(i) + ':   ' + result + '   :' + str(x) )
        
        
        #os.remove(os.path.join(args.feature_dir, 'train_feature.npy')) # X_train is a tuble
        #os.remove(os.path.join(args.feature_dir, 'train_class.npy'))   # y_train is a tuble
        
        #os.remove(os.path.join(args.feature_dir, 'test_feature.npy'))
        #os.remove(os.path.join(args.feature_dir, 'test_class.npy'))

       
  
         

    

    #if not os.path.exists(args.feature_dir):                                  
    #    os.makedirs(args.feature_dir)
    #    extract_feature(args)
    #extract_feature(args)
    #topk_retrieval(args)
