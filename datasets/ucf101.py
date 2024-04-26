
"""
parts of the code are based on the following github codebases, we thank the authors, aknowledge their work, and give them credit for their open source code. 

# https://github.com/TengdaHan/CoCLR/blob/main/dataset/lmdb_dataset.py
            
#https://github.com/sjenni/temporal-ssl
#https://github.com/sjenni/temporal-ssl/blob/master/PreprocessorVideo.py
                
https://github.com/xudejing/video-clip-order-prediction
https://github.com/xudejing/video-clip-order-prediction/blob/master/datasets/ucf101.py
"""

"""Dataset utils for NN."""


import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile
import pickle
import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
import cv2
from skvideo.io import ffprobe
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import tqdm
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
# importing sys
###skvideo.setFFmpegPath('C:\FFmpeg\bin')
class UCF101Dataset(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, args, root_dir, train=True, transform=None, test_sample_num=10):
        
        print('Useing :: UCF101 Dateset')
        
        self.sampling_mode = args.sampling_mode
        self.test_mode = args.test_mode
        self.skip_rate = args.skip_rate
        self.root_dir = root_dir
        self.clip_len = args.cl
        self.split = args.split
        self.train = train
        self.transform= transform
        self.test_sample_num = test_sample_num
        self.args = args
        self.video_full_path = ''
        
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'Splits', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]   #<class 'pandas.core.frame.DataFrame'> #Reads The class idxs and labels
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]   #The original classInd.txt and the videos txt splits has to be modified to start at idx=0 

        if self.train:
            train_split_path = os.path.join(root_dir, 'Splits', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'Splits', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split'+ self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        
       
        
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
       
        ##################################################################
        class_idx = self.class_label2idx[videoname[:videoname.find("\\")]]     # Windows Version
        # class_idx = self.class_label2idx[videoname[:videoname.find("/")]]    # Mac Version
        
        filename = os.path.join(self.root_dir, 'Videos', videoname)
        self.video_full_path = filename 
        
        #ndarray of dimension (T, M, N, C), where T is the number of frames, M is the height, N is width, and C is depth.
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
        # random select a clip for train
        if self.train:
            
            clip = self.clip_sampler(videodata)
            #clip_start = random.randint(0, length - self.clip_len)
            #clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transform:
                #buffer = self.transform(torch.from_numpy(buffer).byte())
                r"""trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3]) """
                
                clip = self.transform(torch.from_numpy(clip).byte())
            else:
                clip = torch.tensor(clip)

            return clip, torch.tensor(int(class_idx))
        
        
        
        
        # sample several clips for test
        else:
            
            all_clips, all_idx = self.test_clip_sampler(videodata, class_idx)
            if (self.test_mode == 'test_1010'):
                return torch.stack(all_clips), torch.tensor(int(class_idx)), videoname[videoname.find("\\") + 1:]
            else:
                return torch.stack(all_clips), torch.tensor(int(class_idx))
        
        
        

    def clip_sampler(self, videodata):
        length, height, width, channel = videodata.shape
        if length >= self.clip_len:
            if self.sampling_mode=='random_skip':
                #print('Sampling Using Random Skip Frames.')
                #https://github.com/sjenni/temporal-ssl
                #https://github.com/sjenni/temporal-ssl/blob/master/PreprocessorVideo.py

                #random.randint(low, high=None, size=None, dtype=int)
                #Return random integers from low (inclusive) to high (exclusive).
                skip_frames=np.random.randint(1,5)
                #print('skip_frames',skip_frames)
                eff_skip = np.amin([(length / self.clip_len), skip_frames])
                eff_skip=int(eff_skip)
                #print ('eff_skip',eff_skip)

                max_frame = length - self.clip_len * eff_skip + 1
                #print('max_frame',max_frame)

                random_offset = int(np.random.uniform(0, max_frame))
                #print('random_offset',random_offset)


                offsets=range(random_offset, random_offset + self.clip_len * eff_skip, eff_skip)

                #print('offsets',offsets)
                #for n in offsets:
                #    print(n)
                
                clip=videodata[offsets]
                return clip
        
            else:
                if self.sampling_mode=='fixed_skip':
                    #print('Sampling Using Fixed Skip Frames.')  # To be implemented
                    skip_frames=self.skip_rate
                    eff_skip = np.amin([(length / self.clip_len), skip_frames])
                    eff_skip=int(eff_skip)
                    max_frame = length - self.clip_len * eff_skip + 1
                    random_offset = int(np.random.uniform(0, max_frame))
                    offsets=range(random_offset, random_offset + self.clip_len * eff_skip, eff_skip)
                    clip=videodata[offsets]
                    return clip
                else:
                    if self.sampling_mode=='sequential':
                        clip_start = random.randint(0, length - self.clip_len)
                        clip = videodata[clip_start: clip_start + self.clip_len]
                        return clip
        else:
            #Repeat some of the frames to pad short videos
            # pad left, only sample once
            # https://github.com/TengdaHan/CoCLR/blob/main/dataset/lmdb_dataset.py
            sequence = np.arange(self.clip_len)
            seq_idx = np.zeros_like(sequence)
            sequence = sequence[sequence < length]
            seq_idx[-len(sequence)::] = sequence
            clip=videodata[seq_idx]
            return clip
        
        
        
        
        
    def test_clip_sampler(self, videodata, class_idx):
        
        length, height, width, channel = videodata.shape
        if self.test_mode=='ten_clips' or self.test_mode =='test_1010':
            if length >= self.clip_len:
                if self.sampling_mode=='sequential':
                    # print('ten clips - sequential testing')
                    # https://github.com/xudejing/video-clip-order-prediction/blob/master/datasets/ucf101.py
                    all_clips = []
                    all_idx = []
                    for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
                        clip_start = int(i - self.clip_len/2)
                        clip = videodata[clip_start: clip_start + self.clip_len]
                        if self.transform:
                            clip = self.transform(torch.from_numpy(clip).byte())
                        else:
                            clip = torch.tensor(clip)
                                
                        all_clips.append(clip)
                        all_idx.append(torch.tensor(int(class_idx)))
                            
                    return all_clips, all_idx
                
                
                
                
                else:
                    # random skip or fixed skip sampling with 10 clips
                    # https://github.com/sjenni/temporal-ssl
                    # https://github.com/sjenni/temporal-ssl/blob/master/PreprocessorVideo.py

                    if self.sampling_mode=='random_skip':    
                        #random.randint(low, high=None, size=None, dtype=int)
                        #Return random integers from low (inclusive) to high (exclusive).
                        skip_frames = np.random.randint(1,5)
                        #print('ten clips - random skip testing')
                    else:
                        if self.sampling_mode=='fixed_skip':
                            skip_frames = self.skip_rate
                            #print('ten clips - fixed skip testing')
                        
            
                    #print('skip_frames',skip_frames)
            
                    eff_skip = np.amin([(length / self.clip_len), skip_frames])
                    eff_skip=int(eff_skip)
                    #print ('eff_skip',eff_skip)
                    
                    max_frame = length - (self.clip_len * eff_skip)
                    #print('max_frame',max_frame)
            
                    if self.test_sample_num is None:
                        start_inds = range(0, max_frame)
                    else:
                        start_inds=np.linspace(0, max_frame, self.test_sample_num)
               
                   
                    #print('start_inds',start_inds)
                    #print('start_inds len', len(start_inds))
                    #for n in start_inds:
                    #     print(n)
            
                    inds_all_sub = [range (int(i), int(i) + self.clip_len * eff_skip, eff_skip) for i in start_inds]   

                    #print('inds_all_sub',inds_all_sub)
                    #print('inds_all_sub len', len(inds_all_sub))
                    
                    all_clips = []
                    all_idx = []
                    for clip_inds in inds_all_sub:
                        clip = videodata [clip_inds]
                        if self.transform:
                            clip = self.transform(torch.from_numpy(clip).byte())
                        else:
                            clip = torch.tensor(clip)
                                
                        all_clips.append(clip)
                        all_idx.append(torch.tensor(int(class_idx)))
                    return all_clips, all_idx
                                
            else:
                #print('the video is too short, padding will be used')
                #Repeat some of the frames to pad short videos
                # pad left, only sample once
                # https://github.com/TengdaHan/CoCLR/blob/main/dataset/lmdb_dataset.py
                sequence = np.arange(self.clip_len)
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < length]
                seq_idx[-len(sequence)::] = sequence
                clip = videodata[seq_idx]
                
                if self.transform:
                    clip = self.transform(torch.from_numpy(clip).byte())
                else:
                    clip = torch.tensor(clip)
                
               
                
                # return the same clip with different transform ten times
                all_clips = []
                all_idx = []
                for x in range(0, self.test_sample_num,1):
                    all_clips.append(clip)
                    all_idx.append(torch.tensor(int(class_idx)))
                return all_clips, all_idx
                
            
        
        else:
            if self.test_mode=='all_seq':
                
                num_test_seq= 32 #14 #30 #28 #26 #24 #22 #20 #18 #16 #14 #12 #10 #8 #6 #4 #2 #32
                ###num_test_seq = self.test_sample_num
                #if self.test_sample_num !=10:
                #    num_test_seq = self.test_sample_num
                #    ###print('Attention !!! ',num_test_seq,' Testing Clips Per Videos Is Used !!!')
                    
                if length >= self.clip_len:
                    
                    #https://github.com/sjenni/temporal-ssl
                    #https://github.com/sjenni/temporal-ssl/blob/master/PreprocessorVideo.py

                    if self.sampling_mode=='random_skip':    
                        #random.randint(low, high=None, size=None, dtype=int)
                        #Return random integers from low (inclusive) to high (exclusive).
                        skip_frames = np.random.randint(1,5)
                        #print('all sequences - random skip testing')
                    else:
                        if self.sampling_mode=='fixed_skip':
                            skip_frames = self.skip_rate
                            #print('all sequences - fixed skip testing')
                        else:
                            if self.sampling_mode=='sequential':
                                skip_frames = 1
                                #print('all sequences - sequential testing')
                                    
                        
                        
                    #print('skip_frames',skip_frames)
            
                    eff_skip = np.amin([(length / self.clip_len), skip_frames])
                    eff_skip=int(eff_skip)
                    #print ('eff_skip',eff_skip)

                    max_frame = length - self.clip_len * eff_skip + 1
                    #print('max_frame',max_frame)
            
                    if num_test_seq is None:
                        start_inds = range(0, max_frame)
                    else:
                        start_inds = range(0, max_frame, np.amax([(max_frame//num_test_seq), 1]))
            
                    # This code does not limite the number of videos by 32 because sometimes
                    # np.amax([(max_frame//num_test_seq) produce a floting point like 3,99
                    # then the step in the range will be 3 and there will 32*0.99 extra frames
                    # in the sequence that could be used as starting point to sample more clips. 
                    # In this situation,
                    # (I think maybe I am wrong)this equation could sample a max of 32+31=63 clip
                    # from a video.it depends on the lengh of the video, and the effective skip 
                    # and the clip length.
            
 
            
                    #print('start_inds',start_inds)
                    #print('start_inds len', len(start_inds))
                    #for n in start_inds:
                    #    print(n)
            
                    inds_all_sub = [range (i, i + self.clip_len * eff_skip, eff_skip) for i in start_inds]   

                    #print('inds_all_sub',inds_all_sub)
                    #print('inds_all_sub len', len(inds_all_sub))
                        
                    all_clips = []
                    all_idx = []
                    for clip_inds in inds_all_sub:
                        clip = videodata [clip_inds]
                        if self.transform:
                            clip = self.transform(torch.from_numpy(clip).byte())
                        else:
                            clip = torch.tensor(clip)
                                
                        all_clips.append(clip)
                        all_idx.append(torch.tensor(int(class_idx)))
                    return all_clips, all_idx
                        
                else:
                    #print('the video is too short, padding will be used')
                    #Repeat some of the frames to pad short videos
                    # pad left, only sample once
                    # https://github.com/TengdaHan/CoCLR/blob/main/dataset/lmdb_dataset.py
                    sequence = np.arange(self.clip_len)
                    seq_idx = np.zeros_like(sequence)
                    sequence = sequence[sequence < length]
                    seq_idx[-len(sequence)::] = sequence
                    clip=videodata[seq_idx]
                   
                   
                    
                    all_clips = []
                    all_idx = []
                    if self.transform:
                        clip = self.transform(torch.from_numpy(clip).byte())
                    else:
                        clip = torch.tensor(clip)
                    all_clips.append(clip)
                    all_idx.append(torch.tensor(int(class_idx)))
                    return all_clips, all_idx
            
            
            else:
                if self.test_mode=='mined_testing':
                    
                    
                      
                    ###self.test_sample_num = 2   # This will override the value of the test_sample_num
                    file_name=os.path.join(self.args.log, self.args.exp_name)+'_clips_dict.pickle'
                    if os.path.exists(file_name):  
                        #print ('Loading the Clips Dictionary')
                        with open(file_name, 'rb') as handle:
                            clips_dict = pickle.load(handle)
                            
                        vid_name = (self.video_full_path.split('\\')[-1]).replace('.avi','')
                        ###print ('vid_name', vid_name)
                        ###print ('clips_dict [vid_name]',clips_dict [vid_name]['dense'])
                        
                        inds_all_sub = clips_dict [vid_name]['dense']
                        #print('inds_all_sub', len(inds_all_sub))
                        all_clips = []
                        all_idx = []
                        for clip_inds in inds_all_sub:
                            clip = videodata [clip_inds]
                            if self.transform:
                                clip = self.transform(torch.from_numpy(clip).byte())
                            else:
                                clip = torch.tensor(clip)
                                    
                            all_clips.append(clip)
                            all_idx.append(torch.tensor(int(class_idx)))
                        return all_clips, all_idx
                            
                            
                    else:
                        raise Exception('No Dict is found !!!')
                    
                        
                    
                    
                    
                    