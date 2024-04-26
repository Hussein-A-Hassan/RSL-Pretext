
"""
parts of the code are based on the following github codebases, we thank the authors, aknowledge their work, and give them credit for their open source code.

https://github.com/xudejing/video-clip-order-prediction
https://github.com/xudejing/video-clip-order-prediction/blob/master/datasets/ucf101.py

#https://github.com/sjenni/temporal-ssl
#https://github.com/sjenni/temporal-ssl/blob/master/PreprocessorVideo.py
                
"""

"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile
import cv2
import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class DSUCF101Dataset(Dataset):
    
    """Duplicated Scene UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, args, root_dir, train=True, transform=None):
        
        
    
        print('Useing :: UCF101 Dateset')
        if (args.select_mode=='fixed' and args.r_frames==2 and args.cl == 16):
           self.classes_idxs={0:0, 2:1, 4:2, 6:3, 8:4, 10:5, 12:6, 16:7}
           
        elif (args.select_mode=='fixed' and args.r_frames==4 and args.cl == 16):
           self.classes_idxs={0:0, 4:1, 8:2, 16:3}
           
        elif (args.select_mode=='fixed' and args.r_frames==8 and args.cl == 16):
              self.classes_idxs={0:0, 16:1}
        
        elif (args.select_mode=='fixed' and args.r_frames==4 and args.cl == 32):
           self.classes_idxs={0:0, 4:1, 8:2, 12:3, 16:4, 20:5, 24:6, 32:7} 
           
        elif (args.select_mode=='fixed' and args.r_frames==8 and args.cl == 32):
           self.classes_idxs={0:0, 8:1, 16:2, 32:3}
           
        elif (args.select_mode=='fixed' and args.r_frames==16 and args.cl == 32):
           self.classes_idxs={0:0, 32:1}
           
        elif (args.select_mode=='fixed' and args.r_frames==8 and args.cl == 64 ):   
           self.classes_idxs={0:0, 8:1, 16:2, 24:3, 32:4, 40:5, 48:6, 64:7}  #classes=[0,8,16,24,32,40,48,64]
            
        elif  ((args.select_mode=='fixed' and args.r_frames==16 and args.cl == 64 )):
           self.classes_idxs={0:0, 16:1, 32:2, 64:3}                       #classes=[0,16,32,64]
                    
           
        self.r_frames=args.r_frames
        self.n_frames=args.n_frames
        self.offset=args.offset
        self.select_mode=args.select_mode
        self.ins_mode=args.ins_mode
        self.p=args.minimum_p
        self.sampling_mode=args.sampling_mode
        self.skip_rate=args.skip_rate
    
        self.root_dir = root_dir
        self.clip_len = args.cl
        self.split = args.split
        self.train = train
        self.transform= transform
        self.toPIL = transforms.ToPILImage()
        
        class_idx_path = os.path.join(root_dir, 'Splits', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]   #
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]   #

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
            class_idx (tensor): class index, [0-1] 0 = no duplication 1= duplication is created 
        """
        
        
        
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        ###class_idx = self.class_label2idx[videoname[:videoname.find("\\")]]
        filename = os.path.join(self.root_dir, 'Videos', videoname)
        
        #ndarray of dimension (T, M, N, C), where T is the number of frames, M is the height, N is width, and C is depth.
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
         
        # random select a clip for train
        if self.train:        
            clip = self.clip_sampler(videodata)
            #clip_start = random.randint(0, length - self.clip_len)
            #clip = videodata[clip_start: clip_start + self.clip_len]
            
            if (random.random()>=self.p):
                # create duplicated videos
                clip, copy_idx=self.repeated_scene_gen(clip, self.r_frames, self.n_frames, self.offset, self.select_mode, self.ins_mode)
                class_idx=copy_idx
                #print((class_idx))
                #class_idx='1' Binary calssification case 
            else:
                #class_idx='0' Binary calssification case
                
                if (self.select_mode=='fixed'  and self.r_frames==2 and self.clip_len == 16):
                   class_idx=16
                   
                elif (self.select_mode=='fixed'  and self.r_frames==4 and self.clip_len == 16):
                   class_idx=16
                   
                elif (self.select_mode=='fixed'  and self.r_frames==8 and self.clip_len == 16):
                     class_idx=16
                
                elif  (self.select_mode=='fixed' and self.r_frames==4 and self.clip_len == 32):
                   class_idx=32
                   
                elif  (self.select_mode=='fixed' and self.r_frames==8 and self.clip_len == 32):
                   class_idx=32
                   
                elif  (self.select_mode=='fixed' and self.r_frames==16 and self.clip_len == 32):
                   class_idx=32   
                
                elif (self.select_mode=='fixed' and self.r_frames==8 and self.clip_len == 64):
                   class_idx=64
                       
                elif (self.r_frames==16 and self.select_mode=='fixed' and self.clip_len == 64):
                   class_idx=64
                       
                #print((class_idx))
         
            if self.transform:
                clip = self.transform(torch.from_numpy(clip).byte())
            else:
                clip = torch.tensor(clip)

            
            return clip, torch.tensor(int(self.classes_idxs[class_idx]))
        
        else:
            # Right now there is no difference between train and test
            
            clip = self.clip_sampler(videodata)
            #clip_start = random.randint(0, length - self.clip_len)
            #clip = videodata[clip_start: clip_start + self.clip_len]
            
            if (random.random()>=self.p):
                # create duplicated videos
                clip, copy_idx=self.repeated_scene_gen(clip, self.r_frames, self.n_frames, self.offset, self.select_mode, self.ins_mode)
                class_idx=copy_idx
                #print((class_idx))
                #class_idx='1' Binary calssification case 
            else:
                #class_idx='0' Binary calssification case
                if (self.select_mode=='fixed'  and self.r_frames==2 and self.clip_len == 16):
                   class_idx=16
                   
                elif (self.select_mode=='fixed'  and self.r_frames==4 and self.clip_len == 16):
                   class_idx=16
                   
                elif (self.select_mode=='fixed'  and self.r_frames==8 and self.clip_len == 16):
                     class_idx=16
                
                elif  (self.select_mode=='fixed' and self.r_frames==4 and self.clip_len == 32):
                   class_idx=32
                   
                elif  (self.select_mode=='fixed' and self.r_frames==8 and self.clip_len == 32):
                   class_idx=32
                   
                elif  (self.select_mode=='fixed' and self.r_frames==16 and self.clip_len == 32):
                   class_idx=32   
                
                elif (self.select_mode=='fixed' and self.r_frames==8 and self.clip_len == 64):
                   class_idx=64
                       
                elif (self.r_frames==16 and self.select_mode=='fixed' and self.clip_len == 64):
                   class_idx=64
                #print((class_idx))
         
            if self.transform:
                clip = self.transform(torch.from_numpy(clip).byte())
            else:
                clip = torch.tensor(clip)

             
            return clip, torch.tensor(int(self.classes_idxs[class_idx]))

        
        
    
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
        
    def get_clip1(self, video_array, clip_len):
        found=False
        total_len=video_array.shape[0]
        while found==False:
            #Return random integers from low (inclusive) to high (exclusive).
            time_index = np.random.randint(0,total_len) # idx >0 and idx<total_len
            if time_index + (clip_len-1) < total_len:
                 found=True
                    
        video_clip=np.empty([clip_len, video_array.shape[1], video_array.shape[2],3])
        video_clip=video_array[time_index:time_index + clip_len,:,:,:]  # time_index + clip_len is excluded
        #print('clip1 shape',video_clip.shape)
        return video_clip
    
    def get_clip2(self, video_array, clip_len, offset, selection_mode):
        
        total_len=video_array.shape[0]
        if (selection_mode=='random'):      #Random Selection with or without offset
            found=False
            while found==False:
                #Return random integers from low (inclusive) to high (exclusive).
                time_index = np.random.randint(0,total_len)
                if (time_index + (clip_len-1)) < total_len:
                    #print('copy index', time_index)
                    left_spaces=time_index
                    right_spaces=(total_len-(time_index + (clip_len)))+clip_len # Added the last clip_len because we have blanks 
                    if (left_spaces>offset+clip_len)or(right_spaces>offset+clip_len):
                        found=True
        
            video_clip=np.empty([clip_len, video_array.shape[1], video_array.shape[2],3])
            video_clip=video_array[time_index:time_index + clip_len,:,:,:]
            return video_clip, time_index 
        else:
            if (selection_mode=='fixed'):
                indices_list=[x for x in range(0,total_len,clip_len)]
                #print(indices_list)
                copy_loc=random.randint(0,len(indices_list)-1)
                time_index=indices_list[copy_loc]
                #print('copy_idx',time_index)
                video_clip=np.empty([clip_len, video_array.shape[1], video_array.shape[2],3])
                video_clip=video_array[time_index:time_index + clip_len,:,:,:]
                return video_clip, time_index 
            
    
    def get_ins_index(self, clip_len, video_len, copy_index, ins_mode, offset):
        
        right_range=[]
        left_range=[]
        indices_list=[]
        ins_index=0
        #range(start, stop, step)
        
        if (copy_index>=clip_len):
            right_range=range(0,(copy_index-clip_len)+1,1)  # possible and valid positions for insertion
        
        if (((video_len)-(copy_index+clip_len))>=clip_len):  
            left_range=range((copy_index+clip_len),(video_len-clip_len)+1,1) # possible and valid positions for insertion
        
        #print('R-->',right_range)
        #print('L-->',left_range)
    
        for x in right_range:
            indices_list.append(x)
        
        for x in left_range:
            indices_list.append(x)
        
        #print(indices_list)
        
        if ins_mode =='random':
            ins_index=indices_list[np.random.randint(0,len(indices_list)-1)]
            #print(ins_index)
        else:
            if (offset!=0 and ins_mode=='faraway' ):
                found=False
                #ins_index=indices_list[np.random.randint(0,len(indices_list)-1)]
                while (found==False):
                    ins_index=indices_list[np.random.randint(0,len(indices_list)-1)]
                    if (ins_index>copy_index):
                        dist=abs(ins_index-(copy_index+clip_len-1))
                        if (dist>=offset):
                            found=True
                            
                    if (ins_index<copy_index):
                        dist=abs(copy_index-(ins_index+clip_len-1))
                        if (dist>=offset):
                            found=True    
                            
            else:
               if (offset!=0 and ins_mode=='close_by' ):
                   found=False
                   #ins_index=indices_list[np.random.randint(0,len(indices_list)-1)]
                   while (found==False):
                       ins_index=indices_list[np.random.randint(0,len(indices_list)-1)]
                       if (ins_index>copy_index):
                           dist=abs(ins_index-(copy_index+clip_len-1))
                           if (dist==offset):
                               found=True
                               
                       if (ins_index<copy_index):
                           dist=abs(copy_index-(ins_index+clip_len-1))
                           if (dist==offset):
                               found=True     
            
                
            ##print('Not Random-->','ins_in:',ins_index,' copy_in:',copy_index, 'offset:',dist)
            
        return ins_index

    def save_video(self, n_video, name):
       
        writer = cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*"mp4v"), 25,(n_video.shape[2],n_video.shape[1]))

        for i in range(n_video.shape[0]):
            
            writer.write ((n_video[i,:,:,:]).astype((np.uint8)))

        writer.release()
        
        
        
        
    def repeated_scene_gen(self ,clip, r_frames, n_frames, offset, select_mode, ins_mode):
        
        if select_mode=='fixed':
            
            total_frames= r_frames + n_frames
        
            video_p1= self.get_clip1(clip, n_frames)
            
            video_p2, copy_index= self.get_clip2(video_p1, r_frames, offset, select_mode)
            
            ins_index= self.get_ins_index(r_frames, total_frames, copy_index, ins_mode, offset)
            
            n_video=np.empty([total_frames, clip.shape[1], clip.shape[2],3])
            
            i=0
            k=0
            r=0
            while (i<total_frames):
                if (i>=ins_index and i<ins_index+r_frames):
                    n_video[i,:,:,:]=video_p2[k,:,:,:]
                    k+=1
                else:
                    n_video[i,:,:,:]=video_p1[r,:,:,:]
                    r+=1
                i+=1
        else:
            
            print('Random Selection is NOT implemented yet')
            
        #save_video(n_video, r'')
        #save_video(clip, r'')
        return n_video, copy_index 