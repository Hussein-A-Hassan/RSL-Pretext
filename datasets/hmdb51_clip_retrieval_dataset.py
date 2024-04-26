
"""

parts of the code are based on the following github codebases, we thank the authors, aknowledge their work, and give them credit for their open source code.

# https://github.com/xudejing/video-clip-order-prediction/blob/master/datasets/ucf101.py

"""



"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile

import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class HMDB51ClipRetrievalDataset(Dataset):
    """HMDB51 dataset for Retrieval. Sample clips for each video. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        sample_num(int): number of clips per video.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, sample_num, train=True, transform=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.sample_num = sample_num
        self.train = train
        self.transform= transform
        #self.toPIL = transforms.ToPILImage()
        
        class_idx_path = os.path.join(root_dir, 'Splits', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'Splits', 'trainlist01.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'Splits', 'testlist01.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('\\')]]
        filename = os.path.join(self.root_dir, 'Videos', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
        all_clips = []
        all_idx = []
        
        # print('Clips Starting Idxs + 8 ::',np.linspace(self.clip_len/2, length-self.clip_len/2, self.sample_num))
        # np.linspace: Return evenly spaced numbers over a specified interval.
        for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.sample_num):
            clip_start = int(i - self.clip_len/2)
            clip = videodata[clip_start: clip_start + self.clip_len]
            if self.transform:
                #trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                #seed = random.random()
                #for frame in clip:
                #    random.seed(seed)
                #    frame = self.toPIL(frame) # PIL image
                #    frame = self.transforms_(frame) # tensor [C x H x W]
                #    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                #clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                clip = self.transform(torch.from_numpy(clip).byte())
            else:
                clip = torch.tensor(clip)
            
            all_clips.append(clip)
            all_idx.append(torch.tensor(int(class_idx)))


        #print('Returned Clips Size::',(torch.stack(all_clips)).size()) ==>  torch.Size([10, 3, 16, 112, 112])
        #print('Returned Clips Idxs ::',(torch.stack(all_idx)).size())  ==>  torch.Size([10])
        return torch.stack(all_clips), torch.stack(all_idx), 'None'
    
    

if __name__ == '__main__':
    print()