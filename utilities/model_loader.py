
"""

"""
import torch
import copy
import os

class Model_Loader():
    def __init__(self, args, model, optimizer):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        
    def load(self):
        return self.load_model(self.args, self.model, self.optimizer)
        
    def load_model(self, args, model, optimizer):
        
        print_sep = '============================================================='
        ckpt = os.path.join(args.log, args.exp_name)+'.tar'
        if os.path.exists(ckpt):
            # loads the checkpoint
            print(print_sep)    
            print("Reloading from previously general saved checkpoint")
            checkpoint = torch.load(ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
            # obtains the epoch the training is to resume from
            epoch_resume = checkpoint["epoch"]+1
            #batch_count= checkpoint["batch_count"]
            #t_acc=checkpoint["acc"]
            starting_epoch = epoch_resume
        else:
            #global start_epoch
            starting_epoch = args.start_epoch
            epoch_resume=0
     
            
            
        best_clip_ckpt = os.path.join(args.log, ('Best-Clip-'+args.exp_name))+'.tar'
        if os.path.exists(best_clip_ckpt):
            best_clip_checkpoint = torch.load(best_clip_ckpt)
            print("Best Clip Checkpoint Reloading ......")
            best_clip_model = best_clip_checkpoint['model_state_dict']   #The State of the Model
            best_clip_acc = best_clip_checkpoint['acc']
            best_clip_loss = best_clip_checkpoint['loss']
            print('Best Clip Val Acc ::',best_clip_acc )
        
        else:
            best_clip_acc = 0.0
            best_clip_loss = float('inf')
            
        best_vid_ckpt = os.path.join(args.log, ('Best-Video-'+args.exp_name))+'.tar'
        if os.path.exists(best_vid_ckpt):
            best_video_checkpoint = torch.load(best_vid_ckpt)
            print("Best Video Checkpoint Reloading ......")
            best_video_model = best_video_checkpoint['model_state_dict']  #The State of the Model
            best_video_acc = best_video_checkpoint['acc']
            best_video_loss = best_video_checkpoint['loss']
            print('Best Video Val Acc ::',best_video_acc )
        
        else:
            best_video_acc = 0.0
            best_video_loss = float('inf')
            
        best_dual_vid_ckpt=os.path.join(args.log, ('Best-Video-All-Seq-'+args.exp_name))+'.tar'
        if os.path.exists(best_dual_vid_ckpt):
            best_dual_video_checkpoint = torch.load(best_dual_vid_ckpt)
            print("Best Dual Video Checkpoint Reloading ......")
            best_dual_video_model = best_dual_video_checkpoint['model_state_dict']  #The State of the Model
            best_dual_video_acc = best_dual_video_checkpoint['acc']
            best_dual_video_loss = best_dual_video_checkpoint['loss']
            print('Best Dual Video Val Acc ::',best_dual_video_acc )
        
        else:
            best_dual_video_acc = 0.0
            best_dual_video_loss = float('inf')
        
        if best_video_acc >= best_dual_video_acc :
            best_acc = best_video_acc
            best_loss = best_video_loss
            print('Best Acc is Ten Clips Acc : ', best_acc)
        else:
            
            best_acc = best_dual_video_acc
            best_loss =  best_dual_video_loss
            print('Best Acc is All Seq Acc : ', best_acc)
        
        
        return model, optimizer, epoch_resume, starting_epoch, best_acc, best_loss    