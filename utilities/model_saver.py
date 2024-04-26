
"""

"""
import torch
import copy
import os
class Model_Saver():
    
    def __init__(self, args, model, optimizer, epoch, epoch_acc, loss, batch_count, best_mode ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.epoch_acc = epoch_acc
        self.loss = loss
        self.batch_count = batch_count
        self.best_mode = best_mode
        
        
    
    def save (self):
        self.save_model(self.args, self.model, self.optimizer, self.epoch, self.epoch_acc, self.loss, self.batch_count, self.best_mode)
        
        
    def save_model(self, args, model, optimizer, epoch, epoch_acc, loss, batch_count, best_mode):

        if (self.best_mode==1):

            path=os.path.join(args.log, args.exp_name)+'.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': epoch_acc,
                'loss':loss,
                'batch_count':batch_count
                            }, path)
            
            path=os.path.join(args.log, args.exp_name )+ '_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': epoch_acc,
                'loss':loss,
                'batch_count':batch_count
                            }, path)
            
            print('Saving The Model at epoch: ', epoch, '& batch_count: ', batch_count,)
            
            #path=os.path.join(args.log, args.exp_name)+'_'+str(epoch)+'.tar'
            #torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'acc': epoch_acc,
            #    'loss':loss,
            #    'batch_count':batch_count
            #                }, path)
            #print('Saving The Model at epoch: ', epoch, '& batch_count: ', batch_count,)
              
              
        else:
            if (self.best_mode==2):
                path=os.path.join(args.log, ('Best-Clip-'+args.exp_name))+'.tar'
                best_clip_model_state_dict=copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_clip_model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': epoch_acc,
                    'loss':loss,
                    'batch_count':batch_count
                    }, path)
                print('Saving The Best Clip Model at epoch: ', epoch, '& batch_count: ', batch_count)
            else:
                if (self.best_mode==3):
                    path=os.path.join(args.log, ('Best-Video-'+args.exp_name))+'.tar'
                    best_vid_model_state_dict=copy.deepcopy(model.state_dict())
                    torch.save({
                         'epoch': epoch,
                         'model_state_dict': best_vid_model_state_dict,
                         'optimizer_state_dict': optimizer.state_dict(),
                         'acc': epoch_acc,
                         'loss':loss,
                         'batch_count':batch_count
                         }, path)
                    
                    path=os.path.join(args.log, ('Best-Video-'+args.exp_name+'_'+str(epoch)))+'.tar'
                    torch.save({
                         'epoch': epoch,
                         'model_state_dict': best_vid_model_state_dict,
                         'optimizer_state_dict': optimizer.state_dict(),
                         'acc': epoch_acc,
                         'loss':loss,
                         'batch_count':batch_count
                         }, path)
                    
                    print('=============================================================')
                    print('Saving The Best Video Model at epoch: ', epoch, '& batch_count: ', batch_count)
                    print('=============================================================')
                else:
                    if (self.best_mode==4):
                        path=os.path.join(args.log, ('Best-Video-All-Seq-'+args.exp_name))+'.tar'
                        best_vid_model_state_dict=copy.deepcopy(model.state_dict())
                        torch.save({
                             'epoch': epoch,
                             'model_state_dict': best_vid_model_state_dict,
                             'optimizer_state_dict': optimizer.state_dict(),
                             'acc': epoch_acc,
                             'loss':loss,
                             'batch_count':batch_count
                             }, path)
                        print('Saving The Best Video Model For Test 2 at epoch: ', epoch, '& batch_count: ', batch_count)