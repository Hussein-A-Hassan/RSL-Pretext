
"""

"""
import os
import datetime
# logger = Logger(args, epoch, train_acc1, train_acc2, val_acc1, val_acc2, None, train_loss, val_loss, None, None, 'SSPL')
class Logger():
    def __init__(self, args, epoch, training_acc, testing_acc, training_loss, testing_loss, info):
        self.args = args
        self.epoch = epoch
        self.training_acc = training_acc
        self.testing_acc = testing_acc
        self.training_loss = training_loss
        self.testing_loss = testing_loss
        self.info = info
        
        
        
    
    def acc_log(self):
        self.logger1(self.args, self.epoch, self.training_acc, self.testing_acc, self.training_loss, self.testing_loss, self.info)

            

            
        
    def logger1(self, args, epoch, training_acc, testing_acc, training_loss, testing_loss, info):
        now = datetime.datetime.now()
        now_str = now.strftime("%d-%m-%Y %H:%M:%S")
        
        # Writing to a file
        log_acc_path = args.log+'\\'+'acc_log.txt'
        if (os.path.exists(log_acc_path)):
            with open(log_acc_path, "a") as file:
            
                file.write((f'{now_str:20}    {epoch:03}    {training_acc:.06f}    {testing_acc:.06f}    {training_loss:.06f}    {testing_loss:.06f} {info}\n'))
        else:
            with open(log_acc_path, "w+") as file:
           
                file.write(( 'Date               Epoch            Train Acc             Testing Acc          Training Loss            Testing Loss \n'))
                file.write((f'{now_str:20}    {epoch:03}    {training_acc:.06f}    {testing_acc:.06f}    {training_loss:.06f}    {testing_loss:.06f} {info}  \n'))
                
