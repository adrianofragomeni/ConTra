from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os
import time


class Summary:
    """ Monitoring metrics """

    def __init__(self,filepath):
    
        self.writer = SummaryWriter(Path(filepath, "logs/runs"))  

    def write_test(self,metrics,epoch):
        
        """ Monitoring metrics """
        self.writer.add_scalars("Loss",{"Test": metrics["testing_loss"]},epoch)
        self.writer.add_scalars("Loss_Inter",{"Test": metrics["testing_loss_Inter"]},epoch)
        self.writer.add_scalars("Loss_Uniformity",{"Test": metrics["testing_loss_Uniformity"]},epoch)
        self.writer.add_scalars("Loss_Neighbouring_Video",{"Test": metrics["testing_loss_Neighbouring_video"]},epoch)
        self.writer.add_scalars("Loss_Neighbouring_Text",{"Test": metrics["testing_loss_Neighbouring_text"]},epoch)

        self.writer.add_scalars("RSum",{"Test": metrics["RSum"]},epoch)
        self.writer.add_scalars("RSum_t2v",{"Test": metrics["RSum_t2v"]},epoch)
        self.writer.add_scalars("RSum_v2t",{"Test": metrics["RSum_v2t"]},epoch)
        self.writer.add_scalars("R1_v2t",{"Test": metrics["R1_v2t"]},epoch)
        self.writer.add_scalars("R5_v2t",{"Test": metrics["R5_v2t"]},epoch)
        self.writer.add_scalars("R10_v2t",{"Test": metrics["R10_v2t"]},epoch)
        self.writer.add_scalars("MR_v2t",{"Test": metrics["MR_v2t"]},epoch)
        self.writer.add_scalars("MR_t2v",{"Test": metrics["MR_t2v"]},epoch)
        self.writer.add_scalars("R1_t2v",{"Test": metrics["R1_t2v"]},epoch)
        self.writer.add_scalars("R5_t2v",{"Test": metrics["R5_t2v"]},epoch)
        self.writer.add_scalars("R10_t2v",{"Test": metrics["R10_t2v"]},epoch)

    def write_train(self,metrics,epoch):
        
        """ Monitoring metrics """
        self.writer.add_scalars("Loss",{"Train": metrics["training_loss"]},epoch)
        self.writer.add_scalars("Loss_Inter",{"Train": metrics["training_loss_Inter"]},epoch)
        self.writer.add_scalars("Loss_Uniformity",{"Train": metrics["training_loss_Uniformity"]},epoch)
        self.writer.add_scalars("Loss_Neighbouring_Video",{"Train": metrics["training_loss_Neighbouring_video"]},epoch)
        self.writer.add_scalars("Loss_Neighbouring_Text",{"Train": metrics["training_loss_Neighbouring_text"]},epoch)
        

def create_log_file(log_path, log_name=""):
    # Make log folder if necessary
    log_folder = os.path.dirname(log_path)
    os.makedirs(log_folder, exist_ok=True)

    # Initialize log files
    with open(log_path, "a") as log_file:
        now = time.strftime("%c")
        log_file.write("==== log {} at {} ====\n".format(log_name, now))



def log_metrics(epoch, metrics, log_path=None):
    """log_path overrides the destination path of the log"""
    
    now = time.strftime("%c")
    message = "(epoch: {}, time: {})".format(epoch, now)
    for k, v in metrics.items():
        message = message + ",{}:{}".format(k, v)

    # Write log message to correct file
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")
    return message


class Monitor:
    """" Monitor all the variables during training and evaluation"""
    
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.train_path = os.path.join(self.checkpoint, "train.txt")
        self.val_path = os.path.join(self.checkpoint, "val.txt")
        create_log_file(self.train_path)
        create_log_file(self.val_path)


    def log_train(self, epoch, metrics):
        log_metrics(epoch, metrics, self.train_path)

    def log_val(self, epoch, metrics):
        log_metrics(epoch, metrics, self.val_path)