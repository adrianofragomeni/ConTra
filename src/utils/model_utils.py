import torch as th
from pathlib import Path
import shutil

class Model_utils:
    
    def __init__(self,filepath,best_metric=0):
        self.Best_RSum=best_metric
        self.filepath=filepath
    
    def save_checkpoint(self,state,filename="checkpoint.pth.tar"):
        """ save checkpoint of the model
        """
        filepath=self.filepath / filename
        th.save(state, filepath)
        if state["best_score"]>self.Best_RSum:
            self.Best_RSum=state["best_score"]
            shutil.copyfile(filepath, Path(self.filepath / "model_best.pth.tar"))
    
    def load_checkpoint(self,model, optimizer,scheduler,filename="checkpoint.pth.tar"):
        """ load checkpoint of the model
        """
        best_model = th.load(Path(self.filepath) / "model_best.pth.tar")
        self.Best_RSum=best_model["best_score"]
        del best_model
        
        resume_path=self.filepath / filename
        
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = th.load(resume_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, start_epoch))
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint["scheduler"])
        
        return model, optimizer,scheduler, start_epoch










