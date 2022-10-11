#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as th
from pathlib import Path
from utils.utils import parsing, get_default_device
from utils.model_utils import Model_utils
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from evaluation.evaluation import Evaluation_metrics
from evaluation.monitoring import Summary,Monitor
from loader.loader_sequence_ActNet import FeatureLoader, ToTensor
from losses import NCE,optimizer_scheduler
import numpy as np
import random
from models.contra_ActNet import ConTraModel

# Random Initialization
np.random.seed(0)
random.seed(0)
th.manual_seed(0)



def train_epoch(training_loader,device, optimizer,net, loss, scheduler,length_context_text,length_context_video):
    
    running_loss = 0.0
    running_loss_Inter = 0.0
    running_loss_uniformity=0.0
    running_loss_Neighbouring_text=0.0
    running_loss_Neighbouring_video=0.0
    negative_embed_text=None
    negative_embed_video=None
    mask=None
    
    for i_batch, sample_batched in enumerate(training_loader):
    
        video_features=sample_batched["video_embed"].to(device)
        text_features=sample_batched["text_embed"].to(device)

        optimizer.zero_grad()
        
        if length_context_video == 1 and length_context_text == 1:

            text_embed,video_embed = net(text_features,video_features,branch="all")
        
        elif length_context_video > 1 and length_context_text == 1:
            
            negatives_features_video=sample_batched["negatives_videos"].to(device)
            mask=sample_batched["masking"]
            text_embed,video_embed = net(text_features,video_features,branch="all")
            negative_embed_video=net(None,negatives_features_video,branch="video")

        elif length_context_video == 1 and length_context_text > 1:            
        
             negatives_features_text=sample_batched["negatives_texts"].to(device)

             mask=sample_batched["masking"]
             
             text_embed,video_embed = net(text_features,video_features,branch="all")
             negative_embed_text=net(negatives_features_text,video_features[mask,:],branch="text")
        
        else:
            
            negatives_features_video=sample_batched["negatives_videos"].to(device)
            negatives_features_text=sample_batched["negatives_texts"].to(device)

            mask=sample_batched["masking"]

            text_embed,video_embed = net(text_features,video_features,branch="all")
            negative_embed_text=net(negatives_features_text,video_features[mask,:],branch="text")
            negative_embed_video=net(None,negatives_features_video,branch="video")
       
        
        train_loss, train_loss_dict = loss(video_embed, text_embed, negative_embed_text,negative_embed_video,mask)

        running_loss += train_loss.item()
        running_loss_Inter += train_loss_dict["Inter_loss"]
        running_loss_uniformity += train_loss_dict["Uniformity_loss"]
        running_loss_Neighbouring_text += train_loss_dict["Neighbouring_text_loss"]
        running_loss_Neighbouring_video += train_loss_dict["Neighbouring_video_loss"]


        train_loss.backward()
        optimizer.step()
        scheduler.step()
        
        
    return {"training_loss" :running_loss / (i_batch + 1), "training_loss_Inter" :running_loss_Inter / (i_batch + 1),
            "training_loss_Uniformity" :running_loss_uniformity / (i_batch + 1) ,"training_loss_Neighbouring_text" :running_loss_Neighbouring_text / (i_batch + 1),
            "training_loss_Neighbouring_video" :running_loss_Neighbouring_video / (i_batch + 1)}

                
def validation_epoch(testing_loader,device, eval_,net, loss,length_context_text,length_context_video):
    
    all_text_embed=[]
    all_video_embed=[]
    
    running_loss = 0.0
    running_loss_Inter = 0.0
    running_loss_uniformity=0.0
    running_loss_Neighbouring_text=0.0
    running_loss_Neighbouring_video=0.0
    negative_embed_text=None
    negative_embed_video=None
    mask=None
    
    with th.no_grad():
        for i_batch, sample_batched in enumerate(testing_loader):

            video_features=sample_batched["video_embed"].to(device)
            text_features=sample_batched["text_embed"].to(device)

            if length_context_video == 1 and length_context_text == 1:
                
                text_embed,video_embed = net(text_features,video_features,branch="all")
            
            elif length_context_video > 1 and length_context_text == 1:
                
                negatives_features_video=sample_batched["negatives_videos"].to(device)
                mask=sample_batched["masking"]
                
                text_embed,video_embed = net(text_features,video_features,branch="all")
                negative_embed_video=net(None,negatives_features_video,branch="video")
    
            elif length_context_video == 1 and length_context_text > 1:            
            
                 negatives_features_text=sample_batched["negatives_texts"].to(device)

                 mask=sample_batched["masking"]
                 
                 text_embed,video_embed = net(text_features,video_features,branch="all")
                 negative_embed_text=net(negatives_features_text,video_features[mask,:],branch="text")
            
            else:
                
                negatives_features_video=sample_batched["negatives_videos"].to(device)
                negatives_features_text=sample_batched["negatives_texts"].to(device)

                mask=sample_batched["masking"]
                
                text_embed,video_embed = net(text_features,video_features,branch="all")
                negative_embed_text=net(negatives_features_text,video_features[mask,:],branch="text")
                negative_embed_video=net(None,negatives_features_video,branch="video")


            val_loss, val_loss_dict = loss(video_embed, text_embed, negative_embed_text,negative_embed_video,mask)
            
            all_video_embed.append(video_embed.cpu().numpy())
            all_text_embed.append(text_embed.cpu().numpy())
            
            running_loss += val_loss
            running_loss_Inter += val_loss_dict["Inter_loss"]
            running_loss_uniformity += val_loss_dict["Uniformity_loss"]
            running_loss_Neighbouring_text += val_loss_dict["Neighbouring_text_loss"]
            running_loss_Neighbouring_video += val_loss_dict["Neighbouring_video_loss"]
        
        all_video_embed=np.vstack(all_video_embed)
        all_text_embed=np.vstack(all_text_embed)

        similarity_matrix= np.matmul(all_text_embed,all_video_embed.T)     
        metrics=eval_.get_metrics(similarity_matrix)
        
        metrics.update({"testing_loss" :running_loss / (i_batch + 1), "testing_loss_Inter" :running_loss_Inter / (i_batch + 1),
            "testing_loss_Uniformity" :running_loss_uniformity / (i_batch + 1) ,"testing_loss_Neighbouring_text" :running_loss_Neighbouring_text / (i_batch + 1),
            "testing_loss_Neighbouring_video" :running_loss_Neighbouring_video / (i_batch + 1)})
        
    return metrics 



def train():
    
    summary= Summary(args_.path_model)
    device= get_default_device()
    
    monitor = Monitor(Path(args_.path_model,"logs"))
    model_utils=Model_utils(Path(args_.path_model))        
        
    print("load model...")
    length_context_video=1+2*args_.m_video
    length_context_text=1+2*args_.m_text
    net= ConTraModel(args_,
                     args_.embed_dim,
                     args_.dropout_text,
                     args_.dropout_video,
                     args_.nhead_text,
                     args_.nhead_video,
                     args_.dim_feedforward_text,
                     args_.dim_feedforward_video,
                     args_.nlayer_text,
                     args_.nlayer_video,
                     length_context_text, 
                     length_context_video,
                     device)

    print("load features...")
    training_set=FeatureLoader(path_dataframe=Path(args_.path_dataframes) / args_.dataset / "retrieval_train.csv",
                              path_features_video=Path(args_.path_features) / args_.dataset / "training_video_features.pkl", 
                              path_features_text=Path(args_.path_features) / args_.dataset / "training_text_features.pkl",
                               dataset=args_.dataset,
                               len_text=length_context_text,
                               len_video=length_context_video,
                               transform=ToTensor())
    training_loader = DataLoader(training_set, 
                                 batch_size=args_.batch_size,
                                 pin_memory=True,
                                 shuffle=True,
                                 num_workers=args_.cpu_count)

    testing_set=FeatureLoader(path_dataframe=Path(args_.path_dataframes) / args_.dataset / "retrieval_test.csv",
                              path_features_video=Path(args_.path_features) / args_.dataset / "validation_video_features.pkl", 
                              path_features_text=Path(args_.path_features) / args_.dataset / "validation_text_features.pkl",
                              dataset=args_.dataset,
                              len_text=length_context_text,
                              len_video=length_context_video,
                              transform=ToTensor())
    testing_loader = DataLoader(testing_set,
                                batch_size=args_.batch_size,
                                pin_memory=True,
                                shuffle=False,
                                num_workers=args_.cpu_count)


    # Optimizers + Loss
    loss=NCE.Lossfunction( device, 
                          args_.lambda1, 
                          args_.lambda2, 
                          args_.lambda3,
                          args_.lambda4, 
                          args_.temperature)
    loss.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args_.lr)
    scheduler = optimizer_scheduler.get_linear_schedule_with_warmup(optimizer,args_.warmup_iteration,len(training_loader)*args_.epochs)

    
    net.train()
    net.to(device)
    eval_=Evaluation_metrics(args_)
    print('\nStarting training loop ...')
    for epoch in tqdm(range(args_.epochs)):
        
        training_metrics=train_epoch(training_loader,device, optimizer,net,loss,scheduler,
                                     length_context_text,length_context_video)
        monitor.log_train(epoch +1,training_metrics)
        summary.write_train(training_metrics,epoch)
        
        net.eval()
        validation_metrics=validation_epoch(testing_loader,device, eval_,net,loss,
                                            length_context_text,length_context_video)
        monitor.log_val(epoch +1,validation_metrics)
        summary.write_test(validation_metrics,epoch)
        
        model_utils.save_checkpoint({
                    "epoch": epoch+1,
                    "state_dict": net.state_dict(),
                    "best_score": validation_metrics[args_.Best_metric],
                    "metrics":validation_metrics,
                    "optimizer": optimizer.state_dict(),
                    "scheduler":scheduler.state_dict()
                })

        net.train()


if __name__== "__main__":

    # create parser
    args_=parsing()
    train()

