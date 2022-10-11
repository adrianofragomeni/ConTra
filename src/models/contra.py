#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as th
import torch.nn as nn
import models.layers as layers
from models.text_model import Sentence_Embedding
import torch.nn.functional as F
from pathlib import Path

class Transformer(nn.Module):
    def __init__(self,d_model, nhead,dim_feedforward, dropout,num_layers,length):
        super().__init__()
        
        self.positional_embedding = nn.Parameter(th.zeros(length, d_model))
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.001)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        
        self.d=nn.Dropout(dropout)
    def forward(self,x,mask):
        
        x = x + self.positional_embedding
        x = self.transformer_encoder(self.d(x.permute(1,0,2)),mask=mask)

        return x


class ConTraModel(nn.Module):
    def __init__(self,
                 args_,
                 embd_dim,
                 dropout_text,
                 dropout_video,
                 nhead_text,
                 nhead_video,
                 dim_feedforward_text,
                 dim_feedforward_video,
                 nlayer_text,
                 nlayer_video,
                 length_text, 
                 length_video,
                 device):
        
        super().__init__()
        
        self.dataset=args_.dataset
        self.length_text=length_text
        self.length_video=length_video
        self.device=device
        
        self.text_module=Sentence_Embedding(args_)
        self.d_text=nn.Dropout(dropout_text)
        self.embd_text = layers.Linear_Unit(embd_dim*4, embd_dim,add_batch_norm=False)
        self.transformer_encoder_text =Transformer(d_model=embd_dim, nhead=nhead_text,
                                                   dim_feedforward=dim_feedforward_text, dropout=dropout_text,
                                                   num_layers=nlayer_text,length=self.length_text)
         
        self.d_video=nn.Dropout(dropout_video)
        
        if self.dataset=="YC2":
            self.embd_video = layers.Linear_Unit(embd_dim*2, embd_dim,add_batch_norm=False) 
        
        elif self.dataset=="Epic":
            self.embd_video = layers.Linear_Unit(embd_dim*6, embd_dim,add_batch_norm=False)
            
        self.transformer_encoder_video =Transformer(d_model=embd_dim, nhead=nhead_video,
                                                    dim_feedforward=dim_feedforward_video, dropout=dropout_video,
                                                    num_layers=nlayer_video,length=self.length_video)

    def forward(self, narration, video_emb,branch,mask=None):
        
        if branch=="all":
            
            text_emb=self.text_module(narration,self.device)
            text_emb=self.embd_text(self.d_text(text_emb.view(video_emb.shape[0],self.length_text,-1)))
            text_emb = self.transformer_encoder_text(text_emb,mask)

            video_emb=self.embd_video(self.d_video(video_emb.view(video_emb.shape[0],self.length_video,-1)))
            video_emb= self.transformer_encoder_video(video_emb,mask) 

            return F.normalize(text_emb[self.length_text//2,:,:]),F.normalize(video_emb[self.length_video//2,:,:])
        
        elif branch=="video":

            video_emb=self.embd_video(self.d_video(video_emb.view(video_emb.shape[0],self.length_video,-1)))
            video_emb = self.transformer_encoder_video(video_emb,mask) 

            return F.normalize(video_emb[self.length_video//2,:,:])
        
        elif branch=="text":
            
            text_emb=self.text_module(narration,self.device)
            text_emb=self.embd_text(self.d_text(text_emb.view(video_emb.shape[0],self.length_text,-1)))
            text_emb= self.transformer_encoder_text(text_emb,mask)
           
            return F.normalize(text_emb[self.length_text//2,:,:])

    @staticmethod
    def pretrained_text_model(model,path_resources):
        
        pretrained_dict = th.load(Path(path_resources) /"s3d_howto100m.pth")
        
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)    
        
        model.text_module.update_embeddings()    
        
        return model    

