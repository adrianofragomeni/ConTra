#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as th
import torch.nn as nn
import torch.nn.functional as F

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
        
        self.length_text=length_text
        self.length_video=length_video
        self.device=device
        
        self.d_text=nn.Dropout(dropout_text)
        self.transformer_encoder_text =Transformer(d_model=embd_dim, nhead=nhead_text,dim_feedforward=dim_feedforward_text, dropout=dropout_text,num_layers=nlayer_text,length=length_text)
         
        self.d_video=nn.Dropout(dropout_video)
        
        self.transformer_encoder_video =Transformer(d_model=embd_dim, nhead=nhead_video,dim_feedforward=dim_feedforward_video, dropout=dropout_video,num_layers=nlayer_video,length=length_video)

    def forward(self, narration, video_emb,branch,mask=None):
        
        if branch=="all":
            text_emb= self.transformer_encoder_text(narration.view(video_emb.shape[0],self.length_text,-1),mask)
            video_emb = self.transformer_encoder_video(video_emb.view(video_emb.shape[0],self.length_video,-1),mask) 
            
            return F.normalize(text_emb[self.length_text//2,:,:]),F.normalize(video_emb[self.length_video//2,:,:])
        
        elif branch=="video":
            video_emb= self.transformer_encoder_video(video_emb.view(video_emb.shape[0],self.length_video,-1),mask) 

            return F.normalize(video_emb[self.length_video//2,:,:])
        
        elif branch=="text":
            text_emb = self.transformer_encoder_text(narration.view(video_emb.shape[0],self.length_text,-1),mask)
            return F.normalize(text_emb[self.length_text//2,:,:])


