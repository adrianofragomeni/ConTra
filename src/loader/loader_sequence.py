from torch.utils.data import Dataset
import torch as th
import numpy as np
import pandas as pd  
import itertools
from natsort import natsorted
import random
import pickle

class FeatureLoader(Dataset):
    
    def __init__(self,path_dataframe,path_features,dataset,len_video,len_text,transform=None):
        
        if dataset=="YC2":
            df=pd.read_csv(path_dataframe).values
        
        elif dataset=="Epic":
            df=pd.read_csv(path_dataframe,usecols = ["narration_id","video_id","start_frame",
                                                     "stop_frame","narration"]).values
            
        self.id_clip=df[:,0]
        self.id_video=df[:,1]
        self.narrations=df[:,4]
        
        with open(path_features, 'rb') as fp:
            self.video_feat = np.array(pickle.load(fp)["features"])
            
        self.len_text=len_text
        self.len_video=len_video
        
        self.sequences=self.create_sequences(len_sequence=max(self.len_video,self.len_text),pad=max(self.len_video,self.len_text)//2)

        self.negative_neighbouring=self.create_negative_neighbouring()
        
        self.transform=transform


    def create_negative_neighbouring(self):
        negatives_neighbouring={}
        for i in range(len(self.sequences)):
            texts=list(self.sequences[i])
            new_texts=[]
            n_middle=self.narrations[texts[len(texts)//2]]
            indices = [i for i, x in enumerate(self.narrations[texts].tolist()) if x == n_middle]
            for ind in range(len(texts)):
                if ind in indices:
                    continue
                else:
                    new_texts.append(texts[ind])           
            negatives_neighbouring[i]=new_texts
        return negatives_neighbouring
    
    
    def create_sequences(self,len_sequence, pad):
        # len,pad=3,1 --- len,pad=5,2 --- len,pad=7,3
        def group(iterable, n):
            "group(s, 3) -> (s0, s1, s2), (s1, s2, s3), (s2, s3, s4), ..."
            itrs = itertools.tee(iterable, n)
            for i in range(1, n):
                for itr in itrs[i:]:
                    next(itr, None)
            return zip(*itrs)
        
        index_videos=sorted(np.unique(self.id_video, return_index=True)[1].tolist() + [len(self.id_clip)])

        index_clip={v:k for k,v in enumerate(self.id_clip)}
        sorted_clip=[index_clip[elem] for elem in natsorted(self.id_clip)]
        
        sequences_index=[]
        for i in range(1,len(index_videos)):
            video=[sorted_clip[index_videos[i-1]]]*pad + sorted_clip[index_videos[i-1]:index_videos[i]] + [sorted_clip[index_videos[i]-1]]*pad
            
            for current_elem in group(video,len_sequence):
                sequences_index.append(current_elem)
        
        return dict(zip(sorted_clip,sequences_index))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self,sequence_idx):
        
        if th.is_tensor(sequence_idx):
            sequence_idx=sequence_idx.tolist()
        
        idx_sequence=self.sequences[sequence_idx]
        idx=idx_sequence[len(idx_sequence)//2]
        
        try:
            idx_negative=random.choice(self.negative_neighbouring[idx])
            idx_neighbouring_negative_sequence=self.sequences[idx_negative]
            negative_texts=self.narrations[list(idx_neighbouring_negative_sequence)].tolist()
            negative_videos=self.video_feat[idx_neighbouring_negative_sequence,:]
        except:
            negative_texts=[""]*max(self.len_video,self.len_text)
            negative_videos=np.zeros_like(self.video_feat[idx_sequence,:])
        
        if self.negative_neighbouring[idx]:
            mask=True
        else:
            mask=False
            
        if self.len_text == self.len_video:
            sample= {"video_embed": self.video_feat[idx_sequence,:], "text_embed": self.narrations[list(idx_sequence)].tolist(),
                     "negatives_texts":negative_texts,"negatives_videos":negative_videos,"masking":mask}
            
        elif self.len_text==1:
            sample= {"video_embed": self.video_feat[idx_sequence,:], "text_embed": self.narrations[[idx]].tolist(), 
                     "negatives_videos":negative_videos,"masking":mask}
            
        elif self.len_video==1:
            sample= {"video_embed": self.video_feat[idx,:], "text_embed": self.narrations[list(idx_sequence)].tolist(), 
                     "negatives_texts":negative_texts,"masking":mask}    
   
            
        if self.transform:
            sample = self.transform(sample,self.len_video)
        
        return sample
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample,len_video):        
        sample["video_embed"]=th.from_numpy(sample["video_embed"]).float()
        try:
            sample["negatives_videos"]=th.from_numpy(sample["negatives_videos"]).float()
        except:
            pass
        
        return sample
