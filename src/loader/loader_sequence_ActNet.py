from torch.utils.data import Dataset
import torch as th
import numpy as np
import pandas as pd  
import itertools
from natsort import natsorted
import random
import pickle

class FeatureLoader(Dataset):
    
    def __init__(self,path_dataframe,path_features_video,path_features_text,dataset,len_video,len_text,transform=None):
        
        self.df=pd.read_csv(path_dataframe).values[:,1:]
        self.dataset=dataset
        
        with open(path_features_video, 'rb') as fp:
            self.video_feat = np.array(pickle.load(fp))
            
        self.len_text=len_text
        self.len_video=len_video
        
        self.sequences=self.create_sequences(len_sequence=max(self.len_video,self.len_text),pad=max(self.len_video,self.len_text)//2)
        
        self.narrations=self.df[:,4]
        
        with open(path_features_text, 'rb') as fp:
            self.text_features = np.array(pickle.load(fp))
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
        
        id_clip=self.df[:,0]
        id_video=self.df[:,1]
        
        index_videos=sorted(np.unique(id_video, return_index=True)[1].tolist() + [len(id_clip)])

        index_clip={v:k for k,v in enumerate(id_clip)}
        sorted_clip=[index_clip[elem] for elem in natsorted(id_clip)]
        
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
            negative_texts=self.text_features[idx_neighbouring_negative_sequence,:]
            negative_videos=self.video_feat[idx_neighbouring_negative_sequence,:]
        except:
            negative_texts=np.zeros_like(self.text_features[idx_sequence,:])
            negative_videos=np.zeros_like(self.video_feat[idx_sequence,:])
        
        if self.negative_neighbouring[idx]:
            mask=True
        else:
            mask=False
            
        if self.len_text == self.len_video:
            sample= {"video_embed": self.video_feat[idx_sequence,:], "text_embed": self.text_features[idx_sequence,:],"negatives_texts":negative_texts,"negatives_videos":negative_videos,"masking":mask}

        elif self.len_text==1:
            sample= {"video_embed": self.video_feat[idx_sequence,:], "text_embed": self.text_features[idx,:], "negatives_videos":negative_videos,"masking":mask}

        elif self.len_video==1:
            sample= {"video_embed": self.video_feat[idx,:], "text_embed": self.text_features[idx_sequence,:], "negatives_texts":negative_texts,"masking":mask}    

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):        
        sample["video_embed"]=th.from_numpy(sample["video_embed"]).float()
        sample["text_embed"]=th.from_numpy(sample["text_embed"]).float()
        try:
            sample["negatives_texts"]=th.from_numpy(sample["negatives_texts"]).float()
        except:
            pass
        
        try:
            sample["negatives_videos"]=th.from_numpy(sample["negatives_videos"]).float()
        except:
            pass
        return sample
