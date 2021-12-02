import numpy as np
import pandas as pd
from collections import OrderedDict
import pickle5 as pickle


def visualization_retrieval(similarity_matrix,task,path_dataframe, path_unique_sentences, path_relevancy_mat, dataset):
        with open(path_relevancy_mat, 'rb') as fp:
            relevance_matrix = pickle.load(fp)
            
        df=pd.read_csv(path_dataframe).values
        id_clip=df[:,0]
        
        if dataset=="Epic":
            narrations=df[:,8]
        else:
            narrations=df[:,4]
        
        dicty_narrations=dict(zip(id_clip,narrations))
        
        df=pd.read_csv(path_unique_sentences).values
        considered_clip=df[:,0]
        unique_narrations=df[:,1]
        
        dicty_unique_narrations=dict(zip(considered_clip,unique_narrations))

        for key,value in dicty_unique_narrations.items():
            dicty_narrations[key]=value
        
        if task == "v2t":
            x_sz, y_sz = similarity_matrix.shape
            ranks = np.argsort(similarity_matrix)[:, ::-1]
            columns = np.repeat(np.expand_dims(np.arange(x_sz), axis=1), y_sz, axis=1)
            ind = relevance_matrix[columns, ranks]        
            results=unique_narrations[ranks]

            qualitative_results=OrderedDict()
            for pos in range(len(id_clip)):
                correct_pos_narration=np.where(ind[pos,:]==1)[0][0]
                qualitative_results[id_clip[pos]]=(results[pos,correct_pos_narration],correct_pos_narration+1,results[pos,:10])
    
        elif task == "t2v":
            
            similarity_matrix=similarity_matrix.T
            relevance_matrix=relevance_matrix.T
            x_sz, y_sz = similarity_matrix.shape
            ranks = np.argsort(similarity_matrix)[:, ::-1]
            columns = np.repeat(np.expand_dims(np.arange(x_sz), axis=1), y_sz, axis=1)
            ind = relevance_matrix[columns, ranks]        
            results_videos=id_clip[ranks]
        
            qualitative_results=OrderedDict()
            for pos in range(len(unique_narrations)):
                correct_pos_narration=np.where(ind[pos,:]==1)[0][0]
                qualitative_results[unique_narrations[pos]]=(results_videos[pos,:10],np.array(list(map(lambda x: dicty_narrations[x],results_videos[pos,:10]))),correct_pos_narration+1)
                

        return qualitative_results
    
    