import pandas as pd
import pickle5 as pickle
from evaluation.single_instances_metrics import single_instance_metrics
from pathlib import Path

class Evaluation_metrics:
    """ Util class for evaluation. """

    def __init__(self,args_):
         
        with open(Path(args_.path_relevancy) / args_.dataset / "instance_relevancy_test.pkl", 'rb') as fp:
            self.relevance_matrix = pickle.load(fp)

        id_clip=pd.read_csv(Path(args_.path_dataframes) / args_.dataset / "retrieval_test.csv").values[:,0]
        considered_clip=pd.read_csv(Path(args_.path_dataframes) / args_.dataset / "retrieval_test_sentence.csv").values[:,0]
        
        self.indexes=[]
        for elem in considered_clip:
            self.indexes.append(id_clip.tolist().index(elem))

        
    def get_measures_t2v(self,similarity_matrix=None):
        """ Return t2v results """
        return single_instance_metrics(similarity_matrix[self.indexes,:], self.relevance_matrix.T,"t2v")
    
    
    def get_measures_v2t(self,similarity_matrix=None):
        """ Return v2t results """

        return single_instance_metrics(similarity_matrix[self.indexes,:].T, self.relevance_matrix,"v2t")

    
    def get_metrics(self,similarity_matrix=None):
        
        results_t2v=self.get_measures_t2v(similarity_matrix)
        results_v2t=self.get_measures_v2t(similarity_matrix)
            
        metrics={**results_t2v, **results_v2t}
        metrics["RSum_t2v"]=results_t2v["R1_t2v"]+results_t2v["R5_t2v"]+results_t2v["R10_t2v"]
        metrics["RSum_v2t"]=results_v2t["R1_v2t"]+results_v2t["R5_v2t"]+results_v2t["R10_v2t"]
        metrics["RSum"]=results_t2v["R1_t2v"]+results_t2v["R5_t2v"]+results_t2v["R10_t2v"]+results_v2t["R1_v2t"]+results_v2t["R5_v2t"]+results_v2t["R10_v2t"]

        
        return metrics
    
