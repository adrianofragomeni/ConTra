
import torch as th
from pathlib import Path
from utils.utils import get_default_device, parsing
from torch.utils.data import DataLoader
from loader.loader_sequence import FeatureLoader, ToTensor
from evaluation.evaluation import Evaluation_metrics
from visualisation.qualitative_results import visualization_retrieval
import numpy as np
import itertools
from prettytable import PrettyTable
from models.contra import ConTraModel


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def eval_model(qualitative_results=False):
    
    device= get_default_device()
    print("load features...")
            
    dataset=FeatureLoader(path_dataframe=Path(args_.path_dataframes) / args_.dataset / "retrieval_test.csv",
                              path_features=Path(args_.path_features) / args_.dataset / "validation_video_features.pkl",                              
                              dataset=args_.dataset,
                              len_text=args_.length_context_text,
                              len_video=args_.length_context_video,
                              transform=ToTensor())
    loader = DataLoader(dataset,
                        batch_size=args_.batch_size,
                        pin_memory=True,
                        shuffle=False,
                        num_workers=args_.cpu_count)


    print("load model...")
        
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
                     args_.length_context_text, 
                     args_.length_context_video,
                     device)
    
    net=net.pretrained_text_model(net,args_.path_resources)
    net.eval()
    
    # load best model
    best_model = th.load(Path(args_.path_model)/"model_best.pth.tar",map_location="cuda")
    best_epoch=best_model["epoch"]
    print("Best Epoch: {}".format(best_epoch))
    net.load_state_dict(best_model['state_dict'])
    
    net.to(device)
    
    print(count_parameters(net))
    
    eval_=Evaluation_metrics(args_)
    
    all_text_embed=[]
    all_video_embed=[]
    
    with th.no_grad():
        for i_batch, sample_batched in enumerate(loader):
            
            video_features=sample_batched["video_embed"].to(device)

            text_features=list(itertools.chain(*np.array(sample_batched["text_embed"],dtype=object).T.tolist()))
            text_embed,video_embed=net(text_features,video_features,branch="all")
            
            all_video_embed.append(video_embed.cpu().numpy())
            all_text_embed.append(text_embed.cpu().numpy())
        
        all_video_embed=np.vstack(all_video_embed)
        all_text_embed=np.vstack(all_text_embed)
        
        similarity_matrix= np.matmul(all_text_embed,all_video_embed.T)     

        metrics=eval_.get_metrics(similarity_matrix)
        print(metrics)
    

    if qualitative_results: 
        return visualization_retrieval(similarity_matrix=similarity_matrix[:,eval_.indexes],
                                       task="t2v",
                                       path_dataframe=Path(args_.path_dataframes) / args_.dataset / "retrieval_test.csv",
                                       path_unique_sentences=Path(args_.path_dataframes) / args_.dataset / "retrieval_test_sentence.csv",
                                       path_relevancy=Path(args_.path_relevancy) / args_.dataset / "instance_relevancy_test.pkl")
    
if __name__== "__main__":

    # create parser
    args_=parsing()
    results=eval_model()
    
