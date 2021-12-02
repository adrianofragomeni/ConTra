import argparse 
import torch as th


def parsing():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, help="Number of epochs",
                        default=1000)
    
    parser.add_argument("--batch-size", type=int, help="Batch size",
                        default=512)
    
    parser.add_argument("--cpu-count", type=int, help="Num workers",
                        default=0)
    
    parser.add_argument("--embed-dim", type=int, help="embedding size",
                        default=512)
    
    parser.add_argument("--temperature", type=float, help="Temperature parameter for contrastive loss",
                        default=0.07)
    
    parser.add_argument('--lr', type=float, help='initial learning rate',
                        default=1e-4)
    
    parser.add_argument("--path-features", type=str, help="path of the features",
                        default="../data/features")

    parser.add_argument("--path-dataframes", type=str, help="path of the dataframes",
                        default="../data/dataframes")

    parser.add_argument("--path-relevancy", type=str, help="path of the relevancy matrix",
                        default="../data/relevancy")    
    
    parser.add_argument("--path-resources", type=str, help="path of the resources",
                        default="../data/resources")  

    parser.add_argument("--path-model", type=str, help="Path of the saved model",
                        default="../data/models/prova")

    parser.add_argument("--warmup-iteration", type=int, help="number of iteration to apply warmup",
                        default=1300)

    parser.add_argument("--dataset", type=str, help="Choose the datasetet (YC2, Epic, ActNet)",
                        default="Epic")

    parser.add_argument("--dropout-text", type=float, help="dropout text input",
                        default=0.3)

    parser.add_argument("--dropout-video", type=int, help="dropout video input",
                        default=0.3)
    
    parser.add_argument("--nlayer-text", type=int, help="number of layer text transformer",
                        default=2)

    parser.add_argument("--nlayer-video", type=int, help="number of layer video transformer",
                        default=2)

    parser.add_argument("--nhead-text", type=int, help="number of heads  text transformer",
                        default=8)    

    parser.add_argument("--nhead-video", type=int, help="number of heads  video transformer",
                        default=8)

    parser.add_argument("--dim-feedforward-text", type=int, help="dimension feedforward text transformer",
                        default=2048)
    
    parser.add_argument("--dim-feedforward-video", type=int, help="dimension feedforward video transformer",
                        default=2048)
    
    parser.add_argument("--length-context-video", type=int, help="length of context video",
                        default=1)
    
    parser.add_argument("--length-context-text", type=int, help="length of context text",
                        default=1)

    parser.add_argument("--lambda1", type=int, help="weight of the CML term in the objective function",
                        default=1)
    
    parser.add_argument("--lambda2", type=int, help="weight of the UNI term in the objective function",
                        default=1)
    
    parser.add_argument("--lambda3", type=int, help="weight of the text NEI term in the objective function",
                        default=0)

    parser.add_argument("--lambda4", type=int, help="weight of the video NEI term in the objective function",
                        default=0)
    
    parser.add_argument("--Best-metric", type=str, help="Choose the best metric (RSum, RSum_v2t, RSum_t2v)",
                        default="RSum")    
    
    args_= parser.parse_args()
    
    return args_



def get_default_device():
    """ Pick GPU or CPU as device"""
    
    if th.cuda.is_available():
        device=th.device("cuda")
    else:
        device=th.device("cpu")
    return device


