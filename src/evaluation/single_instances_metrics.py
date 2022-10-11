import numpy as np

def single_instance_metrics(similarity_matrix,relevance_matrix, type_retrieval):

    x_sz, y_sz = similarity_matrix.shape
    # find the ranks of each item
    ranks = np.argsort(similarity_matrix)[:, ::-1]
    columns = np.repeat(np.expand_dims(np.arange(x_sz), axis=1), y_sz, axis=1)
    
    #Rerank relevancy matrix
    ind = relevance_matrix[columns, ranks]> 0.9999


    metrics = {}
    metrics['R1_{}'.format(type_retrieval)] = np.round(np.mean(np.any(ind[:,:1],1))*100,2)
    metrics['R5_{}'.format(type_retrieval)] = np.round(np.mean(np.any(ind[:,:5],1))*100,2)
    metrics['R10_{}'.format(type_retrieval)] = np.round(np.mean(np.any(ind[:,:10],1))*100,2)
    metrics['MR_{}'.format(type_retrieval)] = np.median((ind==1).argmax(axis=1)) + 1
    
    return metrics
