import os
import sys
import time
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from scipy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances
import scanpy as sc

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    #from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def GetCluster(X, res, n):
    adata0=sc.AnnData(X)
    if adata0.shape[0]>200000:
       np.random.seed(adata0.shape[0])#set seed 
       adata0=adata0[np.random.choice(adata0.shape[0],200000,replace=False)] 
    sc.pp.neighbors(adata0, n_neighbors=n, use_rep="X")
    sc.tl.louvain(adata0,resolution=res)
    Y_pred_init=adata0.obs['louvain']
    Y_pred_init=np.asarray(Y_pred_init,dtype=int)
    if np.unique(Y_pred_init).shape[0]<=1:
        #avoid only a cluster
        exit("Error: There is only a cluster detected. The resolution:"+str(res)+"is too small, please choose a larger resolution!!")
    else: 
        print("Estimated n_clusters is: ", np.shape(np.unique(Y_pred_init))[0]) 
    return(np.shape(np.unique(Y_pred_init))[0])

def torch_PCA(X, k, center=True, scale=False):
    X = X.t()
    n,p = X.size()
    ones = torch.ones(n).cuda().view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
    H = torch.eye(n).cuda() - h
    X_center =  torch.mm(H.double(), X.double())
    covariance = 1/(n-1) * torch.mm(X_center.t(), X_center).view(p,p)
    scaling = torch.sqrt(1/torch.diag(covariance)).double() if scale else torch.ones(p).cuda().double()
    scaled_covariance = torch.mm(torch.diag(scaling).view(p,p), covariance)
    eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
    components = (eigenvectors[:, :k])
    #explained_variance = eigenvalues[:k, 0]
    return components
