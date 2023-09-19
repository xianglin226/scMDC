from time import time
import math, os
from sklearn import metrics
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scMDC import scMultiCluster
import numpy as np
import collections
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize, clr_normalize_each_cell
from utils import *
from functools import reduce
from LRP import LRP


if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=8, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='Simulation.1.h5')
    parser.add_argument('--cluster_index_file', default='label.txt')
    parser.add_argument('--maxiter', default=10000, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--gamma', default=.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float,
                        help='fuzziness of clustering loss')       
    parser.add_argument('--phi1', default=0.001, type=float,
                        help='coefficient of KL loss in pretraining stage')
    parser.add_argument('--phi2', default=0.001, type=float,
                        help='coefficient of KL loss in clustering stage')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/')
    parser.add_argument('--ae_weight_file', default='AE_weights_1.pth.tar')
    parser.add_argument('--resolution', default=0.2, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('--embedding_file', action='store_true', default=False)
    parser.add_argument('--prediction_file', action='store_true', default=False)
    parser.add_argument('-el','--encodeLayer', nargs='+', default=[256,64,32,16])
    parser.add_argument('-dl1','--decodeLayer1', nargs='+', default=[16,64,256])
    parser.add_argument('-dl2','--decodeLayer2', nargs='+', default=[16,20])
    parser.add_argument('--sigma1', default=2.5, type=float)
    parser.add_argument('--sigma2', default=1.5, type=float)
    parser.add_argument('--f1', default=1000, type=float, help='Number of mRNA after feature selection')
    parser.add_argument('--f2', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
    parser.add_argument('--filter1', action='store_true', default=False, help='Do mRNA selection')
    parser.add_argument('--filter2', action='store_true', default=False, help='Do ADT/ATAC selection')
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--beta', default=1., type=float,
                        help='coefficient of the clustering fuzziness')
    parser.add_argument('--margin', default=1., type=float,
                        help='margin of difference between logits')
    parser.add_argument('--lamda', default=100., type=float,
                        help='coefficient of the clustering perturbation loss')
    parser.add_argument('--lr', default=0.001, type=int)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    print(args)
    data_mat = h5py.File(args.data_file)
    x1 = np.array(data_mat['X1'])
    x2 = np.array(data_mat['X2'])
    #y = np.array(data_mat['Y']) - 1
    data_mat.close()
    

    clust_ids = np.loadtxt(args.cluster_index_file, delimiter=",").astype(int)

    #Gene features
    if args.filter1:
        importantGenes = geneSelection(x1, n=args.f1, plot=False)
        x1 = x1[:, importantGenes]
    if args.filter2:
        importantGenes = geneSelection(x2, n=args.f2, plot=False)
        x2 = x2[:, importantGenes]

    adata1 = sc.AnnData(x1)
    #adata1.obs['Group'] = y

    adata1 = read_dataset(adata1,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata1 = normalize(adata1,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    adata2 = sc.AnnData(x2)
    #adata2.obs['Group'] = y
    adata2 = read_dataset(adata2,
                     transpose=False,
                     test_split=False,
                     copy=True)
    
    adata2 = normalize(adata2,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    #adata2 = clr_normalize_each_cell(adata2)

    input_size1 = adata1.n_vars
    input_size2 = adata2.n_vars
    print(adata1.X.shape)
    print(adata2.X.shape)

    print(args)
    
    encodeLayer = list(map(int, args.encodeLayer))
    decodeLayer1 = list(map(int, args.decodeLayer1))
    decodeLayer2 = list(map(int, args.decodeLayer2))
    
    model = scMultiCluster(input_dim1=input_size1, input_dim2=input_size2, tau=args.tau,
                        encodeLayer=encodeLayer, decodeLayer1=decodeLayer1, decodeLayer2=decodeLayer2,
                        activation='elu', sigma1=args.sigma1, sigma2=args.sigma2, gamma=args.gamma, 
                        cutoff = args.cutoff, phi1=args.phi1, phi2=args.phi2, device=args.device).to(args.device)
    
    print(str(model))

    if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
    else:
        print("==> no checkpoint found at '{}'".format(args.ae_weights))
        raise ValueError
    
    n_clusters = np.unique(clust_ids).shape[0]
    print("n cluster is: " + str(n_clusters))

    Z = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device)).data.cpu().numpy()
    
    cluster_list = np.unique(clust_ids).astype(int).tolist()
    print(cluster_list)

    model_explainer = LRP(model, X1=adata1.X, X2=adata2.X, Z=Z, clust_ids=clust_ids, n_clusters=n_clusters, beta=args.beta).to(args.device)
    
    #for clust_c in [cluster_ind[0]]: #range(args.n_clusters):
    #    for clust_k in [cluster_ind[1]]: #range(clust_c+1, args.n_clusters):
    #        print("Cluster"+str(clust_c)+" vs Cluster"+str(clust_k))
    #        rel_score1, rel_score2  = model_explainer.calc_carlini_wagner_one_vs_one(clust_c, clust_k, margin=args.margin, lamda=args.lamda, max_iter=args.maxiter, lr=args.lr)
    #        print(rel_score1.shape)
    #        print(rel_score2.shape)
    #        np.savetxt(args.save_dir + "/" + str(clust_c)+"_vs_"+str(clust_k)+"_rel_mRNA_scores.csv", rel_score1, delimiter=",")
    #        np.savetxt(args.save_dir + "/" + str(clust_c)+"_vs_"+str(clust_k)+"_rel_ADT_scores.csv", rel_score2, delimiter=",")

    for clust_c in cluster_list:
        print("Cluster"+str(clust_c)+" vs Rest")
        rel_score1, rel_score2 = model_explainer.calc_carlini_wagner_one_vs_rest(clust_c, margin=args.margin, lamda=args.lamda, max_iter=args.maxiter, lr=args.lr)
        np.savetxt(args.save_dir + "/" + str(clust_c)+"_vs_rest_rel_mRNA_scores.csv", rel_score1, delimiter=",")
        np.savetxt(args.save_dir + "/" + str(clust_c)+"_vs_rest_rel_ADT_scores.csv", rel_score2, delimiter=",")
