from time import time
import math, os
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scMDC_batch import scMultiClusterBatch
import numpy as np
import collections
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize
from utils import *

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=27, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='Normalized_filtered_BMNC_GSE128639_Seurat.h5')
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--gamma', default=.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float,
                        help='weight of clustering loss')
    parser.add_argument('--phi1', default=0.001, type=float,
                        help='coefficient of KL loss in pretraining stage')
    parser.add_argument('--phi2', default=0.001, type=float,
                        help='coefficient of KL loss in clustering stage')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--lr', default=1., type=float)
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
    parser.add_argument('--nbatch', default=2, type=int)
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_labels', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    data_mat = h5py.File(args.data_file)
    x1 = np.array(data_mat['X1'])
    x2 = np.array(data_mat['X2'])
    if not args.no_labels:
         y = np.array(data_mat['Y'])
    b = np.array(data_mat['Batch'])
    enc = OneHotEncoder()
    enc.fit(b.reshape(-1, 1))
    B = enc.transform(b.reshape(-1, 1)).toarray()
    data_mat.close()
    
    #Gene filter
    if args.filter1:
        importantGenes = geneSelection(x1, n=args.f1, plot=False)
        x1 = x1[:, importantGenes]
    if args.filter2:
        importantGenes = geneSelection(x2, n=args.f2, plot=False)
        x2 = x2[:, importantGenes]

    # preprocessing scRNA-seq read counts matrix
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

    input_size1 = adata1.n_vars
    input_size2 = adata2.n_vars
    
    print(args)
    
    encodeLayer = list(map(int, args.encodeLayer))
    decodeLayer1 = list(map(int, args.decodeLayer1))
    decodeLayer2 = list(map(int, args.decodeLayer2))
    
    model = scMultiClusterBatch(input_dim1=input_size1, input_dim2=input_size2, n_batch = args.nbatch, tau=args.tau,
                        encodeLayer=encodeLayer, decodeLayer1=decodeLayer1, decodeLayer2=decodeLayer2,
                        activation='elu', sigma1=args.sigma1, sigma2=args.sigma2, gamma=args.gamma,
                        cutoff = args.cutoff, phi1=args.phi1, phi2=args.phi2, device=args.device).to(args.device)
    
    print(str(model))
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, 
                X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, B = B, batch_size=args.batch_size, 
                epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    print('Pretraining time: %d seconds.' % int(time() - t0))
    
    #get k
    latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device), torch.tensor(B).to(args.device), batch_size=args.batch_size)
    latent = latent.cpu().numpy()
    if args.n_clusters == -1:
       n_clusters = GetCluster(latent, res=args.resolution, n=args.n_neighbors)
    else:
       print("n_cluster is defined as " + str(args.n_clusters))
       n_clusters = args.n_clusters

    if not args.no_labels:
          y_pred,_ = model.fit(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, 
             X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, B=B, y=y,
             n_clusters=n_clusters, batch_size=args.batch_size, num_epochs=args.maxiter, 
             update_interval=args.update_interval, tol=args.tol, lr=args.lr, save_dir=args.save_dir)
    else:
          y_pred,_ = model.fit(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, 
             X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, B=B, y=None,
             n_clusters=n_clusters, batch_size=args.batch_size, num_epochs=args.maxiter, 
             update_interval=args.update_interval, tol=args.tol, lr=args.lr, save_dir=args.save_dir)
    print('Total time: %d seconds.' % int(time() - t0))
    
    if args.prediction_file:
       np.savetxt(args.save_dir + "/" + str(args.run) + "_pred.csv", y_pred, delimiter=",")
    
    if args.embedding_file:
       final_latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device), torch.tensor(B).to(args.device), batch_size=args.batch_size)
       final_latent = final_latent.cpu().numpy()
       np.savetxt(args.save_dir + "/" + str(args.run) + "_embedding.csv", final_latent, delimiter=",")

    if not args.no_labels:
        y_pred_ = best_map(y, y_pred)
        ami = np.round(metrics.adjusted_mutual_info_score(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print('Final: AMI= %.4f, NMI= %.4f, ARI= %.4f' % (ami, nmi, ari))
    else:
         print("No labels for evaluation!")
