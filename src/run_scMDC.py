from time import time
import math, os
from sklearn import metrics
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
from preprocess import read_dataset, normalize
from utils import cluster_acc, GetCluster

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--cutoff1', default=0.3, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--cutoff2', default=0.3, type=float, help='Start to train combined layer after what ratio of batch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='./realdata/10X_PBMC_newCount_filtered_1000G.H5')
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=600, type=int)
    parser.add_argument('--gamma1', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--gamma2', default=.1, type=float,
                        help='coefficient of latent autoencoder loss')                    
    parser.add_argument('--gamma3', default=.001, type=float,
                        help='coefficient of KL loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--lr', default=.01, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/')
    parser.add_argument('--ae_weight_file', default='AE_weights_1.pth.tar')
    parser.add_argument('--resolution', default=0.2, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('--embedding_file', default=-1)
    parser.add_argument('--prediction_file', default=-1)
    parser.add_argument('-l1','--encodeLayer1', nargs='+', default=[256,128,64])
    parser.add_argument('-l2','--encodeLayer2', nargs='+', default=[8])
    parser.add_argument('-l3','--encodeLayer3', nargs='+', default=[64,16])
    parser.add_argument('--sigma1', default=3., type=float)
    parser.add_argument('--sigma2', default=0.2, type=float)

    args = parser.parse_args()
    print(args.gamma1)
    print(args.gamma2)
    print(args.gamma3)
    data_mat = h5py.File(args.data_file)
    x1 = np.array(data_mat['X1'])
    x2 = np.array(data_mat['X2'])
    y = np.array(data_mat['Y'])
    data_mat.close()

    # preprocessing CITE-seq read counts matrix
    adata1 = sc.AnnData(x1)
    adata1.obs['Group'] = y

    adata1 = read_dataset(adata1,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata1 = normalize(adata1,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    adata2 = sc.AnnData(x2)
    adata2.obs['Group'] = y
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

    print(adata1.X.shape)
    print(adata2.raw.X.shape)
    print(x2.shape)
    print(adata2.X.shape)
    print(y.shape)
    
    encodeLayer1 = list(map(int, args.encodeLayer1))
    decodeLayer1 = encodeLayer1[::-1]
    encodeLayer2 = list(map(int, args.encodeLayer2))
    if len(encodeLayer2) >1:
       decodeLayer2 = encodeLayer2[::-1]
    else:
       decodeLayer2 = encodeLayer2
       
    encodeLayer3 = list(map(int, args.encodeLayer3))
    if len(encodeLayer3) >1:
       decodeLayer3 = encodeLayer3[::-1]
    else:
       decodeLayer3 = encodeLayer3
       
    model = scMultiCluster(input_dim1=input_size1, input_dim2=input_size2,
                        zencode_dim=encodeLayer3, zdecode_dim=decodeLayer3, 
                        encodeLayer1=encodeLayer1, decodeLayer1=decodeLayer1, encodeLayer2=encodeLayer2, decodeLayer2=decodeLayer2,
                        sigma1=args.sigma1, sigma2=args.sigma2, gamma1=args.gamma1, gamma2=args.gamma2, gamma3=args.gamma3, cutoff1 = args.cutoff1, cutoff2 = args.cutoff2).cuda()
    
    print(str(model))
    
    #pretraing stage
    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    
    #estimate k
    latent = model.encodeBatch(torch.tensor(adata1.X).cuda(), torch.tensor(adata2.X).cuda()).cpu().numpy()
    if args.n_clusters == -1:
       n_clusters = GetCluster(latent, res=args.resolution, n=args.n_neighbors)
    else:
       print("n_cluster is defined as " + str(args.n_clusters))
       n_clusters = args.n_clusters
    
    #clustering stage
    y_pred, _, _, _, _ = model.fit(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, y=y, n_clusters=n_clusters, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, lr=args.lr, save_dir=args.save_dir)
    print('Total time: %d seconds.' % int(time() - t0))
    
    if args.prediction_file != -1:
       np.savetxt(args.prediction_file + "_pred.csv", y_pred, delimiter=",")
    
    if args.embedding_file != -1:
       final_latent = model.encodeBatch(torch.tensor(adata1.X).cuda(), torch.tensor(adata2.X).cuda()).cpu().numpy()
       np.savetxt(args.embedding_file + "_embedding.csv", final_latent, delimiter=",")

    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('Final: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
