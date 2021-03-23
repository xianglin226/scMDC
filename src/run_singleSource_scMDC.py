from time import time
import math, os
import scanpy as sc
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from singleSource_scMDC import scMultiCluster
import numpy as np
import collections
from sklearn import metrics
import h5py
from preprocess import read_dataset, normalize
from utils import cluster_acc, GetCluster


if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of batch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='./realdata/10X_PBMC_newCount_filtered_1000G.H5')
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=500, type=int)
    parser.add_argument('--gamma1', default=.001, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--gamma2', default=.001, type=float,
                        help='coefficient of KL loss')
    parser.add_argument('--gamma3', default=.1, type=float,
                        help='coefficient of latent autoencoder loss')
    parser.add_argument('--update_interval', default=100, type=int)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/')
    parser.add_argument('--ae_weight_file', default='AE_weights_1.pth.tar')
    parser.add_argument('--resolution', default=0.3, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('--embedding_file', default=-1)
    parser.add_argument('--prediction_file', default=-1)
    parser.add_argument('-l1','--encodeLayer1', nargs='+', default=[256,128,64])
    parser.add_argument('-l2','--encodeLayer2', nargs='+', default=[8])
    parser.add_argument('--sigma1', default=2.5, type=float)
    parser.add_argument('--sigma2', default=.1, type=float)
    parser.add_argument('--source', default="RNA")


    args = parser.parse_args()

    data_mat = h5py.File(args.data_file)
    x1 = np.array(data_mat['X1']) # test for single model, set x1 as X3
    x2 = np.array(data_mat['X2'])
    y = np.array(data_mat['Y'])
    x2_zero = [i for i, val in enumerate((x2==0).all(1)) if val]
    x1 = np.delete(x1,x2_zero,0)
    x2 = np.delete(x2,x2_zero,0)
    y = np.delete(y,x2_zero,0)
    data_mat.close()
    
    # preprocessing scRNA-seq read counts matrix
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
    if args.source == "Combined":
       input_size1 = input_size1 + input_size2
       x3 = np.concatenate([adata1.X,adata2.X], axis=-1)
       x3_raw = np.concatenate([adata1.raw.X,adata2.raw.X], axis=-1)
       adata3 = sc.AnnData(x3_raw)
       adata3.obs['Group'] = y
       adata3 = read_dataset(adata3,
                     transpose=False,
                     test_split=False,
                     copy=True)

       adata3 = normalize(adata3,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
       
       x3_sf = adata3.obs.size_factors
       print(x3.shape)
       print(x3_raw.shape)
       print(x3_sf.shape)
       
    print(args)
    print(y.shape)

    encodeLayer1 = list(map(int, args.encodeLayer1))
    encodeLayer2 = list(map(int, args.encodeLayer2))
    model = scMultiCluster(input_dim1=input_size1, input_dim2=input_size2,
                        zencode_dim=[64, 16], zdecode_dim=[16, 64], 
                        encodeLayer1=encodeLayer1, decodeLayer1=encodeLayer1[::-1], encodeLayer2=encodeLayer2, decodeLayer2=encodeLayer2[::-1],
                        sigma1=args.sigma1, sigma2=args.sigma2, gamma1=args.gamma1, gamma2=args.gamma2, gamma3=args.gamma3, cutoff = args.cutoff).cuda()
    
    print(str(model))

    t0 = time()
    if args.ae_weights is None:
       if args.source == "RNA":
          model.pretrain_autoencoder_rna(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
       elif args.source == "Protein":
          model.pretrain_autoencoder_protein(X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
       elif args.source == "Combined":
          model.pretrain_autoencoder_rna(X1=x3, X_raw1=x3_raw, sf1=x3_sf, batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)   
       else:
          exit("Wrong source")
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
    
    #get k
    if args.source == "RNA":
       latent = model.encodeBatch_rna(torch.tensor(adata1.X).cuda()).cpu().numpy()
    elif args.source == "Combined":
       latent = model.encodeBatch_rna(torch.tensor(x3).cuda()).cpu().numpy()
    elif args.source == "Protein":
       latent = model.encodeBatch_protein(torch.tensor(adata2.X).cuda()).cpu().numpy()
    else: 
       exit("Wrong source")
       
    if args.n_clusters == -1:
       n_clusters = GetCluster(latent, res=args.resolution, n=args.n_neighbors)
    else:
       print("n_cluster is defined as" + str(args.n_clusters))
       n_clusters = args.n_clusters  
    
    if args.source == "RNA":
       y_pred, _, _, _, _ = model.fit_RNA(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, y=y, n_clusters=n_clusters, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    elif args.source == "Combined":
       y_pred, _, _, _, _ = model.fit_RNA(X1=x3, X_raw1=x3_raw, sf1=x3_sf, y=y, n_clusters=n_clusters, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    elif args.source == "Protein":
       y_pred, _, _, _, _ = model.fit_protein(X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, y=y, n_clusters=n_clusters, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    else: 
       exit("Wrong source")
       
    print('Total time: %d seconds.' % int(time() - t0))
    
    if args.prediction_file != -1:
       np.savetxt(args.prediction_file + "_" + args.source + "pred.csv", y_pred, delimiter=',')
       
    if args.embedding_file != -1:  
       if args.source == "RNA":
          final_latent = model.encodeBatch_rna(torch.tensor(adata1.X).cuda()).cpu().numpy()
       elif args.source == "Combined":
          final_latent = model.encodeBatch_rna(torch.tensor(x3).cuda()).cpu().numpy()
       elif args.source == "Protein":
          final_latent = model.encodeBatch_protein(torch.tensor(adata2.X).cuda()).cpu().numpy()
       np.savetxt(args.embedding_file + "_" + args.source + "embedding.csv", final_latent, delimiter=",")
    
    
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('Final: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
