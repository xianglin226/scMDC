# scMDC
Single Cell Multi-omics deep clustering
# Dependencies
Python 3.8.1

Pytorch 1.6.0

Scanpy 1.6.0

SKlearn 0.22.1

Numpy 1.18.1

# Run scMDC
See the example script in script/run_scMDC.sh

# Arguments
--n_clusters: number of clusters (K); scMDC will estimate K if this arguments is set to -1.  
--cutoff1: A ratio of epoch before which the model only train the low-level autoencoders.
--cutoff2: A ratio of batch before which the model only train the low-level autoencoders.  
--batch_size: batch size.  
--data_file: path to the data input.  
Data format: H5.  
Structure: X1(RNA), X2(ADT), Y(label, if exit).  
--maxiter: maximum epochs of training. Default: 2000.  
--pretrain_epochs: number of epochs for pre-training. Default: 600.  
--gamma1: coefficient of clustering loss. Default: 0.01.  
--gamma2: coefficient of latent autoencoder loss. Default: 0.1. 
--gamma3: coefficient of KL loss. Default: 0.001. 
--update_interval: the interval to check the performance. Default: 100.  
--tol: the criterion to stop the model, which is a percentage of changed labels. Default: 0.0001.  
--ae_weights: path of the weight file.  
--save_dir: the directory to store the outputs.  
--ae_weight_file: the directory to store the weights.  
--resolution: the resolution parameter to estimate k. Default: 0.2.  
--n_neighbors: the n_neighbors parameter to estimate K. Default: 30.  
--embedding_file: the directory to store embedding output. Default: -1, means no embedding output.  
--prediction_file: the directory to store prediction output. Default: -1, means no prediction output.  
--encodeLayer1: layers of the low-level encoder for RNA: Default: [256,128,64].  
--encodeLayer2: layers of the low-level encoder for ADT: Default: [8].  
--encodeLayer3: layers of the high-level encoder. Default:[64,16].  
--sigma1: noise on RNA data. Default: 3..  
--sigma2: noise on ADT data. Default: 0.2.  
