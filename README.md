# scMDC
Single Cell Multi-omics deep clustering (**scMDC v1.0.1**)

We develop a novel multimodal deep learning method, scMDC, for single-cell multi-omics data clustering analysis. scMDC is an end-to-end deep model that explicitly characterizes different data sources and jointly learns latent features of deep embedding for clustering analysis. Extensive simulation and real-data experiments reveal that scMDC outperforms existing single-cell single-modal and multimodal clustering methods on different single-cell multimodal datasets. The linear scalability of running time makes scMDC a promising method for analyzing large multimodal datasets.

## Table of contents
- [Network diagram](#diagram)
- [Dependencies](#Dependencies)
- [Usage](#Usage)
- [Output](#Output)
- [Arguments](#Arguments)
- [Citation](#Citation)
- [Contact](#contact)

## <a name="diagram"></a>Network diagram
![Model structure](https://github.com/xianglin226/scMDC/blob/master/src/fig1_.png?raw=true)  

## <a name="Dependencies"></a>Dependencies
Python 3.8.1

Pytorch 1.6.0

Scanpy 1.6.0

SKlearn 0.22.1

Numpy 1.18.1

h5py 2.9.0  

All experiments of scMDC in this study are conducted on Nvidia Tesla P100 (16G) GPU.
We suggest to install the dependencies in a conda environment (conda create -n scMDC).  
It takes few minutes to install the dependencies.  
scMDC takes about 3 minutes to cluster a dataset with 5000 cells.  

## <a name="Usage"></a>Usage  
1) Prepare the input data in h5 format. (See readme in 'dataset' folder)  
2) Run scMDC according to the running script in "script" folder (Note the parameter settings if you work on mRNA+ATAC data and use run_scMDC_batch.py for multi-batch data clustering)  
3) Run DE analysis by run_LRP.py based on the well-trained scMDC model (refer the LRP running script in "script" folder)  

## <a name="Output"></a>Output  
1) scMDC outputs a latent representation of data which can be used for further downstream analyses and visualized by t-SNE or Umap; 
2) Multi-batch scMDC outputs a latent representation of integrated datasets on which the batch effects are corrected.  
3) LRP outputs a gene rank which indicates the importances of genes for a given cluster and can be used for pathway analysis.  

## <a name="Arguments"></a>Arguments
--n_clusters: number of clusters (K); scMDC will estimate K if this arguments is set to -1.  
--cutoff: A ratio of epoch before which the model only train the low-level autoencoders.   
--batch_size: batch size.  
--data_file: path to the data input.  
Data format: H5.  
Structure: X1(RNA), X2(ADT or ATAC), Y(label, if exit), Batch (Batch indicator for multi-batch data clustering).  
--maxiter: maximum epochs of training. Default: 10000.  
--pretrain_epochs: number of epochs for pre-training. Default: 400.  
--gamma: coefficient of clustering loss. Default: 0.1.  
--phi1 and phi2: coefficient of KL loss in pretraining and clustering stage. Default: 0.001 for CITE-Seq; 0.005 for SMAGE-Seq*.  
--update_interval: the interval to check the performance. Default: 1.  
--tol: the criterion to stop the model, which is a percentage of changed labels. Default: 0.001.  
--ae_weights: path of the weight file.  
--save_dir: the directory to store the outputs.  
--ae_weight_file: the directory to store the weights.  
--resolution: the resolution parameter to estimate k. Default: 0.2.  
--n_neighbors: the n_neighbors parameter to estimate K. Default: 30.  
--embedding_file: if save embedding file. Default: No  
--prediction_file: if save prediction file. Default: No  
--encodeLayer: layers of the low-level encoder for RNA: Default: [256,64,32,16] for CITE-Seq; [256,128,64] for SMAGE-seq.  
--decodeLayer1: layers of the low-level encoder for ADT: Default: [16,64,256] for CITE-Seq. [64,128,256] for SMAGE-seq.  
--decodeLayer2: layers of the high-level encoder. Default:[16,20] for CITE-Seq. [64,128,256] for SMAGE-seq.  
--sigma1: noise on RNA data. Default: 2.5.  
--sigma2: noise on ADT data. Default: 1.5 for CITE-Seq; 2.5 for SMAGE-Seq  
--filter1: if do feature selection on Genes. Default: No.  
--filter2: if do feature selection on ATAC. Default: No.  
--f1: Number of high variable genes (in X1) used for clustering if doing the featue selection. Default: 2000  
--f2: Number of high variable genes from ATAC (in X2) used for clustering if doing the featue selection. Default: 2000  
*We denote 10X Single-Cell Multiome ATAC + Gene Expression technology as SMAGE-seq for convenience.  


## <a name="Citation"></a>Citation
Lin, X., Tian, T., Wei, Z., & Hakonarson, H. (2022). Clustering of single-cell multi-omics data with a multimodal deep learning method. Nature Communications, 13(1), 1-18.

## <a name="contact"></a>Contact
Xiang Lin <xl456@njit.edu>
