# Example Real datasets
Example datasets are in h5 format and can be downloaded from https://drive.google.com/drive/folders/1HTCrb3HZpbN8Nte5xGhY517NMdyzqftw?usp=sharing  
  
Required objects in h5 file for running scMDC
1) X1: mRNA count matrix
2) X2: ADT count matrix
3) Y: True labels (if exist)
4) Batch: batch indicator (for multi-batch analysis)

Other objects in the h5 files:
1) ADT: feature names in ADT count matirx (only in CITE-seq data)
2) GenesFromPeaks: feature names in the gene-to-cell matrix mapped from scATAC-seq (only in SMAGE-seq data)
3) Genes: feature names in mRNA count matrix
4) Cell types: cell type of each cell (if exist).
5) Barcodes: cell barcodes (if exits)  

Note: for using filtered datasets (Normalized_filtred*), use X1_ and X2_ as inputs.  
