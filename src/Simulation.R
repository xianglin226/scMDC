library(SymSim)
library(rhdf5)
library(Seurat)

phyla <- read.tree("tree.txt")
phyla2 <- read.tree("tree.txt")
data(gene_len_pool)

#This is a simulation script with adding batch effect
#First we need to modify the DivideBatches2 function in SymSim to make the same batch partition in mRNA and ADT data.
DivideBatches2 <- function(observed_counts_res, batchIDs, batch_effect_size=1){
  observed_counts <- observed_counts_res[["counts"]]
  meta_cell <- observed_counts_res[["cell_meta"]]
  ncells <- dim(observed_counts)[2]
  ngenes <- dim(observed_counts)[1]
  nbatch <- unique(batchIDs)
  meta_cell2 <- data.frame(batch = batchIDs, stringsAsFactors = F)
  meta_cell <- cbind(meta_cell, meta_cell2)
  mean_matrix <- matrix(0, ngenes, nbatch)
  gene_mean <- rnorm(ngenes, 0, 1)
  temp <- lapply(1:ngenes, function(igene) {
    return(runif(nbatch, min = gene_mean[igene] - batch_effect_size, 
                 max = gene_mean[igene] + batch_effect_size))
  })
  mean_matrix <- do.call(rbind, temp)
  batch_factor <- matrix(0, ngenes, ncells)
  for (igene in 1:ngenes) {
    for (icell in 1:ncells) {
      batch_factor[igene, icell] <- rnorm(n = 1, mean = mean_matrix[igene, 
                                                                    batchIDs[icell]], sd = 0.01)
    }
  }
  observed_counts <- round(2^(log2(observed_counts) + batch_factor))
  return(list(counts = observed_counts, cell_meta = meta_cell))
}

for(k in 1:10){
  ##RNA
  ncells = 1000
  nbatchs = 2
  batchIDs <- sample(1:nbatchs, ncells, replace = TRUE)
  print(k)
  print("Simulate RNA")
  ngenes <- 2000
  gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
  true_RNAcounts_res <- SimulateTrueCounts(ncells_total=ncells, 
                                           min_popsize=50, 
                                           i_minpop=1, 
                                           ngenes=ngenes, 
                                           nevf=10, 
                                           evf_type="discrete", 
                                           n_de_evf=6, 
                                           vary="s", 
                                           Sigma=0.6, 
                                           phyla=phyla,
                                           randseed=k+1000)

  observed_RNAcounts <- True2ObservedCounts(true_counts=true_RNAcounts_res[[1]], 
                                            meta_cell=true_RNAcounts_res[[3]], 
                                            protocol="UMI", 
                                            alpha_mean=0.00075, 
                                            alpha_sd=0.0001, 
                                            gene_len=gene_len, 
                                            depth_mean=50000, 
                                            depth_sd=3000,
  )

  batch_RNAcounts <- DivideBatches2(observed_RNAcounts, batchIDs, batch_effect_size = 1)
  
  ## Add batch effects 
  print((sum(batch_RNAcounts$counts==0)-sum(true_RNAcounts_res$counts==0))/sum(true_RNAcounts_res$counts>0))
  print(sum(batch_RNAcounts$counts==0)/prod(dim(batch_RNAcounts$counts)))
  
  ##ADT
  print("Simulate ADT")
  nadts <- 100
  gene_len <- sample(gene_len_pool, nadts, replace = FALSE)
  #The true counts of the five populations can be simulated:
  true_ADTcounts_res <- SimulateTrueCounts(ncells_total=ncells, 
                                           min_popsize=50, 
                                           i_minpop=1, 
                                           ngenes=nadts, 
                                           nevf=10, 
                                           evf_type="discrete", 
                                           n_de_evf=6, 
                                           vary="s", 
                                           Sigma=0.3, 
                                           phyla=phyla2,
                                           randseed=k+1000)
  
  observed_ADTcounts <- True2ObservedCounts(true_counts=true_ADTcounts_res[[1]], 
                                            meta_cell=true_ADTcounts_res[[3]], 
                                            protocol="UMI", 
                                            alpha_mean=0.045, 
                                            alpha_sd=0.01, 
                                            gene_len=gene_len, 
                                            depth_mean=50000, 
                                            depth_sd=3000,
  )
  
  ## Add batch effects 
  batch_ADTcounts <- DivideBatches2(observed_ADTcounts, batchIDs, batch_effect_size = 1)
  
  print((sum(batch_ADTcounts$counts==0)-sum(true_ADTcounts_res$counts==0))/sum(true_ADTcounts_res$counts>0))
  print(sum(batch_ADTcounts$counts==0)/prod(dim(batch_ADTcounts$counts)))
  
  y1 = batch_RNAcounts$cell_meta$pop
  y2 = batch_ADTcounts$cell_meta$pop
  batch1 = batch_ADTcounts$cell_meta$batch
  batch2 = batch_RNAcounts$cell_meta$batch
  print(sum(y1==y2))
  print(sum(batch1 == batch2))
  
  counts1 <- batch_RNAcounts[[1]]
  counts2 <- batch_ADTcounts[[1]]
  
  #filter
  rownames(counts2) <- paste("G",1:nrow(counts2),sep = "")
  colnames(counts2) <- paste("C",1:ncol(counts2),sep = "")
  
  pbmc <- CreateSeuratObject(counts = counts2, project = "P2", min.cells = 0, min.features = 0)
  pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize")
  pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 30)
  counts2 <- counts2[pbmc@assays[["RNA"]]@var.features,]
  
  h5file = paste("./batch/Simulation.", k, ".h5", sep="")
  h5createFile(h5file)
  h5write(as.matrix(counts1), h5file,"X1")
  h5write(as.matrix(counts2), h5file,"X2")
  h5write(y1, h5file,"Y")
  h5write(batch1, h5file,"Batch")
}
