library(rhdf5)
library(splatter)

#Randomly assign cell numbers in each group but with low diiferences
rand_vect_cont <- function(N, M, sd = 1) {
  vec <- abs(rnorm(N, M/N, sd))
  vec / sum(vec) * M
}

get_group <- function(N, M, sd=1){
  pro = rand_vect_cont(N,1,sd)
  sample(seq(1,N,1),M, prob = pro, replace = T)
}

for (i in 1:20) {
  #4 groups, 500 cells 
  groups <- get_group(4,500, sd=0.1)
  
  params = newLun2Params()
  params <- setParam(params, "seed", i)
  #simulate 1000 genes
  params <- setParam(params, "nGenes", 1000)
  #set dropout rate to 0.5, 0.7 and 0.9, mean and disp of gene and ZINB were adjusted according to the real datasets.
  #Different datasets have different mean and sd of genes, but sd is always about 4-fold of mean.
  #For exmaple, mean and sd of genes for PBMC data is about 1 and 5; for a randomly-picked simulated data (with medium signal and 0.9 dropout), mean and sd is about 1.5 and 6
  params <- setParam(params, "zi.params", data.frame(Mean=rep(3,1000), Disp=rep(0.1,1000), Prop=rep(0.9,1000))) #p 0.5, 0.7, 0.9
  params <- setParam(params, "cell.plates", groups)
  #enlarge the library size differences among cells 
  params <- setParam(params, "cell.libMod",5)
  #change the signal among groups to low = 0.25, mid = 0.5, and high = 1
  params <- setParam(params, "plate.var",0.5) 
  
  #simulate RNA by ZINB model
  sim1 <- lun2Simulate(params = params, zinb = T, verbose = T)
  dat1 <- counts(sim1)
  
  #simulate 20 ADTS
  params <- setParam(params, "nGenes", 20)
  #ADT mean and disp were adjusted according to the real datasets. 
  #Different datasets have quite different mean and sd of ADT, but sd is always about 1.5 to 2-fold of mean.
  #For example, PBMC dataset has mean and sd of ADT about 300 and 600; a random-picked simulated data has mean and sd about 220 and 400.
  params <- setParam(params, "gene.params", data.frame(Mean=rep(50,20), Disp=rep(1,20)))
  
  #simulate ADT by NB model
  sim2 <- lun2Simulate(params = params, zinb = F, verbose = T)
  dat2 <- counts(sim2)
  
  res <- paste("Simulation",i,"H5",sep = ".")
  h5createFile(res)
  h5write(dat1, res, "X1")
  h5write(dat2, res, "X2")
  h5write(groups, res, "Y")
  h5closeAll()
}
