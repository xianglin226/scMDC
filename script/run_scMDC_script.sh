#!/bin/bash -l
#SBATCH --gres=gpu:1

#Here are the commands for the real data experiments. We test ten times on each dataset.

f=../datasets/GSE128639_BMNC_annodata.h5
echo "Run CITE-seq BMNC"
python -u run_scMDC.py --n_clusters 27 --ae_weight_file AE_weights_bmnc.pth.tar --data_file $f --save_dir citeseq_bmnc /
--embedding_file --prediction_file --filter1

f=../datasets/10X_pbmc_granulocyte_plus.h5
echo "Run SMAGE-seq PBMC10K"
python -u run_scMDC.py --n_clusters 12 --ae_weight_file AE_weights_pbmc10k.pth.tar --data_file $f --save_dir atac_pbmc10k /
--embedding_file --prediction_file --filter1 --filter2 --f1 2000 --f2 2000 -el 256 128 64 -dl1 64 128 256 -dl2 64 128 256 -fi1 0.005 -fi2 0.005 -signma2 2.5 -tau .1

f=../datasets/GSE128639_BMNC_annodata.h5
echo "Run multi-batch CITE-seq SLN111"
python -u run_scMDC_batch.py --n_clusters 35 --ae_weight_file AE_weights_sln111.pth.tar --data_file $f --save_dir citeseq_sln111 /
--embedding_file --prediction_file --filter1
