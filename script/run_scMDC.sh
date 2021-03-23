#!/bin/bash -l
#SBATCH --gres=gpu:1

f=../datasets/10XPBMC_filtered_1000G.H5
echo "10XPBMC"
for i in {1..10}
 do
   python -u ../src/run_scMDC.py --n_clusters 7 --ae_weight_file ./AE_weights/AE_weights_GSE128639.pth.tar --data_file $f \
   #--embedding_file ./real_embedding/MAE/GSE128639/$i --prediction_file ./real_embedding/MAE/GSE128639/$i
done

f=../datasets/GSE100866.filtered_1000G.H5
echo "GSE100866"
for i in {1..10}
 do
   python -u ../src/run_scMDC.py --n_clusters 6 --ae_weight_file ./AE_weights/AE_weights_GSE128639.pth.tar --data_file $f \
   #--embedding_file ./real_embedding/MAE/GSE128639/$i --prediction_file ./real_embedding/MAE/GSE128639/$i
done
