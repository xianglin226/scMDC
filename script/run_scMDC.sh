#!/bin/bash -l
#SBATCH -J scMDC
#SBATCH -p datasci
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -w node413

module purge
conda activate torch_py38

f=../datasets/10XPBMC_filtered_1000G.H5
echo "Final 10XPBMC"
for i in {1..10}
 do
   python -u ../src/run_scMDC.py --n_clusters 7 --ae_weight_file ./AE_weights/AE_weights_GSE128639.pth.tar --data_file $f \
   #--embedding_file ./real_embedding/MAE/GSE128639/$i --prediction_file ./real_embedding/MAE/GSE128639/$i
done

f=../datasets/GSE100866.filtered_1000G.H5
echo "Final GSE100866"
for i in {1..10}
 do
   python -u ../src/run_scMDC.py --n_clusters 6 --ae_weight_file ./AE_weights/AE_weights_GSE128639.pth.tar --data_file $f \
   #--embedding_file ./real_embedding/MAE/GSE128639/$i --prediction_file ./real_embedding/MAE/GSE128639/$i
done
#163633 test --gamma 0.01 --beta .1 --fi 0.01
#163637 test --gamma 0.01 --beta .1 --fi 0.1
