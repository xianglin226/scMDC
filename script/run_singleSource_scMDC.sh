#!/bin/bash -l
#SBATCH -J Single
#SBATCH -p datasci
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -w node412

module purge
conda activate torch_py38

f=../datasets/GSE100866.filtered_1000G.H5
echo "Final GSE100866 Combined"
for i in {1..10}
 do
   python -u ../src/run_singleSource_scMDC.py --n_clusters 6 --ae_weight_file ./AE_weights/AE_weights_inhouse.pth.tar --data_file $f --source RNA \
   #--embedding_file ./real_embedding/single/inhouse/Combined/$i --prediction_file ./real_embedding/single/inhouse/Combined/$i
done

# echo "Final inhouse RNA"
# for i in {1..10}
 # do
  # python -u run_scMultiCluster_gpu_singleSource_2to1.py --n_clusters 6 --gamma 0.001 --beta .1 --fi 0.001 --ae_weight_file AE_weights_inhouse.pth.tar --data_file $f \
   # --source RNA --embedding_file ./real_embedding/single/inhouse/RNA/$i --prediction_file ./real_embedding/single/inhouse/RNA/$i
# done

# echo "Final inhouse ADT"
# for i in {1..10}
 # do
   # python -u run_scMultiCluster_gpu_singleSource_2to1.py --n_clusters 6 --gamma 0.001 --beta .1 --fi 0.001 --ae_weight_file AE_weights_inhouse.pth.tar --data_file $f \
   # --source Protein --embedding_file ./real_embedding/single/inhouse/ADT/$i --prediction_file ./real_embedding/single/inhouse/ADT/$i
# done