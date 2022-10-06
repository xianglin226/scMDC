# Run DE analysis (LRP) based on the well-trained scMDC model and its results.
f=../datasets/CITESeq_GSE128639_BMNC_annodata.h5
python -u run_scMDC.py --n_clusters 27 --ae_weight_file ./out_bmnc_full/AE_weights_bmnc.pth.tar --data_file $f --prediction_file bmnc --save_dir out_bmnc_full --filter1
python -u run_LRP.py --n_clusters 27 --ae_weights ./out_bmnc_full/AE_weights_bmnc.pth.tar --cluster_index_file ./out_bmnc_full/1_pred.csv --data_file $f --save_dir out_bmnc_full --filter1 
