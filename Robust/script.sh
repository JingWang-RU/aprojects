exp_out_dir=/exp
result_file=${exp_out_dir}/result.log
inc_learn=(0 0 0)
query_size=50
nc_model_type=wx
data_dir=~/data
customer=god
dr_qrel_file=${data_dir}/${customer}/${customer}_test_dr_qrels.txt

echo query_size: $query_size learn: ${ncd_learn[@]} >> $result_file
echo "***  evaluation" >> $result_file
runfile=$exp_out_dir/inc_${query_size}/og_nc_${nc_model_type}_nc_score_test.txt
python ../run_file.py $runfile
eval $dr_qrel_file $runfile -m ndcg_cut.10,20 -m recall -m P.1,5,10,20 -m map >> $result_file
