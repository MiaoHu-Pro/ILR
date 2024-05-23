#!/bin/bash
#SBATCH --job-name=run_bert
#SBATCH -p gpu
#SBATCH --time=72:00:00
#SBATCH --gres gpu:1
#SBATCH --output=./result_out/hadoop_77K_e3_2negative_no_relation_des_head_or_tail_are_replaced_with_isolated_max_seq_400_no_relation_des-out-%j.out
#SBATCH --partition=k2-gpu
#SBATCH --exclude=gpu[114]

#default mem is 7900M
#SBATCH --mem 80000M

module add nvidia-cuda
module add apps/python3

nvidia-smi

export PYTHONPATH=$PYTHONPATH:/users/40305887/gridware/share/python/3.6.4/lib/python3.6/site-packages




echo "the job start "



python  relation_tail_prediciton.py    --task_name kg    --do_relation_tail_prediction   --data_dir ./data/RedHat  --model_type  BERT   --model_name_or_path  bert-base-cased   --max_seq_length  400   --per_gpu_train_batch_size   16    --learning_rate 5e-5   --gradient_accumulation_steps 2  --eval_batch_size  512   --pre_process_data  ./pre_process_data   --negative   2   --num_train_epochs   2   --output_dir  ./output_RedHat_e2_2_bert_base_max_seq_400/

echo "the job end "





