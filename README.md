
PyTorch implementation of paper [Issue Links Retrieval for New Issues in Issue Tracking Systems]([NLDB2024]) accepted by NLDB 2024.
# Issue Links Retrieval for New Issues in Issue Tracking Systems
Issue Tracking Systems (ITSs) are used to manage software issues, 
including  `feature requests` and `bug` reports. 
Issues are often interconnected by labelled links 
to show the relationships between issues. Recent studies 
in duplicate bug report retrieval (DBRR) aim to retrieve 
relevant duplicate bug reports o for a query bug report 
s  to form duplicate pairs (s, o). However, the DBRR 
task has its limitations: (1) only focuses on the issues 
with the `bug` type and overlooks other issue types, e.g., 
`task` and `improvement`;  (2) only issues with respect to the 
`duplicate` link type are retrieved, whereas a query s may have
other link types, e.g., `relates` and `depends on`, 
with candidate issues o. This paper goes beyond the 
DBRR task and proposes a new task of issue link retrieval
(ILR) for any issue types and link types. For a given query s,
the ILR task will locate all relevant tail issues o with respect 
to a given link type r to build issue links (s, r, o). 
This paper presents novel methods using pre-trained language models (e.g., GPT-2) 
to embed issue links for computing probability scores to retrieve issue links.
Our methods are evaluated on four datasets, demonstrating high effectiveness 
in issue link retrieval.

### Install python virtual environment

```bash
# install virtual environment module
python3 -m pip install --user virtualenv

# create virtual environment
python3 -m venv env_name
source env_name/bin/activate
# install python packages
pip install requests
pip install -r requirements.txt
```

### Run the code

```bash 
bash gpu_run_relation_tail_prediction.sh 
```
```
or run
python  relation_tail_prediciton.py    --task_name kg    --do_relation_tail_prediction   --data_dir ./data/RedHat  --model_type  BERT   --model_name_or_path  bert-base-cased   --max_seq_length  400   --per_gpu_train_batch_size   16    --learning_rate 5e-5   --gradient_accumulation_steps 2  --eval_batch_size  512   --pre_process_data  ./pre_process_data   --negative   2   --num_train_epochs   2   --output_dir  ./output_RedHat_e2_2_bert_base_max_seq_400/
```

`--do_relation_tail_prediction`: set this flag to run the relation tail prediction task.
  
 `--data_dir`: the path of dataset,
 
 `--model_type`: name of the model name

  `--model_name_or_path`: the path/type of the pre-trained model.
  
  `--max_seq_length`: maximum sequence length of input text.
  
  `--per_gpu_train_batch_size`: batch size for training.
  
  `--learning_rate`: learning rate for training.
  
  `--gradient_accumulation_steps`: number of gradient accumulation steps.
  
  `--eval_batch_size`: batch size for evaluation.
  
  `--pre_process_data`: the path of pre-processed data.
  
  `--negative`: number of negative samples per training instance.
  
  `--num_train_epochs`: the number of training epoches.
  
  `--output_dir`: the path of output directory.
   

### Data
Download data from [here](https://qubstudentcloud-my.sharepoint.com/:f:/g/personal/40305887_ads_qub_ac_uk/EoN0DtgpNXdJnvPPpwY6P3UBYd2dh-ViXkwTigkpelsWxg?e=EOC3Fj)
### Contact:
This paper has been accepted by the 29th International Conference on Natural Language & Information Systems (NLDB 2024). The published version can be viewed by this link [](will give). If you use any code from our repo in your paper, pls cite:
```buildoutcfg
@inproceedings{hu2024issue,
  title={Issue Links Retrieval for New Issues in Issue Tracking Systems},
  author={Miao Hu, Zhiwei Lin, Adele Marshall}
    booktitle={The 29th International Conference on Natural Language & Information Systems (NLDB 2024)},
    year={2024}
 }
```

Feel free to contact MiaoHu ([mhu05@qub.ac.uk](mhu05@qub.ac.uk)),  if you have any further questions.
