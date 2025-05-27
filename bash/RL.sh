#!/bin/bash

# python mainRL.py run_dir=pretrained_models/mdm-smpl_clip_smplrifke_kitml group_name=TESI experiment_name=KIT_TMR++_Train_2-5 train_batch_size=16 train_split="train" dataset_name='' reward="TMR++" iterations=2500 path_model="RL_Model_2-5_Train_KIT" path_res="ResultRL_2-5_Train_KIT"
# python mainRL.py run_dir=pretrained_models/mdm-smpl_clip_smplrifke_kitml group_name=TESI experiment_name=KIT_TMR++_Val_2-5 train_batch_size=16 train_split="val" dataset_name='' reward="TMR++" iterations=2500 path_model="RL_Model_2-5_Val_KIT" path_res="ResultRL_2-5_Val_KIT"

python mainRL.py group_name="TESI" experiment_name="TMR++_Train_2-5" train_batch_size=16 train_split="train" dataset_name='short_' reward="TMR++" iterations=2500 path_model="RL_Model_2-5_Train" path_res="ResultRL_2-5_Train"
python mainRL.py group_name="TESI" experiment_name="TMR++_Val_2-5" train_batch_size=16 train_split="val" dataset_name='short_' reward="TMR++" iterations=2500 path_model="RL_Model_2-5_Val" path_res="ResultRL_2-5_Val"