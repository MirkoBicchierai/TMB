#!/bin/bash

# HUMAN3DML 2-10 sec (masked)
python mainRL.py group_name="TESI" experiment_name="TMR++_Train_2-10" train_split="train" dataset_name='' reward="TMR++" train_batch_size=16 iterations=2500 path_model="RL_Model_2-10_Train" path_res="ResultRL_2-10_Train" device_swag="cuda:0" num_grad_accumulation_steps=2
python mainRL.py group_name="TESI" experiment_name="TMR++_Val_2-10" train_split="val" dataset_name='' reward="TMR++" train_batch_size=16 iterations=2500 path_model="RL_Model_2-10_Val" path_res="ResultRL_2-10_Val" device_swag="cuda:0" num_grad_accumulation_steps=2

# KITML 2-10 sec (masked), # fix config.json of model first
python mainRL.py run_dir="pretrained_models/mdm-smpl_clip_smplrifke_kitml" group_name="TESI" experiment_name="KIT_TMR++_Train_2-10" train_split="train" dataset_name='' reward="TMR++" train_batch_size=16 iterations=2500 path_model="RL_Model_2-10_Train_KIT" path_res="ResultRL_2-10_Train_KIT" device_swag="cuda:0" num_grad_accumulation_steps=2
python mainRL.py run_dir="pretrained_models/mdm-smpl_clip_smplrifke_kitml" group_name="TESI" experiment_name="KIT_TMR++_Val_2-10" train_split="val" dataset_name='' reward="TMR++" train_batch_size=16 iterations=2500 path_model="RL_Model_2-10_Val_KIT" path_res="ResultRL_2-10_Val_KIT" device_swag="cuda:0" num_grad_accumulation_steps=2