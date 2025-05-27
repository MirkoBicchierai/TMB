#!/bin/bash

# HUMAN3DML 2-5 sec (fixed 2.5)
python mainRL.py group_name="TESI" experiment_name="TMR++_Train_Fixed_2-5" sequence_fixed=true time=2.5 train_batch_size=32 train_split="train" dataset_name='short_' reward="TMR++" iterations=2500 path_model="RL_Model_Fixed_2-5_Train" path_res="ResultRL_Fixed_2-5_Train" num_grad_accumulation_steps=1
python mainRL.py group_name="TESI" experiment_name="TMR++_Val_Fixed_2-5" sequence_fixed=true time=2.5 train_batch_size=32 train_split="val" dataset_name='short_' reward="TMR++" iterations=2500 path_model="RL_Model_Fixed_2-5_Val" path_res="ResultRL_Fixed_2-5_Val" num_grad_accumulation_steps=1

# HUMAN3DML 2-5 sec (masked)
python mainRL.py group_name="TESI" experiment_name="TMR++_Train_2-5" train_batch_size=16 train_split="train" dataset_name='short_' reward="TMR++" iterations=2500 path_model="RL_Model_2-5_Train" path_res="ResultRL_2-5_Train" num_grad_accumulation_steps=2
python mainRL.py group_name="TESI" experiment_name="TMR++_Val_2-5" train_batch_size=16 train_split="val" dataset_name='short_' reward="TMR++" iterations=2500 path_model="RL_Model_2-5_Val" path_res="ResultRL_2-5_Val" num_grad_accumulation_steps=2

# HUMAN3DML 2-10 sec (fixed 5)
python mainRL.py group_name="TESI" experiment_name="TMR++_Train_Fixed_2-10" sequence_fixed=true time=5 train_batch_size=16 train_split="train" dataset_name='' reward="TMR++" iterations=2500 path_model="RL_Model_Fixed_2-10_Train" path_res="ResultRL_Fixed_2-10_Train" num_grad_accumulation_steps=2
python mainRL.py group_name="TESI" experiment_name="TMR++_Val_Fixed_2-10" sequence_fixed=true time=5 train_batch_size=16 train_split="val" dataset_name='' reward="TMR++" iterations=2500 path_model="RL_Model_Fixed_2-10_Val" path_res="ResultRL_Fixed_2-10_Val" num_grad_accumulation_steps=2

# KITML 2-10 sec (fixed 5)
python mainRL.py run_dir="pretrained_models/mdm-smpl_clip_smplrifke_kitml" group_name="TESI" experiment_name="KIT_TMR++_Train_Fixed_2-10" sequence_fixed=true time=5 train_batch_size=16 train_split="train" dataset_name='' reward="TMR++" iterations=2500 path_model="RL_Model_Fixed_2-10_Train_KIT" path_res="ResultRL_Fixed_2-10_Train_KIT" num_grad_accumulation_steps=2
python mainRL.py run_dir="pretrained_models/mdm-smpl_clip_smplrifke_kitml" group_name="TESI" experiment_name="KIT_TMR++_Val_Fixed_2-10" sequence_fixed=true time=5 train_batch_size=16 train_split="val" dataset_name='' reward="TMR++" iterations=2500 path_model="RL_Model_Fixed_2-10_Val_KIT" path_res="ResultRL_Fixed_2-10_Val_KIT" num_grad_accumulation_steps=2
