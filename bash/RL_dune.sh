#!/bin/bash

python mainRL.py group_name="TESI" experiment_name="TMR++_Train_2-10" train_split="train" dataset_name='' reward="TMR++" train_batch_size=16 iterations=2500 path_model="RL_Model_2-10_Train" path_res="ResultRL_2-10_Train" device_swag="cuda:0"
python mainRL.py group_name="TESI" experiment_name="TMR++_Val_2-10" train_split="val" dataset_name='' reward="TMR++" train_batch_size=16 iterations=2500 path_model="RL_Model_2-10_Val" path_res="ResultRL_2-10_Val" device_swag="cuda:0"