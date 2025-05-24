#!/bin/bash

python mainRL.py group_name=TESI experiment_name=TMR++_Train_2-10_guidance_1 sequence_fixed=false dataset_name='' reward="TMR++" val_num_batch=0 val_batch_size=256 train_epochs=4 train_batch_size=8 lora=true num_prompts_dataset=64 num_gen_per_prompt=4 iterations=2500 guidance_weight=1.0