#!/bin/bash

python mainRL_split_reward.py  group_name=New_reward_magic2 experiment_name=split-TEST train_epochs=4 train_batch_size=20 lora=true num_prompts_dataset=64 num_gen_per_prompt=4 iterations=10000 sequence_fixed=false val_num_batch=0 val_batch_size=32