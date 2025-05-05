#!/bin/bash

#python mainRL2.py group_name=Dataset-controllability-simple experiment_name=sequence-masked sequence_fixed=false  val_num_batch=0 val_batch_size=256 train_epochs=4 train_batch_size=20 lora=true num_prompts_dataset=65 num_gen_per_prompt=4 iterations=10000

python mainRL3.py  group_name=New_reward_magic2 experiment_name=better_plus_TMR train_epochs=4 train_batch_size=20 lora=true num_prompts_dataset=64 num_gen_per_prompt=4 iterations=10000 sequence_fixed=false val_num_batch=0 val_batch_size=256