#!/bin/bash

python mainRL.py group_name=NormalizationLayerFreeze-short experiment_name=real_TMR_test_lora_swag sequence_fixed=false dataset_name='short_' val_num_batch=0 val_batch_size=256 train_epochs=4 train_batch_size=48 lora=true num_prompts_dataset=72 num_gen_per_prompt=4 iterations=10000