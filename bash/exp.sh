#!/bin/bash

python mainRL.py group_name=NormalizationLayerFreeze-short experiment_name=lora_special_no_autocast_ekko_3_fix_ratio0.75 dataset_name="short_" sequence_fixed=true train_epochs=3 train_batch_size=40 lora=true num_prompts_dataset=64 num_gen_per_prompt=4  iterations=10000
