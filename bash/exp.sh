#!/bin/bash

python mainRL.py group_name=NormalizationLayerFreeze-walk experiment_name=lora_vanilla_no_autocast dataset_name="walk_" train_batch_size=40 lora=true num_prompts_dataset=64 num_gen_per_prompt=4  iterations=10000
