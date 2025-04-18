#!/bin/bash

python mainRL.py group_name=NormalizationLayerFreeze experiment_name=vanilla3 train_batch_size=48 guidance_weight_train=1 guidance_weight_generation=1 guidance_weight_valid=1 num_prompts_dataset=64 num_gen_per_prompt=4  iterations=10000
