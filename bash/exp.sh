#!/bin/bash

HYDRA_FULL_ERROR=1 python mainRL.py group_name=NormalizationLayerFreeze-SHORT experiment_name=lora_special_no_autocast dataset_name="short_" train_batch_size=20 lora=true num_prompts_dataset=64 num_gen_per_prompt=4  iterations=10000