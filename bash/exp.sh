#!/bin/bash

python mainRL.py group_name=NormalizationLayerFreeze-SHORT experiment_name=lora_special_no_autocastspyro dataset_name="short_" train_epochs=3 train_batch_size=20 lora=true num_prompts_dataset=64 num_gen_per_prompt=4  iterations=10000