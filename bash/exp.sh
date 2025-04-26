#!/bin/bash

python mainRL.py group_name=NormalizationLayerFreeze-walk2 experiment_name=lora_vanilla_no_autocast_spyro_maschere_3_fix dataset_name="walk2_" sequence_fixed=false train_epochs=3 train_batch_size=20 lora=true num_prompts_dataset=64 num_gen_per_prompt=4  iterations=10000
