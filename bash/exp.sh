#!/bin/bash

python mainRL.py group_name=NormalizationLayerFreeze-short experiment_name=real_TMR2 sequence_fixed=false dataset_name='short_' tmr_plus_plus=true val_num_batch=0 val_batch_size=256 train_epochs=4 train_batch_size=20 lora=true num_prompts_dataset=65 num_gen_per_prompt=4 iterations=10000