#!/bin/bash

python mainRL.py group_name=Dataset-Human3DML-Val experiment_name=TMR_guidance_1 sequence_fixed=false dataset_name='' tmr_plus_plus=true val_num_batch=4 val_batch_size=256 train_epochs=4 train_batch_size=8 lora=true num_prompts_dataset=64 num_gen_per_prompt=4 iterations=10000 guidance_weight=1.0