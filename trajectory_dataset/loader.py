import os
import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from TMR.mtt.load_tmr_model import load_tmr_model_easy

# load your TMR text‐encoder
tmr_forward = load_tmr_model_easy(device="cpu",
                                  dataset="tmr_humanml3d_kitml_guoh3dfeats")


class TrajectoryPromptDataset(Dataset):
    """
    PyTorch Dataset for prompt-trajectory-duration data.

    Expects JSON files named {dataset_name}{split}.json under data_dir.
    """
    def __init__(self,
                 data_dir: str,
                 split: str,
                 dataset_name: str = ""):
        self.data_dir = Path(data_dir)
        fname = f"{dataset_name}{split}.json"
        with open(self.data_dir / fname, 'r') as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item       = self.samples[idx]
        prompt     = item['prompt']
        trajectory = torch.tensor(item['trajectory'], dtype=torch.float)
        duration   = torch.tensor(item['duration'], dtype=torch.float)

        # optional text embedding
        emb = tmr_forward([prompt])[0]

        return {
            'text':       prompt,
            'trajectory': trajectory,  # shape (CHECKPOINTS, 2)
            'duration':   duration,
            'tmr_text':   emb
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts        = [b['text'] for b in batch]
        durations    = torch.stack([b['duration'] for b in batch])
        trajectories = torch.stack([b['trajectory'] for b in batch])
        tmr_texts    = torch.stack([b['tmr_text'] for b in batch])

        return {
            'text':         texts,
            'durations':    durations,
            'trajectories': trajectories,
            'tmr_text':     tmr_texts
        }