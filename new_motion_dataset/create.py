import os
import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

# === CONFIGURATION ===
SUBJECTS       = ['A person', 'A man']
VERBS          = ['is walking',  'is pacing', 'walks']
ADVERBS        = ['']
POSITION_RANGE = (-2.0, 2.0)
MIN_DURATION   = 6.0
MAX_DURATION   = 10.0
N_SAMPLES      = 10000

# precompute the max possible distance (corner to origin)
_max_dist = math.hypot(POSITION_RANGE[1], POSITION_RANGE[1])

# === DATA GENERATION ===
def generate_split(n_samples: int,
                   output_path: Path,
                   dataset_name: str = "",
                   train_frac: float = 0.8,
                   val_frac: float = 0.1) -> None:
    """
    Generate and save JSON splits (train/val/test) under output_path.
    Filenames: {dataset_name}train.json, {dataset_name}val.json, {dataset_name}test.json
    """
    all_data: List[Dict[str, Any]] = []
    for _ in range(n_samples):
        # prompt variation
        subj = random.choice(SUBJECTS)
        verb = random.choice(VERBS)
        adv  = random.choice(ADVERBS)
        prompt = f"{subj} {verb}{adv}."

        # random position
        x = round(random.uniform(*POSITION_RANGE), 2)
        y = round(random.uniform(*POSITION_RANGE), 2)

        # compute distance and scale to duration
        dist     = math.hypot(x, y)
        raw_dur  = (dist / _max_dist) * MAX_DURATION
        duration = round(raw_dur, 0)
        if duration < MIN_DURATION:
            duration = MIN_DURATION

        all_data.append({
            'prompt':  prompt,
            'position': [x, y],
            'duration': duration
        })

    random.shuffle(all_data)
    n_train = int(train_frac * n_samples)
    n_val   = int(val_frac * n_samples)
    splits = {
        'train': all_data[:n_train],
        'val':   all_data[n_train:n_train + n_val],
        'test':  all_data[n_train + n_val:]
    }

    output_path.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        fname = output_path / f"{dataset_name}{split}.json"
        with open(fname, 'w') as f:
            json.dump(items, f, indent=2)
        print(f"Wrote {len(items)} samples to {fname}")


# === PYTORCH DATASET ===
class PositionPromptDataset(Dataset):
    """
    PyTorch Dataset for prompt-position-duration data.

    Expects JSON files named {dataset_name}{split}.json under data_dir.
    """
    def __init__(self,
                 data_dir: str,
                 split: str,
                 dataset_name: str = ""):
        self.data_dir = Path(data_dir)
        self.split    = split
        fname = f"{dataset_name}{split}.json"
        with open(self.data_dir / fname, 'r') as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts   = [item['prompt'] for item in batch]
        durations = torch.tensor([item['duration'] for item in batch], dtype=torch.float)
        positions = torch.tensor([item['position'] for item in batch], dtype=torch.float)
        return {
            'prompts':   prompts,
            'durations': durations,
            'positions': positions
        }


# === USAGE ===
if __name__ == '__main__':
    generate_split(n_samples=N_SAMPLES,
                   output_path=Path('./data'),
                   dataset_name='pos_')
