import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset

# === CONFIGURATION ===
DIRECTIONS = ['forward', 'backward', 'left', 'right']

DIRECTION_SYNONYMS = {
    'forward':  ['forward', 'ahead'],
    'backward': ['backward', 'back'],
    'left':     ['left', 'to the left'],
    'right':    ['right', 'to the right']
}

START_VERBS = ['Walk', 'Move', 'Proceed', 'Advance', 'Head']
CONT_VERBS  = [
    'then go', 'then move', 'then proceed',
    'next, go', 'after that, go', 'subsequently, head'
]

# === DATA GENERATION ===
def gen_steps(n_steps: int,
              min_dist: int = 1,
              max_dist: int = 5) -> List[Tuple[str, int]]:
    """Return a list of n_steps (direction, distance_int) tuples."""
    return [
        (random.choice(DIRECTIONS), random.randint(min_dist, max_dist))
        for _ in range(n_steps)
    ]


def steps_to_prompt(steps: List[Tuple[str, int]]) -> str:
    """Turn [(dir,dist),…] into an English prompt with synonyms."""
    parts = []
    for i, (dir_key, dist) in enumerate(steps):
        verb = random.choice(START_VERBS) if i == 0 else random.choice(CONT_VERBS)
        dir_word = random.choice(DIRECTION_SYNONYMS[dir_key])
        meter = "meter" if dist == 1 else "meters"
        parts.append(f"{verb} {dir_word} for {dist} {meter}")
    return ", ".join(parts)


def generate_split(n_samples: int,
                   output_path: Path,
                   dataset_name: str = "",  # e.g. "motion_"
                   train_frac: float = 0.8,
                   val_frac: float = 0.1,
                   speed: float = 1.0) -> None:
    """
    Generate and save JSON splits (train/val/test) under output_path.
    Filenames: {dataset_name}train.json, {dataset_name}val.json, {dataset_name}test.json
    """
    # build all samples
    all_data: List[Dict[str, Any]] = []
    for _ in range(n_samples):
        k = 1#random.randint(1, 1)
        steps = gen_steps(k)
        prompt = steps_to_prompt(steps)
        total_dist = sum(dist for _, dist in steps)
        duration = int(round(total_dist / speed))
        all_data.append({
            'prompt': prompt,
            'duration': duration,
            'steps': steps
        })

    # shuffle and split
    random.shuffle(all_data)
    n_train = int(train_frac * n_samples)
    n_val   = int(val_frac * n_samples)
    splits = {
        'train': all_data[:n_train],
        'val':   all_data[n_train:n_train + n_val],
        'test':  all_data[n_train + n_val:]
    }

    # ensure directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # save each split
    for split, items in splits.items():
        fname = output_path / f"{dataset_name}{split}.json"
        with open(fname, 'w') as f:
            json.dump(items, f, indent=2)
        print(f"Wrote {len(items)} samples to {fname}")


# === PYTORCH DATASET ===
class MotionPromptDataset(Dataset):
    """
    PyTorch Dataset for prompt-duration-steps data.

    Expects JSON files named {dataset_name}{split}.json under data_dir.
    """
    def __init__(self,
                 data_dir: str,
                 split: str,
                 dataset_name: str = ""):
        """
        Args:
            data_dir: base folder containing JSON splits
            split: one of 'train','val','test' (or prefixed by dataset_name)
            dataset_name: optional prefix for filenames (e.g. 'motion_')
        """
        self.data_dir = Path(data_dir)
        self.split = split
        filename = f"{dataset_name}{split}.json"
        with open(self.data_dir / filename, 'r') as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        # could add tokenization or feature extraction here
        return sample

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts   = [item['prompt'] for item in batch]
        durations = torch.tensor([item['duration'] for item in batch], dtype=torch.int)
        steps     = [item['steps'] for item in batch]
        return {
            'prompts': prompts,
            'durations': durations,
            'steps': steps
        }


# === USAGE ===
if __name__ == '__main__':
    # Example: generate 10k samples and save to ./data
    generate_split(n_samples=1000,
                   output_path=Path('./motion_dataset'),
                   dataset_name='motion_')

    # Then in your config:
    # data:
    #   _target_: 'motion_prompt_dataset.MotionPromptDataset'
    #   data_dir: './data'
    #   dataset_name: 'motion_'
    # And load as:
    # train_dataset = instantiate(cfg.data, split=str(c.dataset_name)+'train')
