import os
import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

# === CONFIGURATION ===
SUBJECTS = ['A person', 'A man', ]
VERBS = ['is walking', 'walks']
ADVERBS = ['', ]

MIN_DURATION = 2
MAX_DURATION = 10

N_SAMPLES = 1000
CHECKPOINTS = 1
HUMAN_SPEED = 0.7
MIN_TRAJECTORY_LENGTH = HUMAN_SPEED * MIN_DURATION - 0.1  # enforce minimum path length


# === DATA GENERATION ===
def generate_split(n_samples: int,
                   output_path: Path,
                   dataset_name: str = "",
                   train_frac: float = 0.8,
                   val_frac: float = 0.1) -> None:
    """
    Generate and save JSON splits (train/val/test) under output_path.
    Each sample: 'prompt', 'trajectory' (10 points), 'duration'
    Trajectory starts at (0,0) excluded; ensures path length >= MIN_TRAJECTORY_LENGTH.
    Prompt appends 'heading forward' if end_y>0 else 'heading backward'.
    """
    all_data: List[Dict[str, Any]] = []
    for _ in range(n_samples):
        # compose prompt intro
        subj = random.choice(SUBJECTS)
        verb = random.choice(VERBS)
        adv = random.choice(ADVERBS)

        duration = random.randint(MIN_DURATION, MAX_DURATION)
        max_pos = duration * HUMAN_SPEED
        min_pos = max(MIN_TRAJECTORY_LENGTH, max_pos - 1)

        # sample until trajectory long enough
        while True:
            x0, y0 = 0.0, 0.0
            x1 = random.uniform(-max_pos, max_pos)
            y1 = random.uniform(-max_pos, max_pos)
            # linear trajectory checkpoints
            trajectory = []
            for i in range(1, CHECKPOINTS + 1):
                t = i / CHECKPOINTS
                xi = round(x0 * (1 - t) + x1 * t, 2)
                yi = round(y0 * (1 - t) + y1 * t, 2)
                trajectory.append([xi, yi])
            # compute path length
            total_dist = math.hypot(trajectory[0][0], trajectory[0][1])
            for i in range(1, len(trajectory)):
                xp, yp = trajectory[i - 1]
                xc, yc = trajectory[i]
                total_dist += math.hypot(xc - xp, yc - yp)
            if min_pos <= total_dist <= max_pos:
                break

        # duration scaling and clamp

        # determine heading based on final y
        heading = 'forward' if y1 > 0 else 'backward'
        prompt = f"{subj} {verb}{adv} {heading}, then stops."

        all_data.append({
            'prompt': prompt,
            'position': trajectory[0],  # fix
            'duration': int(duration)
        })

    # shuffle and split
    random.shuffle(all_data)
    n_train = int(train_frac * n_samples)
    n_val = int(val_frac * n_samples)
    splits = {
        'train': all_data[:n_train],
        'val': all_data[n_train:n_train + n_val],
        'test': all_data[n_train + n_val:]
    }

    output_path.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        fname = output_path / f"{dataset_name}{split}.json"
        with open(fname, 'w') as f:
            json.dump(items, f, indent=2)
        print(f"Wrote {len(items)} samples to {fname}")


# # === PYTORCH DATASET ===
# class TrajectoryPromptDataset(Dataset):
#     def __init__(self,
#                  data_dir: str,
#                  split: str,
#                  dataset_name: str = ""):
#         self.data_dir = Path(data_dir)
#         fname = f"{dataset_name}{split}.json"
#         with open(self.data_dir / fname, 'r') as f:
#             self.samples = json.load(f)
#
#     def __len__(self) -> int:
#         return len(self.samples)
#
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         item       = self.samples[idx]
#         prompt     = item['prompt']
#         trajectory = torch.tensor(item['trajectory'], dtype=torch.float)
#         duration   = torch.tensor(item['duration'], dtype=torch.float)
#         emb        = tmr_forward([prompt])[0]
#         return {'text': prompt,
#                 'trajectory': trajectory,
#                 'duration': duration,
#                 'tmr_text': emb}
#
#     @staticmethod
#     def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#         return {
#             'text':        [b['text'] for b in batch],
#             'durations':   torch.stack([b['duration'] for b in batch]),
#             'trajectories':torch.stack([b['trajectory'] for b in batch]),
#             'tmr_text':    torch.stack([b['tmr_text'] for b in batch])
#         }

if __name__ == '__main__':
    generate_split(n_samples=N_SAMPLES,
                   output_path=Path('./data'),
                   dataset_name='pos_')
