import torch
from torch.utils.data import Dataset, DataLoader
import json

from TMR.mtt.load_tmr_model import load_tmr_model_easy

# load your TMR text‐encoder
tmr_forward = load_tmr_model_easy(device="cpu",
                                  dataset="tmr_humanml3d_kitml_guoh3dfeats")

class MovementDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        prompt  = item['prompt']
        duration = float(item['duration'])           # scalar
        position = item['position']                  # [x, y] list

        # get text embedding
        emb = tmr_forward([prompt])[0]               # assume emb shape (D,)

        return {
            'text':     prompt,
            'duration': torch.tensor(duration),      # torch.float32
            'length':   torch.tensor(duration * 20), # if you still need this
            'position': torch.tensor(position),      # torch.float32, shape (2,)
            'tmr_text': emb                          # torch.float32, shape (D,)
        }

def movement_collate_fn(batch):
    # batch is a list of dicts
    texts     = [b['text'] for b in batch]
    durations = torch.stack([b['duration'] for b in batch])
    lengths   = torch.stack([b['length']   for b in batch]).int()
    positions = torch.stack([b['position'] for b in batch])
    tmr_texts = torch.stack([b['tmr_text'] for b in batch])

    return {
        'text':      texts,
        'durations': durations,
        'length':   lengths,
        'positions': positions,
        'tmr_text':  tmr_texts
    }

if __name__ == "__main__":
    # example usage
    dataset    = MovementDataset('data/pos_train.json')
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=movement_collate_fn)

    batch = next(iter(dataloader))
    print(batch['lengths'][0])
