import torch
from torch.utils.data import Dataset, DataLoader
import json

from TMR.mtt.load_tmr_model import load_tmr_model_easy
tmr_forward = load_tmr_model_easy(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")

class MovementDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Mapping directions to integers
        self.direction_map = {
            'forward': 0,
            'backward': 1,
            'left': 2,
            'right': 3
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = item['prompt']
        duration = item['duration']
        steps = item['steps']

        # Encode steps: map directions to integers and keep distances
        directions = [self.direction_map[direction] for direction, _ in steps]
        distances = [distance for _, distance in steps]

        directions = torch.tensor(directions, dtype=torch.long)
        distances = torch.tensor(distances, dtype=torch.float)

        duration = torch.tensor(duration, dtype=torch.float)

        emb = tmr_forward([prompt])  # Get the embedding for the prompt

        return {
            'text': prompt,
            'length': duration * 20,
            'directions': directions,
            'distances': distances,
            'tmr_text': emb[0]
        }

def movment_collate_fn(batch):
    # Get the longest sequence length to pad
    max_directions_len = max([len(sample['directions']) for sample in batch])
    max_distances_len = max([len(sample['distances']) for sample in batch])

    directions_padded = []
    distances_padded = []
    lengths = []

    tmr = []

    for sample in batch:
        directions_len = len(sample['directions'])
        distances_len = len(sample['distances'])

        # Padding directions and distances with zeros to the max length
        directions_padding = torch.zeros(max_directions_len - directions_len, dtype=torch.long)
        distances_padding = torch.zeros(max_distances_len - distances_len, dtype=torch.float)

        directions_padded.append(torch.cat([sample['directions'], directions_padding]))
        distances_padded.append(torch.cat([sample['distances'], distances_padding]))
        lengths.append(sample['length'])
        tmr.append(sample['tmr_text'])

    # Stack tensors into a batch
    batch_output = {
        'tmr_text' : torch.stack(tmr),
        'text': [sample['text'] for sample in batch],  # Keeping text as is
        'length': torch.stack(lengths),  # Stack lengths
        'directions': torch.stack(directions_padded),  # Stack padded directions
        'distances': torch.stack(distances_padded)  # Stack padded distances
    }

    return batch_output

if __name__ == "__main__":
    dataset = MovementDataset('motion_test.json')  # Assuming you saved it as data.json
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)

    for batch in dataloader:
        print(batch[0]["length"])
        break