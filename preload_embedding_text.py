import json

from tqdm import tqdm

from TMR.mtt.load_tmr_model import load_tmr_model_easy

tmr_forward = load_tmr_model_easy(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")

# Load the JSON data
with open('/motion_dataset/motion_test.json', 'r') as f:
    data = json.load(f)

# Iterate through each entry and add "text_tmr": "" to each annotation
for entry in tqdm(data):
    for annotation in entry.get("annotations", []):
        emb = tmr_forward([annotation["text"]])
        annotation["text_tmr"] = emb[0].tolist()

# Save the modified JSON back to file
with open('/home/mbicchierai/Tesi Magistrale/motion_dataset/motion_test_mod.json', 'w') as f:
    json.dump(data, f, indent=2)
