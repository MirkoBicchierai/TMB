import os
import shutil
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from src.tools.smpl_layer import SMPLH
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import read_config
from src.tools.extract_joints import extract_joints
from src.model.text_encoder import TextToEmb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

def render(x_starts, out_formats, infos, smplh, joints_renderer, smpl_renderer, texts):

    for idx, (x_start, length, text) in enumerate(zip(x_starts, infos["all_lengths"], texts)):

        x_start = x_start[:length]

        extracted_output = extract_joints(
            x_start.detach().cpu(),
            infos["featsname"],
            fps= infos["fps"],
            value_from= "smpl",
            smpl_layer= smplh,
        )

        file_path = "ResultRL/"+str(idx)+"/"
        os.makedirs(file_path, exist_ok=True)

        if "smpl" in out_formats:
            path = file_path + str(idx)+"_smpl.npy"
            np.save(path, x_start)

        if "joints" in out_formats:
            path = file_path + str(idx)+"_joints.npy"
            np.save(path, extracted_output["joints"])

        if "vertices" in extracted_output and "vertices" in out_formats:
            path = file_path + str(idx)+"_verts.npy"
            np.save(path, extracted_output["vertices"])

        if "smpldata" in extracted_output and "smpldata" in out_formats:
            path = file_path + str(idx)+"_smpl.npz"
            np.savez(path, **extracted_output["smpldata"])

        if "videojoints" in out_formats:
            video_path = file_path + str(idx)+"_joints.mp4"
            joints_renderer(extracted_output["joints"], title="", output=video_path, canonicalize=False)

        if "vertices" in extracted_output and "videosmpl" in out_formats:
            print(f"SMPL rendering {idx}")
            video_path = file_path + str(idx)+"_smpl.mp4"
            smpl_renderer(extracted_output["vertices"], title="", output=video_path)

        if "txt" in out_formats:
            path = file_path + str(idx) + ".txt"
            with open(path, "w") as file:
                file.write(f"Motion:\n- {text}")

def get_embeddings(text_model, batch, device):
    with torch.no_grad():
        tx_emb = text_model(batch["text"])
        tx_emb_uncond = text_model(["" for _ in batch["text"]])

        if isinstance(tx_emb, torch.Tensor):
            tx_emb = {
                "x": tx_emb[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(device),
            }
            tx_emb_uncond = {
                "x": tx_emb_uncond[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(device),
            }
    return tx_emb, tx_emb_uncond

def train(diffusion_freeze, diffusion_rl, train_dataloader, epoch, epochs, device, infos, text_model, smplh, joints_renderer, smpl_renderer):
    diffusion_freeze.eval()
    diffusion_rl.train()

    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
    out_formats = ['videojoints','videosmpl','joints', 'txt']

    for batch_idx, batch in enumerate(train_bar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)
            x_starts = diffusion_freeze(tx_emb, tx_emb_uncond, infos)

        #loss, xstart = diffusion_rl.diffusion_stepRL(batch)

        #render(x_starts, out_formats, infos, smplh, joints_renderer, smpl_renderer, batch["text"])

def create_folder_results(name):
    results_dir = name
    os.makedirs(results_dir, exist_ok=True)

    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

@hydra.main(config_path="configs", config_name="TrainRL", version_base="1.3")
def main(c: DictConfig):

    create_folder_results("ResultRL")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = read_config(c.run_dir)

    ckpt_name = c.ckpt
    ckpt_path = os.path.join(c.run_dir, ckpt_name)
    print("Loading the checkpoint")
    ckpt = torch.load(str(ckpt_path), map_location=c.device)

    joints_renderer = instantiate(c.joints_renderer)
    smpl_renderer = instantiate(c.smpl_renderer)

    print("Loading the models")
    normalizer_dir = "pretrained_models/mdm-smpl_clip_smplrifke_humanml3d" if cfg.dataset=="humanml3d" else "pretrained_models/mdm-smpl_clip_smplrifke_kitml"
    cfg.diffusion.motion_normalizer.base_dir = os.path.join(normalizer_dir, "motion_stats")
    cfg.diffusion.text_normalizer.base_dir = os.path.join(normalizer_dir, "text_stats")

    diffusion_freeze = instantiate(cfg.diffusion)
    diffusion_freeze.load_state_dict(ckpt["state_dict"])
    diffusion_freeze = diffusion_freeze.to(device)

    diffusion_rl = instantiate(cfg.diffusion)
    diffusion_rl.load_state_dict(ckpt["state_dict"])
    diffusion_rl = diffusion_rl.to(device)

    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split="val")
    test_dataset = instantiate(cfg.data, split="test")

    batch_size = 24
    fps = 20
    time = 5

    infos = {
        "all_lengths": torch.tensor(np.full(batch_size, time * fps)).to(device),
        "featsname": cfg.motion_features,
        "fps": fps,
        "guidance_weight" : c.guidance
    }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=12,
        collate_fn=train_dataset.collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=12,
        collate_fn=val_dataset.collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=12,
        collate_fn=test_dataset.collate_fn
    )

    text_model = TextToEmb(
        modelpath=cfg.data.text_encoder.modelname, mean_pooling=cfg.data.text_encoder.mean_pooling, device=device
    )

    joint_stype = "both" # "smpljoints"

    smplh = SMPLH(
        path = "deps/smplh",
        jointstype = joint_stype,
        input_pose_rep = "axisangle",
        gender = c.gender,
    )

    epochs = 100

    for epoch in range(epochs):
        train(diffusion_freeze,diffusion_rl, train_dataloader, epoch, epochs, device, infos, text_model,smplh, joints_renderer, smpl_renderer)


if __name__ == "__main__":
    main()