import os
import shutil

import numpy as np
import torch
import wandb
from torch import nn

from src.tools.extract_joints import extract_joints


def render(x_starts, infos, smplh, joints_renderer, smpl_renderer, texts, file_path, ty_log, out_formats, video_log=False):
    # out_formats = ['txt', 'smpl', 'joints', 'txt', 'smpl', 'videojoints', 'videosmpl']
    tmp = file_path

    for idx, (x_start, length, text) in enumerate(zip(x_starts, infos["all_lengths"], texts)):

        x_start = x_start[:length]

        extracted_output = extract_joints(
            x_start.detach().cpu(),
            infos["featsname"],
            fps=infos["fps"],
            value_from="smpl",
            smpl_layer=smplh,
        )

        file_path = tmp + str(idx) + "/"
        os.makedirs(file_path, exist_ok=True)

        if "smpl" in out_formats:
            path = file_path + str(idx) + "_smpl.npy"
            np.save(path, x_start.detach().cpu())

        if "joints" in out_formats:
            path = file_path + str(idx) + "_joints.npy"
            np.save(path, extracted_output["joints"])

        if "vertices" in extracted_output and "vertices" in out_formats:
            path = file_path + str(idx) + "_verts.npy"
            np.save(path, extracted_output["vertices"])

        if "smpldata" in extracted_output and "smpldata" in out_formats:
            path = file_path + str(idx) + "_smpl.npz"
            np.savez(path, **extracted_output["smpldata"])

        if "videojoints" in out_formats:
            video_path = file_path + str(idx) + "_joints.mp4"
            joints_renderer(extracted_output["joints"], title="", output=video_path, canonicalize=False)
            if video_log:
                wandb.log({ty_log: {"Video-joints": wandb.Video(video_path, format="mp4", caption=text)}})

        if "vertices" in extracted_output and "videosmpl" in out_formats:
            print(f"SMPL rendering {idx}")
            video_path = file_path + str(idx) + "_smpl.mp4"
            smpl_renderer(extracted_output["vertices"], title="", output=video_path)

        if "txt" in out_formats:
            path = file_path + str(idx) + ".txt"
            with open(path, "w") as file:
                file.write(text)


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


def get_embeddings_2(text_model, batch, n, device):
    with torch.no_grad():
        tx_emb = text_model(batch["text"])
        tx_emb_uncond = text_model(["" for _ in batch["text"]])

        if isinstance(tx_emb, torch.Tensor):
            tx_emb = {
                "x": tx_emb[:, None].repeat(n, 1, 1),
                "length": torch.tensor([1 for _ in range(len(tx_emb) * n)]).to(device),
            }
            tx_emb_uncond = {
                "x": tx_emb_uncond[:, None].repeat(n, 1, 1),
                "length": torch.tensor([1 for _ in range(len(tx_emb_uncond) * n)]).to(device),
            }
    return tx_emb, tx_emb_uncond

def create_folder_results(name):
    results_dir = name
    os.makedirs(results_dir, exist_ok=True)

    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def freeze_normalization_layers(model):
    for param in model.denoiser.parameters():
        param.requires_grad = True

    for name, module in model.denoiser.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for param in module.parameters():
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters DENOISER MODEL: {total_params}")
    print(
        f"Trainable parameters of DENOISER MODEL after freeze normalization layers: {trainable_params} ({trainable_params / total_params:.2%})")
    print(f"Frozen parameters after freeze normalization layers: {frozen_params} ({frozen_params / total_params:.2%})")