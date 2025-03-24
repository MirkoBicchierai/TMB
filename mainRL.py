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
import wandb

wandb.init(
    # Set the project where this run will be logged
    project="TM-BM",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name="experiment_1",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 1e-3,
        "epochs": 4,
    })

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

def render(x_starts, infos, smplh, joints_renderer, smpl_renderer, texts):

    out_formats = ['joints', 'txt', 'smpl','videojoints', 'videosmpl'] #
    joints = []
    smpls = []
    smpls_data = []
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
            np.save(path, x_start.detach().cpu())
            smpls.append(x_start.detach().cpu())

        if "joints" in out_formats:
            path = file_path + str(idx)+"_joints.npy"
            np.save(path, extracted_output["joints"])
            joints.append(extracted_output["joints"])

        if "vertices" in extracted_output and "vertices" in out_formats:
            path = file_path + str(idx)+"_verts.npy"
            np.save(path, extracted_output["vertices"])

        if "smpldata" in extracted_output and "smpldata" in out_formats:
            path = file_path + str(idx)+"_smpl.npz"
            np.savez(path, **extracted_output["smpldata"])
            smpls_data.append(**extracted_output["smpldata"])

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

    return smpls, joints, smpls_data

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

def reward(sequences, infos, smplh):

    batch_size = sequences.shape[0]
    device = sequences.device
    reward_scores = torch.zeros(batch_size, device=device)

    for idx in range(batch_size):

        x_start = sequences[idx]  # [200, 205]
        length = infos["all_lengths"][idx].item()  # Get length as integer
        x_start = x_start[:length]  # Truncate to actual length [length, 205]

        extracted_output = extract_joints(
            x_start.detach().cpu(),
            infos["featsname"],
            fps=infos["fps"],
            value_from="smpl",
            smpl_layer=smplh,
        )

        joints = extracted_output["joints"]  # [length, num_joints, 3]

        joints = torch.tensor(joints, device=device)
        # Compute velocity (difference between consecutive frames)
        if length >= 2:
            velocity = joints[1:] - joints[:-1]  # [length-1, J, 3]
            velocity_mag = torch.norm(velocity, dim=2)  # [length-1, J]
            avg_velocity = velocity_mag.mean()
        else:
            avg_velocity = torch.tensor(0.0, device=device)

        # Compute acceleration (difference between consecutive velocities)
        if length >= 3:
            acceleration = velocity[1:] - velocity[:-1]  # [length-2, J, 3]
            acceleration_mag = torch.norm(acceleration, dim=2)  # [length-2, J]
            avg_acceleration = acceleration_mag.mean()
        else:
            avg_acceleration = torch.tensor(0.0, device=device)

        # Combine velocity and smoothness (negative acceleration) into reward
        # Weights can be adjusted based on desired balance
        reward_scores[idx] = avg_velocity - 0.5 * avg_acceleration

    return reward_scores


def generate(model, train_dataloader, iteration, iterations, device, infos, text_model, smplh, joints_renderer, smpl_renderer):

    model.eval()
    train_bar = tqdm(train_dataloader, desc=f"Iteration {iteration + 1}/{iterations} [Generate new dataset]")

    examples = 10
    dataset = {
        "r": [],
        "xt_1": [],
        "xt": [],
        "t": [],
        "log_like": [],

        "batch-length": [],
        "batch-tx-x":[],
        "batch-tx-mask": [],
        "batch-x" : [],
        "batch-mask" : [],
        "batch-tx-length" : []
    }

    for batch_idx, batch in enumerate(train_bar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)
            sequences = model(tx_emb, tx_emb_uncond, infos)
            # smpls, joints, smpls_data = render(sequences, infos, smplh, joints_renderer, smpl_renderer, batch["text"])
            recon_loss, xt_1, xt, t, log_like = model.diffusion_stepRL(batch)
            r = reward(sequences, infos, smplh)


        # Append to dataset
        dataset["r"].append(r)
        dataset["xt_1"].append(xt_1)
        dataset["xt"].append(xt)
        dataset["t"].append(t)
        dataset["log_like"].append(log_like)

        dataset["batch-length"].append(batch["length"])
        dataset["batch-x"].append(batch["x"])
        dataset["batch-mask"].append(batch["mask"])

        tx = batch["tx"]
        dataset["batch-tx-x"].append(tx["x"])
        dataset["batch-tx-mask"].append(tx["mask"])
        dataset["batch-tx-length"].append(tx["length"])

        if (batch_idx + 1) % examples == 0:
            break

    for key in dataset:
        dataset[key] = torch.cat(dataset[key], dim=0)

    return dataset


def get_batch(dataset, i, minibatch_size):

    batch_length = dataset["batch-length"][i: i + minibatch_size]
    batch_x = dataset["batch-x"][i: i + minibatch_size]
    batch_tx_mask = dataset["batch-tx-mask"][i: i + minibatch_size]
    batch_tx_x = dataset["batch-tx-x"][i: i + minibatch_size]
    batch_tx_length = dataset["batch-tx-length"][i: i + minibatch_size]
    batch_mask = dataset["batch-mask"][i: i + minibatch_size]

    tx = {
        "x": batch_tx_x,
        "mask": batch_tx_mask,
        "length": batch_tx_length,
    }

    batch = {
        "tx": tx,
        "x": batch_x,
        "mask": batch_mask,
        "length": batch_length
    }

    return batch

def train(model, optimizer, dataset, iteration, iterations, device):

    model.train()

    delta = 1e-8
    dataset["advantage"] = (dataset["r"] - torch.mean(dataset["r"], dim=0)) / (torch.std(dataset["r"], dim=0) + delta)

    epochs = 4
    minibatch_size = 8

    train_bar = tqdm(range(epochs), desc=f"Iteration {iteration + 1}/{iterations} [Train]")
    for e in train_bar:
        epoch_advantages = []
        minibatch_bar = tqdm(range(0, dataset["r"].shape[0], minibatch_size), leave=False, desc="Minibatch")

        for minibatch in minibatch_bar:
            optimizer.zero_grad()

            r = dataset["r"][minibatch : minibatch + minibatch_size]
            xt_1 = dataset["xt_1"][minibatch : minibatch + minibatch_size]
            xt = dataset["xt"][minibatch : minibatch + minibatch_size]
            t = dataset["t"][minibatch : minibatch + minibatch_size]
            log_like = dataset["log_like"][minibatch : minibatch + minibatch_size]
            advantage = dataset["advantage"][minibatch : minibatch + minibatch_size].to(device)

            batch = get_batch(dataset,minibatch,minibatch_size)

            _, _, _, _, new_log_like = model.diffusion_stepRL(batch, t=t, xt=xt, A=xt_1)

            ratio = torch.exp(new_log_like-log_like)

            adv_per_ratio = -advantage*ratio

            epsilon = 0.2
            clipped_adv_per_ratio = -torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
            final_advantage = torch.max(adv_per_ratio, clipped_adv_per_ratio).mean()

            epoch_advantages.append(final_advantage.item())

            final_advantage.backward()
            optimizer.step()
            minibatch_bar.set_postfix(advantage=f"{final_advantage.item():.4f}")

        avg_advantage = sum(epoch_advantages) / len(epoch_advantages)
        train_bar.set_postfix(avg_advantage=f"{avg_advantage:.4f}")
        wandb.log({"loss": avg_advantage})


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

    diffusion_rl = instantiate(cfg.diffusion)
    diffusion_rl.load_state_dict(ckpt["state_dict"])
    diffusion_rl = diffusion_rl.to(device)

    train_dataset = instantiate(cfg.data, split="train")

    batch_size = 128
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

    lr = 1e-3
    optimizer = torch.optim.Adam(diffusion_rl.parameters(), lr=lr)

    iterations = 100
    for iteration in range(iterations):
        train_datasets_rl = generate(diffusion_rl, train_dataloader, iteration, iterations, device, infos, text_model, smplh, joints_renderer, smpl_renderer)
        train(diffusion_rl, optimizer, train_datasets_rl, iteration, iterations, device)


if __name__ == "__main__":
    main()
    wandb.finish()