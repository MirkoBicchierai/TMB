import os
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"


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



def stillness_reward(sequences, infos, smplh, texts, regularization_weight=1, epsilon=1e-6):
    joint_positions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]

        output = extract_joints(
            x_start.detach().cpu(),
            'smplrifke',
            fps=20,
            value_from='smpl',
            smpl_layer=smplh,
        )

        joints = torch.as_tensor(output["joints"])
        joint_positions.append(joints)

    joints = torch.stack(joint_positions)  # (batch_size, N_frames, 22, 3)
    dt = 1.0 / 20

    # Compute velocity and acceleration
    velocities = torch.diff(joints, dim=1) / dt
    velocity_loss = torch.mean(velocities.pow(2), dim=(1, 2, 3))

    accelerations = torch.diff(velocities, dim=1) / dt
    acceleration_loss = torch.mean(accelerations.pow(2), dim=(1, 2, 3))

    # Compute displacement from initial pose
    initial_pose = joints[:, 0:1, :, :]
    displacement = torch.mean((joints - initial_pose).pow(2), dim=(1, 2, 3))

    # Final reward: encourage minimal movement
    reward = regularization_weight * (1.0 / (velocity_loss + acceleration_loss + displacement + epsilon))

    return reward


def generate(model, train_dataloader, iteration, iterations, device, infos, text_model, smplh, num_examples):
    model.eval()

    dataset = {
        "r": [],
        "xt_1": [],
        "xt": [],
        "t": [],
        "log_like": [],

        "mask":[],
        "length": [],
        "tx_x": [],
        "tx_mask": [],
        "tx_length": [],
        "tx_uncond_x": [],
        "tx_uncond_mask": [],
        "tx_uncond_length": [],

    }

    for batch_idx, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        generate_bar = tqdm(range(num_examples), desc=f"Iteration {iteration + 1}/{iterations} [Generate new dataset]")
        for i in generate_bar:

            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)
            sequences, results_by_timestep = model.diffusionRL(tx_emb=tx_emb, tx_emb_uncond=tx_emb_uncond, infos=infos, guidance_weight=1.0)

            r = stillness_reward(sequences, infos, smplh, batch["text"])

            timesteps = sorted(results_by_timestep.keys(), reverse=True)
            batch_size = r.shape[0]

            # Store text embeddings just once, with repeat handling during concatenation
            all_rewards = []
            all_xt_new = []
            all_xt_old = []
            all_t = []
            all_log_probs = []

            #y
            all_mask = []
            all_lengths = []
            all_tx_x = []
            all_tx_length = []
            all_tx_mask = []
            all_tx_uncond_x = []
            all_tx_uncond_length = []
            all_tx_uncond_mask = []


            for t in timesteps:
                experiment = results_by_timestep[t]
                experiment = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in experiment.items()}

                if t == 0:
                    all_rewards.append(r.cpu())
                else:
                    all_rewards.append(torch.zeros_like(r).cpu())

                all_xt_new.append(experiment["xt_new"])
                all_xt_old.append(experiment["xt_old"])
                all_t.append(torch.full((batch_size,), t, device=r.device).cpu())
                all_log_probs.append(experiment["log_prob"])

                #y
                all_mask.append(experiment["mask"])
                all_lengths.append(experiment["length"])
                all_tx_x.append(experiment["tx-x"])
                all_tx_length.append(experiment["tx-length"])
                all_tx_mask.append(experiment["tx-mask"])
                all_tx_uncond_x.append(experiment["tx_uncond-x"])
                all_tx_uncond_length.append(experiment["tx_uncond-length"])
                all_tx_uncond_mask.append(experiment["tx_uncond-mask"])

            # Concatenate all the results for this batch
            dataset["r"].append(torch.cat(all_rewards, dim=0).view(100, batch_size).T.clone())
            dataset["xt_1"].append(torch.cat(all_xt_new, dim=0).view(100, batch_size, 100, 205).permute(1, 0, 2, 3))
            dataset["xt"].append(torch.cat(all_xt_old, dim=0).view(100, batch_size, 100, 205).permute(1, 0, 2, 3))
            dataset["t"].append(torch.cat(all_t, dim=0).view(100, batch_size).T)
            dataset["log_like"].append(torch.cat(all_log_probs, dim=0).view(100, batch_size).T)

            #y
            dataset["mask"].append(torch.cat(all_mask, dim=0).view(100, batch_size, 100).permute(1, 0, 2))
            dataset["length"].append(torch.cat(all_lengths, dim=0).view(100, batch_size).T)
            dataset["tx_x"].append(torch.cat(all_tx_x, dim=0).view(100, batch_size, 1, 512).permute(1, 0, 2, 3))
            dataset["tx_length"].append(torch.cat(all_tx_length, dim=0).view(100, batch_size).T)
            dataset["tx_mask"].append(torch.cat(all_tx_mask, dim=0).view(100, batch_size, 1).permute(1, 0, 2))
            dataset["tx_uncond_x"].append(torch.cat(all_tx_uncond_x, dim=0).view(100, batch_size, 1, 512).permute(1, 0, 2, 3))
            dataset["tx_uncond_length"].append(torch.cat(all_tx_uncond_length, dim=0).view(100, batch_size).T)
            dataset["tx_uncond_mask"].append(torch.cat(all_tx_uncond_mask, dim=0).view(100, batch_size, 1).permute(1, 0, 2))

        break

    for key in dataset:
        dataset[key] = torch.cat(dataset[key], dim=0)

    return dataset


def get_y(dataset, i, minibatch_size, infos, diff_step, device):
    tx_x = dataset["tx_x"][i: i + minibatch_size]
    tx_mask = dataset["tx_mask"][i: i + minibatch_size]
    tx_length = dataset["tx_length"][i: i + minibatch_size]
    tx_uncond_x = dataset["tx_uncond_x"][i: i + minibatch_size]
    tx_uncond_mask = dataset["tx_uncond_mask"][i: i + minibatch_size]
    tx_uncond_length = dataset["tx_uncond_length"][i: i + minibatch_size]

    mask = dataset["mask"][i: i + minibatch_size]
    lengths = dataset["length"][i: i + minibatch_size]

    tx = {
        "x": tx_x.view(diff_step * minibatch_size, *tx_x.shape[2:]).to(device),
        "mask": tx_mask.view(diff_step * minibatch_size, *tx_mask.shape[2:]).to(device),
        "length": tx_length.view(diff_step * minibatch_size).to(device)
    }

    tx_uncond = {
        "x": tx_uncond_x.view(diff_step * minibatch_size, *tx_uncond_x.shape[2:]).to(device),
        "mask": tx_uncond_mask.view(diff_step * minibatch_size, *tx_uncond_mask.shape[2:]).to(device),
        "length": tx_uncond_length.view(diff_step * minibatch_size).to(device)
    }

    y = {
        "length": lengths.view(diff_step * minibatch_size).to(device),
        "mask": mask.view(diff_step * minibatch_size, *mask.shape[2:]).to(device),
        "tx": tx,
        "tx_uncond": tx_uncond,
        "infos": infos,
    }

    return y


def prepare_dataset(dataset):
    dataset_size = dataset["r"].shape[0]
    shuffle_indices = torch.randperm(dataset_size)

    for key in dataset:
        dataset[key] = dataset[key][shuffle_indices]

    return dataset


def train(model, optimizer, dataset, iteration, iterations, infos, device, batch_size, epochs):
    model.train()

    delta = 1e-7
    mask = dataset["r"] != 0
    mean_r = torch.mean(dataset["r"][mask], dim=0)
    std_r = torch.std(dataset["r"][mask], dim=0)

    dataset["advantage"] = torch.zeros_like(dataset["r"])
    dataset["advantage"][mask] = (dataset["r"][mask] - mean_r) / (std_r + delta)
    # dataset["advantage"] = (dataset["r"] - mean_r) / (std_r + delta)

    num_minibatches = (dataset["r"].shape[0] + batch_size - 1) // batch_size

    diff_step = dataset["xt_1"][0].shape[1]

    train_bar = tqdm(range(epochs), desc=f"Iteration {iteration + 1}/{iterations} [Train]")
    for e in train_bar:
        tot_loss = 0
        minibatch_bar = tqdm(range(0, dataset["r"].shape[0], batch_size), leave=False, desc="Minibatch")
        dataset = prepare_dataset(dataset)
        for batch_idx in minibatch_bar:
            optimizer.zero_grad()

            # r = dataset["r"][batch_idx: batch_idx + batch_size].to(device)
            xt_1 = dataset["xt_1"][batch_idx: batch_idx + batch_size].to(device)
            xt = dataset["xt"][batch_idx: batch_idx + batch_size].to(device)
            t = dataset["t"][batch_idx: batch_idx + batch_size].to(device)
            log_like = dataset["log_like"][batch_idx: batch_idx + batch_size].to(device)
            advantage = dataset["advantage"][batch_idx: batch_idx + batch_size].to(device)

            real_batch_size = xt_1.shape[0]
            y = get_y(dataset, batch_idx, real_batch_size, infos, diff_step, device)

            new_log_like = model.diffusionRL(y=y, infos=infos, t=t.view(diff_step * real_batch_size),
                                                 xt=xt.view(diff_step * real_batch_size, *xt.shape[2:]),
                                                 A=xt_1.view(diff_step * real_batch_size, *xt_1.shape[2:]),
                                                 guidance_weight=1.0).view(real_batch_size, diff_step )

            ratio = torch.exp(new_log_like - log_like)

            epsilon = 1e-4 # 0.2
            clip_adv = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
            policy_loss = -torch.min(ratio * advantage, clip_adv).sum(1).mean()

            tot_loss += policy_loss.item()

            policy_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            minibatch_bar.set_postfix(policy_loss=f"{policy_loss.item():.4f}")

        epoch_loss = tot_loss / num_minibatches
        train_bar.set_postfix(epoch_loss=f"{epoch_loss:.4f}")


@hydra.main(config_path="configs", config_name="TrainRL", version_base="1.3")
def main(c: DictConfig):
    # create_folder_results("ResultRL")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = read_config(c.run_dir)

    ckpt_name = c.ckpt
    ckpt_path = os.path.join(c.run_dir, ckpt_name)
    print("Loading the checkpoint")
    ckpt = torch.load(str(ckpt_path), map_location=c.device)

    text_model = TextToEmb(
        modelpath=cfg.data.text_encoder.modelname, mean_pooling=cfg.data.text_encoder.mean_pooling, device=device
    )

    joint_stype = "both"  # "smpljoints"

    smplh = SMPLH(
        path="deps/smplh",
        jointstype=joint_stype,
        input_pose_rep="axisangle",
        gender=c.gender,
    )

    print("Loading the models")
    normalizer_dir = "pretrained_models/mdm-smpl_clip_smplrifke_humanml3d" if cfg.dataset == "humanml3d" else "pretrained_models/mdm-smpl_clip_smplrifke_kitml"
    cfg.diffusion.motion_normalizer.base_dir = os.path.join(normalizer_dir, "motion_stats")
    cfg.diffusion.text_normalizer.base_dir = os.path.join(normalizer_dir, "text_stats")

    diffusion_rl = instantiate(cfg.diffusion)
    diffusion_rl.load_state_dict(ckpt["state_dict"])
    diffusion_rl = diffusion_rl.to(device)

    train_dataset = instantiate(cfg.data, split="train")

    # generations dataset params
    num_examples = 4
    batch_size = 2

    # training params
    iterations = 10000
    train_epochs = 5
    train_batch_size = 6

    # optimizer params
    lr = 2e-6
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay=1e-2

    # validation/test params
    val_iter = 1000
    val_batch_size = 64

    fps = 20
    time = 5
    infos = {
        "all_lengths": torch.tensor(np.full(2048, time * fps)).to(device),
        "featsname": cfg.motion_features,
        "fps": fps,
        "guidance_weight": c.guidance
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


    optimizer = torch.optim.AdamW(diffusion_rl.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


    for iteration in range(iterations):

        train_datasets_rl = generate(diffusion_rl, train_dataloader, iteration, iterations, device, infos, text_model, smplh, num_examples)
        train(diffusion_rl, optimizer, train_datasets_rl, iteration, iterations, infos, device, train_batch_size, train_epochs)



if __name__ == "__main__":
    main()
    wandb.finish()
