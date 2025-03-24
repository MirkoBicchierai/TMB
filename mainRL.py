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
    out_formats = ['joints', 'txt', 'smpl', 'videojoints', 'videosmpl']  #
    joints = []
    smpls = []
    smpls_data = []
    for idx, (x_start, length, text) in enumerate(zip(x_starts, infos["all_lengths"], texts)):

        x_start = x_start[:length]

        extracted_output = extract_joints(
            x_start.detach().cpu(),
            infos["featsname"],
            fps=infos["fps"],
            value_from="smpl",
            smpl_layer=smplh,
        )

        file_path = "ResultRL/" + str(idx) + "/"
        os.makedirs(file_path, exist_ok=True)

        if "smpl" in out_formats:
            path = file_path + str(idx) + "_smpl.npy"
            np.save(path, x_start.detach().cpu())
            smpls.append(x_start.detach().cpu())

        if "joints" in out_formats:
            path = file_path + str(idx) + "_joints.npy"
            np.save(path, extracted_output["joints"])
            joints.append(extracted_output["joints"])

        if "vertices" in extracted_output and "vertices" in out_formats:
            path = file_path + str(idx) + "_verts.npy"
            np.save(path, extracted_output["vertices"])

        if "smpldata" in extracted_output and "smpldata" in out_formats:
            path = file_path + str(idx) + "_smpl.npz"
            np.savez(path, **extracted_output["smpldata"])
            smpls_data.append(**extracted_output["smpldata"])

        if "videojoints" in out_formats:
            video_path = file_path + str(idx) + "_joints.mp4"
            joints_renderer(extracted_output["joints"], title="", output=video_path, canonicalize=False)

        if "vertices" in extracted_output and "videosmpl" in out_formats:
            print(f"SMPL rendering {idx}")
            video_path = file_path + str(idx) + "_smpl.mp4"
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


def reward(sequences):
    velocity = sequences[:, 1:, :] - sequences[:, :-1, :]
    velocity_magnitude = torch.norm(velocity, dim=-1)  # Shape: [48, 99]

    # Compute mean velocity per sequence (not across batch)
    score = -torch.mean(velocity_magnitude, dim=1)  # Shape: [48]

    return score


def generate(model, train_dataloader, iteration, iterations, device, infos, text_model, smplh, joints_renderer, smpl_renderer, num_examples):
    model.eval()
    train_bar = tqdm(train_dataloader, desc=f"Iteration {iteration + 1}/{iterations} [Generate new dataset]")

    dataset = {
        "r": [],
        "xt_1": [],
        "xt": [],
        "t": [],
        "log_like": [],

        "tx_emb_x":[],
        "tx_emb_mask": [],
        "tx_emb_length":[],
        "tx_emb_uncond_x": [],
        "tx_emb_uncond_mask": [],
        "tx_emb_uncond_length": [],
    }

    for batch_idx, batch in enumerate(train_bar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)
            sequences, results_by_timestep = model.diffusionRL(tx_emb, tx_emb_uncond, infos)
            # smpls, joints, smpls_data = render(sequences, infos, smplh, joints_renderer, smpl_renderer, batch["text"])
            r = reward(sequences)

        timesteps = sorted(results_by_timestep.keys(), reverse=True)
        batch_size = r.shape[0]

        # Calculate how many times we need to repeat the embeddings (once per timestep)
        num_timesteps = len(timesteps)

        # Store text embeddings just once, with repeat handling during concatenation
        tx_emb_x_repeated = tx_emb["x"].repeat(num_timesteps, 1, 1)
        tx_emb_mask_repeated = tx_emb["mask"].repeat(num_timesteps, 1)
        tx_emb_length_repeated = tx_emb["length"].repeat(num_timesteps)

        tx_emb_uncond_x_repeated = tx_emb_uncond["x"].repeat(num_timesteps, 1, 1)
        tx_emb_uncond_mask_repeated = tx_emb_uncond["mask"].repeat(num_timesteps, 1)
        tx_emb_uncond_length_repeated = tx_emb_uncond["length"].repeat(num_timesteps)

        dataset["tx_emb_x"].append(tx_emb_x_repeated)
        dataset["tx_emb_mask"].append(tx_emb_mask_repeated)
        dataset["tx_emb_length"].append(tx_emb_length_repeated)

        dataset["tx_emb_uncond_x"].append(tx_emb_uncond_x_repeated)
        dataset["tx_emb_uncond_mask"].append(tx_emb_uncond_mask_repeated)
        dataset["tx_emb_uncond_length"].append(tx_emb_uncond_length_repeated)

        # Process all timesteps and their corresponding outputs
        all_rewards = []
        all_xt_new = []
        all_xt_old = []
        all_t = []
        all_log_probs = []

        for i, t in enumerate(timesteps):
            experiment = results_by_timestep[t]

            if i == 0:
                all_rewards.append(r)
            else:
                all_rewards.append(torch.zeros_like(r))

            all_xt_new.append(experiment["xt_new"])
            all_xt_old.append(experiment["xt_old"])
            all_t.append(torch.full((batch_size,), t, device=r.device))
            all_log_probs.append(experiment["log_prob"])

        # Concatenate all the results for this batch
        dataset["r"].append(torch.cat(all_rewards, dim=0))
        dataset["xt_1"].append(torch.cat(all_xt_new, dim=0))
        dataset["xt"].append(torch.cat(all_xt_old, dim=0))
        dataset["t"].append(torch.cat(all_t, dim=0))
        dataset["log_like"].append(torch.cat(all_log_probs, dim=0))

        if (batch_idx + 1) % num_examples == 0:
            break

    for key in dataset:
        dataset[key] = torch.cat(dataset[key], dim=0)

    return dataset


def get_batch(dataset, i, minibatch_size):

    tx_emb_x = dataset["tx_emb_x"][i: i + minibatch_size]
    tx_emb_mask = dataset["tx_emb_mask"][i: i + minibatch_size]
    tx_emb_length = dataset["tx_emb_length"][i: i + minibatch_size]
    tx_emb_uncond_x = dataset["tx_emb_uncond_x"][i: i + minibatch_size]
    tx_emb_uncond_mask = dataset["tx_emb_uncond_mask"][i: i + minibatch_size]
    tx_emb_uncond_length = dataset["tx_emb_uncond_length"][i: i + minibatch_size]

    tx_emb= {
        "x": tx_emb_x,
        "mask": tx_emb_mask,
        "length": tx_emb_length
    }

    tx_emb_uncond= {
        "x": tx_emb_uncond_x,
        "mask": tx_emb_uncond_mask,
        "length": tx_emb_uncond_length
    }

    return tx_emb, tx_emb_uncond


def prepare_dataset(dataset):
    dataset_size = dataset["r"].shape[0]
    shuffle_indices = torch.randperm(dataset_size)

    for key in dataset:
        dataset[key] = dataset[key][shuffle_indices]

    return dataset


def train(model, optimizer, dataset, iteration, iterations, infos, device, batch_size, epochs):
    model.train()

    delta = 1e-8
    dataset["advantage"] = (dataset["r"] - torch.mean(dataset["r"], dim=0)) / (torch.std(dataset["r"], dim=0) + delta)

    num_minibatches = (dataset["r"].shape[0] + batch_size - 1) // batch_size

    train_bar = tqdm(range(epochs), desc=f"Iteration {iteration + 1}/{iterations} [Train]")
    for e in train_bar:
        tot_loss = 0
        minibatch_bar = tqdm(range(0, dataset["r"].shape[0], batch_size), leave=False, desc="Minibatch")
        dataset = prepare_dataset(dataset)
        for batch_idx in minibatch_bar:
            optimizer.zero_grad()

            r = dataset["r"][batch_idx: batch_idx + batch_size]
            xt_1 = dataset["xt_1"][batch_idx: batch_idx + batch_size]
            xt = dataset["xt"][batch_idx: batch_idx + batch_size]
            t = dataset["t"][batch_idx: batch_idx + batch_size]
            log_like = dataset["log_like"][batch_idx: batch_idx + batch_size]
            advantage = dataset["advantage"][batch_idx: batch_idx + batch_size].to(device)

            tx_emb, tx_emb_uncond = get_batch(dataset, batch_idx, batch_size)

            new_log_like = model.diffusionRL(tx_emb, tx_emb_uncond, infos, t=t, xt=xt, A=xt_1)

            ratio = torch.exp(new_log_like - log_like)

            adv_per_ratio = -advantage * ratio

            epsilon = 0.2
            clipped_adv_per_ratio = -torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
            final_advantage = torch.max(adv_per_ratio, clipped_adv_per_ratio).mean()

            tot_loss += final_advantage.item()

            final_advantage.backward()
            optimizer.step()
            minibatch_bar.set_postfix(advantage=f"{final_advantage.item():.4f}")

        epoch_loss = tot_loss / num_minibatches
        train_bar.set_postfix(avg_advantage=f"{epoch_loss:.4f}")
        wandb.log({"loss": epoch_loss})


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
    normalizer_dir = "pretrained_models/mdm-smpl_clip_smplrifke_humanml3d" if cfg.dataset == "humanml3d" else "pretrained_models/mdm-smpl_clip_smplrifke_kitml"
    cfg.diffusion.motion_normalizer.base_dir = os.path.join(normalizer_dir, "motion_stats")
    cfg.diffusion.text_normalizer.base_dir = os.path.join(normalizer_dir, "text_stats")

    diffusion_rl = instantiate(cfg.diffusion)
    diffusion_rl.load_state_dict(ckpt["state_dict"])
    diffusion_rl = diffusion_rl.to(device)

    train_dataset = instantiate(cfg.data, split="train")

    num_examples = 2
    batch_size = 32

    train_batch_size = 16
    train_epochs = 3

    fps = 20
    time = 5

    infos = {
        "all_lengths": torch.tensor(np.full(batch_size, time * fps)).to(device),
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

    lr = 2e-6
    #optimizer = torch.optim.AdamW(diffusion_rl.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    optimizer = torch.optim.AdamW(diffusion_rl.parameters(), lr=lr)

    iterations = 100
    for iteration in range(iterations):
        train_datasets_rl = generate(diffusion_rl, train_dataloader, iteration, iterations, device, infos, text_model,
                                     smplh, joints_renderer, smpl_renderer, num_examples)
        train(diffusion_rl, optimizer, train_datasets_rl, iteration, iterations, infos, device, train_batch_size, train_epochs)


if __name__ == "__main__":
    main()
    wandb.finish()
