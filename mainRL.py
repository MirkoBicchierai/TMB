import argparse
import itertools
import os
import shutil
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.tools.smpl_layer import SMPLH
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import read_config
from src.tools.extract_joints import extract_joints
from src.model.text_encoder import TextToEmb
import wandb
from colorama import Fore, Style, init
from TMR.mtt.load_tmr_model import load_tmr_model_easy
from src.tools.guofeats.motion_representation import joints_to_guofeats
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.model.tmr import get_sim_matrix

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

tmr_forward = load_tmr_model_easy(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")


def render(x_starts, infos, smplh, joints_renderer, smpl_renderer, texts, file_path, video_log=False):
    out_formats = ['txt', 'videojoints', 'smpl']  # 'joints', 'txt', 'smpl', 'videojoints', 'videosmpl'
    tmp = file_path
    ty_log = ""

    if "VAL" in file_path:
        ty_log = "Validation"
    else:
        ty_log = "Test"

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


def get_embeddings_2(text_model, batch,n, device):
    with torch.no_grad():
        tx_emb = text_model(batch["text"])
        tx_emb_uncond = text_model(["" for _ in batch["text"]])

        if isinstance(tx_emb, torch.Tensor):
            tx_emb = {
                "x": tx_emb[:, None].repeat(n, 1, 1),
                "length": torch.tensor([1 for _ in range(len(tx_emb)*n)]).to(device),
            }
            tx_emb_uncond = {
                "x": tx_emb_uncond[:, None].repeat(n, 1, 1),
                "length": torch.tensor([1 for _ in range(len(tx_emb_uncond)*n)]).to(device),
            }
    return tx_emb, tx_emb_uncond


def stillness_reward(sequences, infos, smplh):
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

    joints = torch.stack(joint_positions)
    dt = 1.0 / 200

    velocities = torch.diff(joints, dim=1) / dt
    velocity_loss = torch.mean(velocities.pow(2), dim=(1, 2, 3))

    reward = velocity_loss
    return - reward


def smpl_to_guofeats(smpl, smplh):
    guofeats = []
    for i in smpl:
        i_output = extract_joints(
            i,
            'smplrifke',
            fps=20,
            value_from='smpl',
            smpl_layer=smplh,
        )
        i_joints = i_output["joints"]  # tensor(N, 22, 3)
        # convert to guofeats, first, make sure to revert the axis, as guofeats have gravity axis in Y
        x, y, z = i_joints.T
        i_joints = np.stack((x, z, -y), axis=0).T
        i_guofeats = joints_to_guofeats(i_joints)
        guofeats.append(i_guofeats)

    return guofeats


def calc_eval_stats(x, smplh):
    x_guofeats = smpl_to_guofeats(x, smplh)
    x_latents = tmr_forward(x_guofeats)  # tensor(N, 256)
    return x_latents


def is_list_of_strings(var):
    return isinstance(var, list) and all(isinstance(item, str) for item in var)


def print_matrix_nicely(matrix: np.ndarray):
    init(autoreset=True)
    for row in matrix:
        max_val = np.max(row)
        line = ""
        for val in row:
            truncated = int(val * 1000) / 1000
            formatted = f"{truncated:.3f}"
            if val == max_val:
                line += f"{Fore.GREEN}{formatted}{Style.RESET_ALL}  "
            else:
                line += f"{formatted}  "
        print(line)


def tmr_reward_special(sequences, infos, smplh, texts, all_embedding_tmr, c):
    motions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]
        motions.append(x_start.detach().cpu())

    x_latents = calc_eval_stats(motions, smplh)
    sim_matrix = get_sim_matrix(x_latents, texts.detach().cpu().type(x_latents.dtype)).numpy()
    # print_matrix_nicely(sim_matrix)

    sim_matrix = torch.tensor(sim_matrix)
    classic_tmr = sim_matrix.diagonal()

    if c.tmr_reward:
        return classic_tmr * c.reward_scale, classic_tmr
    else:

        sim_matrix_tmp = get_sim_matrix(x_latents, all_embedding_tmr.detach().cpu().type(x_latents.dtype)).numpy()
        # print_matrix_nicely(sim_matrix_tmp)

        sim_matrix_tmp = (sim_matrix_tmp + 1) / 2
        sim_matrix = (sim_matrix + 1) / 2
        diagonal_values = sim_matrix.diagonal()

        # Calculate similarity between texts and all_embedding_tmr and find the most similar embedding in all_embedding_tmr
        text_to_all_sim = torch.matmul(texts.detach().cpu(), all_embedding_tmr.transpose(0, 1))
        matching_indices = torch.argmax(text_to_all_sim, dim=1)

        special = []
        for i in range(sim_matrix_tmp.shape[0]):
            # Get the index to exclude for this row
            exclude_idx = matching_indices[i].item()
            # Make a copy of the row and set the element to exclude to NaN
            row_copy = sim_matrix_tmp[i].copy()
            row_copy[exclude_idx] = np.nan

            num_elements = len(row_copy)
            num_to_nan = int(num_elements * c.masking_ratio)
            indices_to_nan = np.random.choice(num_elements, num_to_nan, replace=False)
            row_copy[indices_to_nan] = np.nan

            # Calculate mean without the excluded element
            row_mean = np.nanmean(row_copy)
            # Calculate special value for this row (real - mean of row of all emb)
            special_value = diagonal_values[i] - row_mean
            special.append(special_value)

        special = torch.tensor(special)

    return special * c.reward_scale, classic_tmr


def preload_tmr_text(dataloader):
    all_embeddings = []
    for batch_idx, batch in enumerate(dataloader):
        all_embeddings.append(batch["tmr_text"])
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


@torch.no_grad()
def generate(model, train_dataloader, iteration, c, device, infos, text_model, smplh,
             train_embedding_tmr):  # , generation_iter
    model.train()

    dataset = {
        "r": [],
        "xt_1": [],
        "xt": [],
        "t": [],
        "log_like": [],

        "tmr": [],

        "mask": [],
        "length": [],
        "tx_x": [],
        "tx_mask": [],
        "tx_length": [],
        "tx_uncond_x": [],
        "tx_uncond_mask": [],
        "tx_uncond_length": [],

    }

    generate_bar = tqdm(enumerate(itertools.islice(itertools.cycle(train_dataloader), 1)),
                        desc=f"Iteration {iteration + 1}/{c.iterations} [Generate new dataset]",
                        total=1, leave=False)

    for batch_idx, batch in generate_bar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        tx_emb, tx_emb_uncond = get_embeddings_2(text_model, batch, c.num_gen_per_prompt, device)

        sequences, results_by_timestep = model.diffusionRL(tx_emb=tx_emb, tx_emb_uncond=tx_emb_uncond, infos=infos,
                                                           guidance_weight=c.guidance_weight_generation)

        reward, tmr = tmr_reward_special(sequences, infos, smplh, batch["tmr_text"].repeat(c.num_gen_per_prompt, 1), train_embedding_tmr, c)

        timesteps = sorted(results_by_timestep.keys(), reverse=True)
        diff_step = len(timesteps)

        batch_size = reward.shape[0]
        seq_len = results_by_timestep[0]["xt_new"].shape[1]

        # Store text embeddings just once, with repeat handling during concatenation
        all_rewards = []
        all_tmr = []
        all_xt_new = []
        all_xt_old = []
        all_t = []
        all_log_probs = []

        # y
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
                all_rewards.append(reward.cpu())
                all_tmr.append(tmr.cpu())
            else:
                all_rewards.append(torch.zeros_like(reward).cpu())
                all_tmr.append(torch.zeros_like(tmr).cpu())

            all_xt_new.append(experiment["xt_new"])
            all_xt_old.append(experiment["xt_old"])
            all_t.append(torch.full((batch_size,), t, device=reward.device).cpu())
            all_log_probs.append(experiment["log_prob"])

            # y
            all_mask.append(experiment["mask"])
            all_lengths.append(experiment["length"])
            all_tx_x.append(experiment["tx-x"])
            all_tx_length.append(experiment["tx-length"])
            all_tx_mask.append(experiment["tx-mask"])
            all_tx_uncond_x.append(experiment["tx_uncond-x"])
            all_tx_uncond_length.append(experiment["tx_uncond-length"])
            all_tx_uncond_mask.append(experiment["tx_uncond-mask"])


        # Concatenate all the results for this batch
        dataset["r"].append(torch.cat(all_rewards, dim=0).view(diff_step, batch_size).T.clone())
        dataset["tmr"].append(torch.cat(all_tmr, dim=0).view(diff_step, batch_size).T.clone())
        dataset["xt_1"].append(
            torch.cat(all_xt_new, dim=0).view(diff_step, batch_size, seq_len, 205).permute(1, 0, 2, 3))
        dataset["xt"].append(
            torch.cat(all_xt_old, dim=0).view(diff_step, batch_size, seq_len, 205).permute(1, 0, 2, 3))
        dataset["t"].append(torch.cat(all_t, dim=0).view(diff_step, batch_size).T)
        dataset["log_like"].append(torch.cat(all_log_probs, dim=0).view(diff_step, batch_size).T)

        # y
        dataset["mask"].append(torch.cat(all_mask, dim=0).view(diff_step, batch_size, seq_len).permute(1, 0, 2))
        dataset["length"].append(torch.cat(all_lengths, dim=0).view(diff_step, batch_size).T)
        dataset["tx_x"].append(torch.cat(all_tx_x, dim=0).view(diff_step, batch_size, 1, 512).permute(1, 0, 2, 3))
        dataset["tx_length"].append(torch.cat(all_tx_length, dim=0).view(diff_step, batch_size).T)
        dataset["tx_mask"].append(torch.cat(all_tx_mask, dim=0).view(diff_step, batch_size, 1).permute(1, 0, 2))
        dataset["tx_uncond_x"].append(
            torch.cat(all_tx_uncond_x, dim=0).view(diff_step, batch_size, 1, 512).permute(1, 0, 2, 3))
        dataset["tx_uncond_length"].append(torch.cat(all_tx_uncond_length, dim=0).view(diff_step, batch_size).T)
        dataset["tx_uncond_mask"].append(
            torch.cat(all_tx_uncond_mask, dim=0).view(diff_step, batch_size, 1).permute(1, 0, 2))

    for key in dataset:
        dataset[key] = torch.cat(dataset[key], dim=0)

    return dataset


def get_batch(dataset, i, minibatch_size, infos, diff_step, device):
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

    r = dataset["r"][i: i + minibatch_size].to(device)
    xt_1 = dataset["xt_1"][i: i + minibatch_size].to(device)
    xt = dataset["xt"][i: i + minibatch_size].to(device)
    t = dataset["t"][i: i + minibatch_size].to(device)
    log_like = dataset["log_like"][i: i + minibatch_size].to(device)

    return y, r, xt_1, xt, t, log_like


def prepare_dataset(dataset):
    dataset_size = dataset["r"].shape[0]
    shuffle_indices = torch.randperm(dataset_size)

    for key in dataset:
        dataset[key] = dataset[key][shuffle_indices]

    return dataset


def train(model, optimizer, dataset, iteration, c, infos, device, old_model=None):
    model.train()
    delta = 1e-7
    mask = dataset["r"] != 0
    mean_r = torch.mean(dataset["r"][mask], dim=0)
    std_r = torch.std(dataset["r"][mask], dim=0)

    mean_tmr = torch.mean(dataset["tmr"][mask], dim=0)
    std_tmr = torch.std(dataset["tmr"][mask], dim=0)

    wandb.log({"Train": {"Mean Reward": mean_r.item(), "Std Reward": std_r.item(), "Mean TMR": mean_tmr.item(),
                         "Std TMR": std_tmr.item(), "iterations": iteration}})

    dataset["advantage"] = torch.zeros_like(dataset["r"])
    dataset["advantage"][mask] = (dataset["r"][mask] - mean_r) / (std_r + delta)
    dataset["advantage"] = (dataset["r"] - mean_r) / (std_r + delta)

    num_minibatches = (dataset["r"].shape[0] + c.train_batch_size - 1) // c.train_batch_size

    diff_step = dataset["xt_1"][0].shape[0]

    train_bar = tqdm(range(c.train_epochs), desc=f"Iteration {iteration + 1}/{c.iterations} [Train]", leave=False)
    for e in train_bar:
        tot_loss = 0
        tot_kl = 0
        tot_policy_loss = 0
        epoch_clipped_elements = 0
        epoch_total_elements = 0

        minibatch_bar = tqdm(range(0, dataset["r"].shape[0], c.train_batch_size), leave=False, desc="Minibatch")
        dataset = prepare_dataset(dataset)
        for batch_idx in minibatch_bar:
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                advantage = dataset["advantage"][batch_idx: batch_idx + c.train_batch_size].to(device)
                real_batch_size = advantage.shape[0]
                y, r, xt_1, xt, t, log_like = get_batch(dataset, batch_idx, real_batch_size, infos, diff_step, device)

                new_log_like, rl_pred = model.diffusionRL(y=y, infos=infos, t=t.view(diff_step * real_batch_size),
                                                          xt=xt.view(diff_step * real_batch_size, *xt.shape[2:]),
                                                          A=xt_1.view(diff_step * real_batch_size, *xt_1.shape[2:]),
                                                          guidance_weight=c.guidance_weight_train)
                new_log_like = new_log_like.view(real_batch_size, diff_step)
                rl_pred = rl_pred.view(real_batch_size, diff_step, *rl_pred.shape[1:])

                if c.betaL > 0:
                    _, old_pred = old_model.diffusionRL(y=y, infos=infos, t=t.view(diff_step * real_batch_size),
                                                        xt=xt.view(diff_step * real_batch_size, *xt.shape[2:]),
                                                        A=xt_1.view(diff_step * real_batch_size, *xt_1.shape[2:]),
                                                        guidance_weight=c.guidance_weight_train)
                    old_pred = old_pred.view(real_batch_size, diff_step, *old_pred.shape[1:])
                    kl_div = ((rl_pred - old_pred) ** 2).sum(1).mean()

            ratio = torch.exp(new_log_like - log_like)

            real_adv = advantage[:, -1:]  # r[:,-1:]

            # Count how many elements need clipping
            lower_bound = 1.0 - c.advantage_clip_epsilon
            upper_bound = 1.0 + c.advantage_clip_epsilon

            too_small = (ratio < lower_bound).sum().item()
            too_large = (ratio > upper_bound).sum().item()
            current_clipped = too_small + too_large
            epoch_clipped_elements += current_clipped
            current_total = ratio.numel()
            epoch_total_elements += current_total

            clip_adv = torch.clamp(ratio, lower_bound, upper_bound) * real_adv
            policy_loss = -torch.min(ratio * real_adv, clip_adv).sum(1).mean()

            if c.betaL > 0:
                combined_loss = c.alphaL * policy_loss + c.betaL * kl_div
                tot_kl += kl_div.item()
            else:
                combined_loss = c.alphaL * policy_loss

            combined_loss.backward()
            tot_loss += combined_loss.item()
            tot_policy_loss += policy_loss.item()

            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None))
            wandb.log({"Train": {"Gradient Norm": grad_norm.item(),
                                 "real_step": (iteration * c.train_epochs + e) * num_minibatches + (
                                         batch_idx // c.train_batch_size)}})

            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
            optimizer.step()

            minibatch_bar.set_postfix(batch_loss=f"{combined_loss.item():.4f}")

        epoch_loss = tot_loss / num_minibatches
        epoch_policy_loss = tot_policy_loss / num_minibatches
        clipping_percentage = 100 * epoch_clipped_elements / epoch_total_elements

        train_bar.set_postfix(epoch_loss=f"{epoch_loss:.4f}")

        if c.betaL > 0:
            epoch_kl = tot_kl / num_minibatches
            wandb.log({"Train": {"loss": epoch_loss, "epochs": iteration * c.train_epochs + e,
                                 "policy_loss": epoch_policy_loss, "kl_loss": epoch_kl,
                                 "trigger-clip": clipping_percentage}})
        else:
            wandb.log({"Train": {"loss": epoch_loss, "epochs": iteration * c.train_epochs + e,
                                 "policy_loss": epoch_policy_loss, "trigger-clip": clipping_percentage}})


def test(model, dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer, c, all_embedding_tmr,
         path):
    os.makedirs(path, exist_ok=True)

    model.eval()

    if c.val_num_batch == 0:
        generate_bar = tqdm(enumerate(dataloader), leave=False, desc=f"[Validation/Test Generations]")
    else:
        generate_bar = tqdm(enumerate(itertools.islice(itertools.cycle(dataloader), c.val_num_batch)),
                            total=c.val_num_batch, leave=False, desc=f"[Validation/Test Generations]")

    total_reward, total_tmr = 0, 0
    batch_count_reward, batch_count_tmr = 0, 0

    for batch_idx, batch in generate_bar:
        tmp_path = path + "batch_" + str(batch_idx) + "/"
        os.makedirs(tmp_path, exist_ok=True)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)
            sequences, _ = model.diffusionRL(tx_emb, tx_emb_uncond, infos, guidance_weight=c.guidance_weight_valid)
            render(sequences, infos, smplh, joints_renderer, smpl_renderer, batch["text"], tmp_path,
                   video_log=True if batch_idx == 0 else False)
            reward, tmr = tmr_reward_special(sequences, infos, smplh, batch["tmr_text"], all_embedding_tmr,
                                             c)  # shape [batch_size]

            total_reward += reward.sum().item()
            batch_count_reward += reward.shape[0]

            total_tmr += tmr.sum().item()
            batch_count_tmr += tmr.shape[0]

    avg_reward = total_reward / batch_count_reward
    avg_tmr = total_tmr / batch_count_tmr

    return avg_reward, avg_tmr


def create_folder_results(name):
    results_dir = name
    os.makedirs(results_dir, exist_ok=True)

    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def freeze_except_last_layers(model):
    # Freeze all parameters by default
    for param in model.denoiser.parameters():
        param.requires_grad = False

    # Unfreeze the last linear layer
    for param in model.denoiser.to_skel_layer.parameters():
        param.requires_grad = True

    # Unfreeze the last transformer layer in seqTransEncoder
    # TransformerEncoder stores layers in a ModuleList called 'layers'
    last_transformer_layer_index = len(model.denoiser.seqTransEncoder.layers) - 1
    for param in model.denoiser.seqTransEncoder.layers[last_transformer_layer_index].parameters():
        param.requires_grad = True

    # Print status to verify
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({trainable_params / total_params:.2%})")
    print(f"Frozen parameters: {frozen_params} ({frozen_params / total_params:.2%})")


def freeze_normalization_layers(model):
    # First, make sure all parameters are trainable by default
    for param in model.denoiser.parameters():
        param.requires_grad = True

    # Then, freeze only the normalization layers
    for name, module in model.denoiser.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for param in module.parameters():
                param.requires_grad = False

    # Print status to verify
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({trainable_params / total_params:.2%})")
    print(f"Frozen parameters: {frozen_params} ({frozen_params / total_params:.2%})")

@hydra.main(config_path="configs", config_name="TrainRL", version_base="1.3")
def main(c: DictConfig):
    # args = parse_arguments()#
    config_dict = OmegaConf.to_container(c, resolve=True)
    wandb.init(
        project="TM-BM",
        name=c.experiment_name,
        config=config_dict,
        group=c.group_name
    )

    # generation_iter = c.generated_dataset_size // (c.num_gen_per_prompt * c.num_prompts_dataset)

    create_folder_results("ResultRL")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = read_config(c.run_dir)

    ckpt_path = os.path.join(c.run_dir, c.ckpt_name)
    print("Loading the checkpoint")
    ckpt = torch.load(str(ckpt_path), map_location=c.device)

    joints_renderer = instantiate(c.joints_renderer)
    smpl_renderer = instantiate(c.smpl_renderer)

    text_model = TextToEmb(
        modelpath=cfg.data.text_encoder.modelname, mean_pooling=cfg.data.text_encoder.mean_pooling, device=device
    )

    smplh = SMPLH(
        path="deps/smplh",
        jointstype=c.joint_stype,
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

    #freeze_except_last_layers(diffusion_rl)
    freeze_normalization_layers(diffusion_rl)

    if c.betaL > 0:
        diffusion_old = instantiate(cfg.diffusion)
        diffusion_old.load_state_dict(ckpt["state_dict"])
        diffusion_old = diffusion_old.to(device)
        diffusion_old.eval()
    else:
        diffusion_old = None

    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split="val")
    test_dataset = instantiate(cfg.data, split="val")

    infos = {
        "all_lengths": torch.tensor(np.full(2048, int(c.time * c.fps))).to(device),
        "featsname": cfg.motion_features,
        "fps": c.fps,
        "guidance_weight": c.guidance
    }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=c.num_prompts_dataset,
        shuffle=True,
        drop_last=False,
        num_workers=c.num_workers,
        collate_fn=train_dataset.collate_fn
    )

    train_embedding_tmr = preload_tmr_text(train_dataloader)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=c.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=c.num_workers,
        collate_fn=val_dataset.collate_fn
    )

    val_embedding_tmr = preload_tmr_text(val_dataloader)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=c.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=c.num_workers,
        collate_fn=test_dataset.collate_fn
    )

    test_embedding_tmr = preload_tmr_text(val_dataloader)

    file_path = "ResultRL/VAL/"
    os.makedirs(file_path, exist_ok=True)

    optimizer = torch.optim.AdamW(diffusion_rl.parameters(), lr=c.lr, betas=(c.beta1, c.beta2), eps=c.eps,
                                  weight_decay=c.weight_decay)
    avg_reward, avg_tmr = test(diffusion_rl, val_dataloader, device, infos, text_model, smplh, joints_renderer,
                               smpl_renderer, c, val_embedding_tmr, path="ResultRL/VAL/OLD/")
    wandb.log({"Validation": {"Reward": avg_reward, "TMR": avg_tmr, "iterations": 0}})

    iter_bar = tqdm(range(c.iterations), desc="Iterations", total=c.iterations)
    for iteration in iter_bar:

        train_datasets_rl = generate(diffusion_rl, train_dataloader, iteration, c, device, infos, text_model, smplh,
                                     train_embedding_tmr)  # , generation_iter
        train(diffusion_rl, optimizer, train_datasets_rl, iteration, c, infos, device, old_model=diffusion_old)

        if (iteration + 1) % c.val_iter == 0:
            avg_reward, avg_tmr = test(diffusion_rl, val_dataloader, device, infos, text_model, smplh, joints_renderer,
                                       smpl_renderer, c, val_embedding_tmr,
                                       path="ResultRL/VAL/" + str(iteration + 1) + "/")
            wandb.log({"Validation": {"Reward": avg_reward, "TMR": avg_tmr, "iterations": iteration + 1}})
            torch.save(diffusion_rl.state_dict(), 'RL_Model/checkpoint_' + str(iteration + 1) + '.pth')
            iter_bar.set_postfix(val_tmr=f"{avg_tmr:.4f}")

    avg_reward, avg_tmr = test(diffusion_rl, test_dataloader, device, infos, text_model, smplh, joints_renderer,
                               smpl_renderer, c, test_embedding_tmr, path="ResultRL/TEST/")
    wandb.log({"Test": {"Reward": avg_reward, "TMR": avg_tmr}})

    torch.save(diffusion_rl.state_dict(), 'RL_Model/model_final.pth')


if __name__ == "__main__":
    main()
    wandb.finish()
