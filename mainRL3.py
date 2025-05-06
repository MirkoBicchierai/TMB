import itertools
import os
import shutil
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
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
from peft import LoraModel, LoraConfig
from new_motion_dataset.loader import MovementDataset, movement_collate_fn
import einops
from torch import Tensor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

tmr_forward = load_tmr_model_easy(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")


def render_swag(x_starts, infos, smplh, texts):
    trans = []
    for idx, (x_start, length) in enumerate(zip(x_starts, infos["all_lengths"])):

        if isinstance(length, torch.Tensor):
            length = int(length.item())

        x_start = x_start[:length]

        extracted_output = extract_joints(
            x_start.detach().cpu(),
            infos["featsname"],
            fps=infos["fps"],
            value_from="smpl",
            smpl_layer=smplh,
        )

        x, y, z = 0, 1, 2
        trajectory = extracted_output["joints"][:, 0, [x, y]]

        # Subtract the starting point (first value) from all points in the trajectory
        trajectory = trajectory - trajectory[0]

        # cmap = plt.get_cmap('coolwarm')
        #
        # # Normalize the data to [0, 1] range for coloring
        # norm = plt.Normalize(vmin=0, vmax=len(trajectory) - 1)
        #
        # # Plotting the trajectory with colors from red to blue
        # plt.figure(figsize=(8, 6))
        #
        # # Scatter plot for each point in the trajectory, coloring based on position
        # plt.scatter(trajectory[:, 0], trajectory[:, 1], c=np.arange(len(trajectory)), cmap=cmap, norm=norm, marker='o')
        #
        # # Add labels and title
        # plt.title('Trajectory in XY plane')
        # plt.xlabel('X coordinate')
        # plt.ylabel('Y coordinate')
        #
        # # Optional: Color bar to show the mapping of the points to the color scale
        # plt.colorbar(label='Index along trajectory')
        #
        # # Grid and legend
        # plt.grid(True)
        #
        # # Save the figure
        # plt.savefig(f"aua/sus_{idx}.png")

        trans.append(trajectory)

    return trans


def ungroup(features: Tensor) -> tuple[Tensor]:
    assert features.shape[-1] == 205
    (
        root_grav_axis,
        vel_trajectory_local,
        vel_angles,
        poses_local_flatten,
        joints_local_flatten,
    ) = einops.unpack(features, [[], [2], [], [132], [69]], "k *")

    poses_local = einops.rearrange(poses_local_flatten, "k (l t) -> k l t", t=6)
    joints_local = einops.rearrange(joints_local_flatten, "k (l t) -> k l t", t=3)
    return root_grav_axis, vel_trajectory_local, vel_angles, poses_local, joints_local


def fast_extract_pelvis_xy_batch(
        features: torch.Tensor,
        first_angle: float = np.pi
) -> torch.Tensor:
    """
    Extract pelvis (X,Y) for a batch of sequences.

    Args:
        features:     (B, T, 205) input feature tensor
        ungroup_fn:   function mapping (N,205) ->
                      (root_grav_axis[N], vel_traj_local[N,2], vel_angles[N], ...)
        first_angle:  initial yaw offset (scalar)

    Returns:
        traj_xy:      (B, T, 2) world‐space pelvis X,Y per frame
    """
    B, T, D = features.shape
    # 1) flatten batch/time to a single N = B*T
    feats_flat = features.reshape(-1, D)  # (B*T, 205)

    # 2) unpack only what we need
    root_grav_axis_flat, vel_traj_local_flat, vel_angles_flat, *_ = ungroup(feats_flat)
    # shapes: (B*T,), (B*T,2), (B*T,)

    # 3) reshape back to (B, T, ...)
    root_grav_axis = root_grav_axis_flat.view(B, T)  # (B, T)
    vel_traj_local = vel_traj_local_flat.view(B, T, 2)  # (B, T, 2)
    vel_angles = vel_angles_flat.view(B, T)  # (B, T)

    # 4) integrate yaw angles
    #    delta = vel_angles[:, :-1]; prepend zero so yaw[0]=first_angle
    delta = vel_angles[:, :-1]
    zeros = torch.zeros(B, 1, device=features.device, dtype=vel_angles.dtype)
    yaw = first_angle + torch.cat([zeros, delta.cumsum(dim=1)], dim=1)  # (B, T)

    # 5) rotate local XY velocities → world frame
    cos = torch.cos(yaw)  # (B, T)
    sin = torch.sin(yaw)  # (B, T)
    vx = cos * vel_traj_local[..., 0] - sin * vel_traj_local[..., 1]
    vy = sin * vel_traj_local[..., 0] + cos * vel_traj_local[..., 1]
    vel_world = torch.stack([vx, vy], dim=2)  # (B, T, 2)

    # 6) integrate to get world‐space XY trajectory
    start = torch.zeros(B, 1, 2, device=features.device, dtype=features.dtype)
    cumsum = vel_world[:, :-1, :].cumsum(dim=1)  # (B, T-1, 2)
    traj_xy = torch.cat([start, cumsum], dim=1)  # (B, T, 2)

    return traj_xy


def final_pelvis_points(traj_xy: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Args:
        traj_xy:  (B, T, 2) world-space pelvis XY per frame
        lengths:  (B,)    true lengths for each sequence (values in [1..T])
    Returns:
        final_pts: (B, 2) the pelvis XY at frame lengths[i]-1 for each batch i
    """
    # traj_xy = fast_extract_pelvis_xy_batch(sequences)
    B, T, _ = traj_xy.shape
    # ensure lengths are at least 1 and at most T
    last_idx = lengths.clamp(min=1, max=T) - 1  # (B,)
    batch_idx = torch.arange(B, device=traj_xy.device)
    final_pts = traj_xy[batch_idx, last_idx]  # (B, 2)
    return final_pts


def render(x_starts, infos, smplh, joints_renderer, smpl_renderer, texts, file_path, ty_log, video_log=False, p=None):
    out_formats = ['txt', 'smpl', 'videojoints']  # ['txt', 'smpl', 'joints', 'txt', 'smpl', 'videojoints', 'videosmpl']
    tmp = file_path

    for idx, (x_start, length, text) in enumerate(zip(x_starts, infos["all_lengths"], texts)):

        if isinstance(length, torch.Tensor):
            length = int(length.item())

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
            joints_renderer(extracted_output["joints"], title="", output=video_path, canonicalize=False, p=p[idx])
            if video_log:
                px, py = p[idx].detach().cpu().numpy()
                wandb.log({ty_log: {
                    "Video-joints": wandb.Video(video_path, format="mp4", caption=text + f"x: {px:.2f} y: {py:.2f}")}})

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


# def make_reference_path(steps, N):
#     """
#     steps = [(dir1, dist1), (dir2, dist2), …]
#        dir = 2-vector unit direction,
#        dist = length in metres
#     N = number of frames
#     """
#
#     # 1) build corner points
#     P = [np.zeros(2)]
#     for d,ℓ in steps:
#         P.append(P[-1] + d*ℓ)
#     # 2) linearly interpolate P to N samples
#     cumlen = np.cumsum([0] + [np.linalg.norm(P[i+1]-P[i]) for i in range(len(P)-1)])
#     u = np.linspace(0, cumlen[-1], N)
#     pts = []
#     for ui in u:
#         # find segment
#         k = np.searchsorted(cumlen, ui) - 1
#         t = (ui - cumlen[k]) / (cumlen[k+1] - cumlen[k])
#         pts.append((1-t)*P[k] + t*P[k+1])
#     return np.vstack(pts)  # shape (N,2)
#
# def replace_first_with_direction(tup):
#     direction_map = {
#         '0': np.array([0, 0.2]),
#         '1': np.array([0, -0.2]),
#         '2': np.array([-0.2, 0]),
#         '3': np.array([0.2, 0])
#     }
#     direction_tensor, second_value_tensor = tup
#     direction_key = str(direction_tensor.item())
#     direction_vector = direction_map.get(direction_key, np.array([0, 0]))  # Default to [0, 0]
#     return (direction_vector, second_value_tensor.cpu().numpy())
#
#
# def path_reward(Q, steps, α=5.0):
#
#     # Apply the function using map
#     steps = list(map(replace_first_with_direction, steps))
#
#     N = len(Q)
#     P = make_reference_path(steps, N)
#     errs = np.linalg.norm(Q - P, axis=1)
#     E_rms = np.sqrt(np.mean(errs**2))
#     return np.exp(-α*E_rms)

def position_reward(Q, P):
    errs = torch.mean((Q - P.detach()) ** 2).cpu().numpy()

    return -np.sqrt(errs)


def compute_reach_reward(
        traj_xy: torch.Tensor,
        lengths: torch.Tensor,
        target_xy: torch.Tensor,
        thresh: float = 0.1,
        bonus: float = 1.0,
        w_shaping: float = 0.5,
) -> torch.Tensor:
    """
    Compute reward for a batch of trajectories.

    Args:
        traj_xy:    (B, T, 2) pelvis XY per frame
        lengths:    (B,) true sequence‐lengths
        target_xy:  (B, 2) goal positions
        thresh:     distance threshold for success bonus
        bonus:      sparse reward if final point is within thresh
        w_shaping:  weight on the potential‐based shaping term

    Returns:
        reward:     (B,) total reward per trajectory
    """
    B, T, _ = traj_xy.shape

    # 1) Sparse final‐step bonus
    final_xy = final_pelvis_points(traj_xy, lengths)  # (B,2)
    d_final = torch.norm(final_xy - target_xy, dim=1)  # (B,)
    sparse = (d_final < thresh).float() * bonus  # (B,)

    # 2) Potential‐based shaping: sum over t of [dist(t−1)−dist(t)]
    #    dist: (B, T) distances at each frame
    dist = torch.norm(traj_xy - target_xy.unsqueeze(1), dim=2)
    #    ∆ = dist[:, t−1] − dist[:, t], for t=1..T−1
    delta = dist[:, :-1] - dist[:, 1:]  # (B, T−1)
    shaping = w_shaping * delta.sum(dim=1)  # (B,)

    # 3) Total reward
    reward = sparse + shaping
    return reward


def tmr_reward_special(sequences, infos, smplh, texts, all_embedding_tmr, c):
    motions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = int(infos["all_lengths"][idx].item())

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
            row_copy[row_copy > c.masking_ratio] = np.nan

            """ 
            num_elements = len(row_copy)
            num_to_nan = int(num_elements * c.masking_ratio)
            indices_to_nan = np.random.choice(num_elements, num_to_nan, replace=False)
            row_copy[indices_to_nan] = np.nan
            """

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
        "positions": [],

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

        if not c.sequence_fixed:
            infos["all_lengths"] = batch["length"].repeat(c.num_gen_per_prompt)

        batch["positions"] = batch["positions"].repeat(c.num_gen_per_prompt, 1)

        sequences, results_by_timestep = model.diffusionRL(tx_emb=tx_emb, tx_emb_uncond=tx_emb_uncond, infos=infos,
                                                           p=batch["positions"])
        #
        # _, tmr = tmr_reward_special(sequences, infos, smplh, batch["tmr_text"].repeat(c.num_gen_per_prompt, 1),
        #                                  train_embedding_tmr, c)

        # Q = render_swag(sequences, infos, smplh, batch["text"])
        # lesghere = []
        #
        # # batch["distances"] = batch["distances"].repeat(c.num_gen_per_prompt,1)
        # for stronzo in range(len(Q)):
        #     r = position_reward(Q[stronzo][-1], batch["positions"])
        #     lesghere.append(r)
        #
        # alpha = 0.00
        # reward = alpha * reward + torch.Tensor(lesghere, device=reward.device) * 10

        Q = fast_extract_pelvis_xy_batch(sequences)
        reward = compute_reach_reward(Q, infos["all_lengths"].long(), batch["positions"]) # + tmr.to(device)

        tmr = torch.zeros_like(reward)

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
        all_positions = []

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
            all_positions.append(experiment["positions"])

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
        dataset["positions"].append(torch.cat(all_positions, dim=0).view(diff_step, batch_size, 2).permute(1, 0, 2))

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
    positions = dataset["positions"][i: i + minibatch_size]

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

    return y, r, xt_1, xt, t, log_like, positions


def prepare_dataset(dataset):
    dataset_size = dataset["r"].shape[0]
    shuffle_indices = torch.randperm(dataset_size)

    for key in dataset:
        dataset[key] = dataset[key][shuffle_indices]

    return dataset


def train(model, optimizer, dataset, iteration, c, infos, device, old_model=None):
    model.train()
    delta = 1e-5
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
            # with torch.autocast(device_type="cuda"):
            advantage = dataset["advantage"][batch_idx: batch_idx + c.train_batch_size].to(device)
            real_batch_size = advantage.shape[0]
            y, r, xt_1, xt, t, log_like, positions = get_batch(dataset, batch_idx, real_batch_size, infos, diff_step,
                                                               device)

            new_log_like, rl_pred = model.diffusionRL(y=y, infos=infos, t=t.view(diff_step * real_batch_size),
                                                      xt=xt.view(diff_step * real_batch_size, *xt.shape[2:]),
                                                      A=xt_1.view(diff_step * real_batch_size, *xt_1.shape[2:]),
                                                      p=positions.view(diff_step * real_batch_size,
                                                                       *positions.shape[2:]).to(device))

            new_log_like = new_log_like.view(real_batch_size, diff_step)
            rl_pred = rl_pred.view(real_batch_size, diff_step, *rl_pred.shape[1:])

            if c.betaL > 0:
                _, old_pred = old_model.diffusionRL(y=y, infos=infos, t=t.view(diff_step * real_batch_size),
                                                    xt=xt.view(diff_step * real_batch_size, *xt.shape[2:]),
                                                    A=xt_1.view(diff_step * real_batch_size, *xt_1.shape[2:]))
                old_pred = old_pred.view(real_batch_size, diff_step, *old_pred.shape[1:])
                kl_div = ((rl_pred - old_pred) ** 2).sum(1).mean()

            ratio = torch.exp(new_log_like - log_like)
            # torch.set_printoptions(precision=4)

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

    ty_log = ""

    if "VAL" in path:
        ty_log = "Validation"
    else:
        ty_log = "Test"

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

            if not c.sequence_fixed:
                infos["all_lengths"] = batch["length"]

            sequences, _ = model.diffusionRL(tx_emb, tx_emb_uncond, infos, p=batch["positions"])

            if (ty_log == "Validation" and batch_idx == 0) or ty_log == "Test":
                render(sequences, infos, smplh, joints_renderer, smpl_renderer, batch["text"], tmp_path, ty_log,
                       video_log=False, p=batch["positions"])
            #
            _, tmr = tmr_reward_special(sequences, infos, smplh, batch["tmr_text"], all_embedding_tmr,
                                             c)  # shape [batch_size]

            # Q_a = render_swag(sequences, infos, smplh, batch["text"])

            Q = fast_extract_pelvis_xy_batch(sequences)
            reward = compute_reach_reward(Q, infos["all_lengths"].long(), batch["positions"])

            # alpha = 0.00
            # reward = alpha * reward + torch.Tensor(lesghere, device=reward.device) * 10

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


@hydra.main(config_path="configs", config_name="TrainRL", version_base="1.3")
def main(c: DictConfig):
    config_dict = OmegaConf.to_container(c, resolve=True)

    wandb.init(
        project="TM-BM",
        name=c.experiment_name,
        config=config_dict,
        group=c.group_name
    )

    create_folder_results("ResultRL")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = read_config(c.run_dir)

    ckpt_path = os.path.join(c.run_dir, c.ckpt_name)
    print("Loading the checkpoint")
    ckpt = torch.load(str(ckpt_path), map_location=device)

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

    cfg.diffusion.denoiser._target_ = "src.model.mdm_smpl_controllable.TransformerDenoiserControllable"
    diffusion_rl = instantiate(cfg.diffusion)

    # ckpt = torch.load(ckpt["state_dict"])
    ckpt_sd = ckpt["state_dict"]

    # 1) Filter out all keys under position_mlp
    filtered_ckpt_sd = {
        k: v
        for k, v in ckpt_sd.items()
        if not k.startswith("position_mlp.")
    }

    # 2) Grab your model’s own state dict…
    model_sd = diffusion_rl.state_dict()

    # 3) Update it with the filtered checkpoint entries…
    model_sd.update(filtered_ckpt_sd)

    # 4) Load into your model (now you don’t need strict=False because you
    #    have a complete dict for all params your model actually has)
    diffusion_rl.load_state_dict(model_sd)

    # diffusion_rl.load_state_dict(ckpt["state_dict"])
    diffusion_rl = diffusion_rl.to(device)

    if c.freeze_normalization_layers:
        freeze_normalization_layers(diffusion_rl)

    if c.lora:
        """LORA"""
        lora_config = LoraConfig(
            r=c.lora_rank,
            lora_alpha=c.lora_alpha,
            target_modules=[
                "to_skel_layer",
                "skel_embedding",
                "tx_embedding.0",
                "tx_embedding.2",
                "seqTransEncoder.layers.0.self_attn.out_proj",
                "seqTransEncoder.layers.1.self_attn.out_proj",
                "seqTransEncoder.layers.2.self_attn.out_proj",
                "seqTransEncoder.layers.3.self_attn.out_proj",
                "seqTransEncoder.layers.4.self_attn.out_proj",
                "seqTransEncoder.layers.5.self_attn.out_proj",
                "seqTransEncoder.layers.6.self_attn.out_proj",
                "seqTransEncoder.layers.7.self_attn.out_proj",

            ],
            lora_dropout=c.lora_dropout,
            bias=c.lora_bias,
        )

        # Freeze all parameters except LoRA
        for name, param in diffusion_rl.denoiser.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

        diffusion_rl.denoiser = LoraModel(diffusion_rl.denoiser, lora_config, "sus")

        # un-freeze the MLP
        for p in diffusion_rl.denoiser.position_mlp.parameters():
            p.requires_grad = True

        # Check trainable parameters
        trainable_params = [name for name, param in diffusion_rl.denoiser.named_parameters() if param.requires_grad]
        print("Trainable LorA layer:", trainable_params)
        total_trainable_params = sum(p.numel() for p in diffusion_rl.denoiser.parameters() if p.requires_grad)
        print(f"Trainable parameters LorA: {total_trainable_params:,}")

        """END LORA"""

    if c.betaL > 0:
        diffusion_old = instantiate(cfg.diffusion)
        diffusion_old.load_state_dict(ckpt["state_dict"])
        diffusion_old = diffusion_old.to(device)
        diffusion_old.eval()
    else:
        diffusion_old = None

    if False:
        train_dataset = instantiate(cfg.data, split=str(c.dataset_name) + "train")
        val_dataset = instantiate(cfg.data, split=str(c.dataset_name) + "val")
        test_dataset = instantiate(cfg.data, split=str(c.dataset_name) + "test")

    infos = {
        "featsname": cfg.motion_features,
        "fps": c.fps,
        "guidance_weight": c.guidance_weight
    }

    if c.sequence_fixed:
        infos["all_lengths"] = torch.tensor(np.full(2048, int(c.time * c.fps))).to(device)

    if False:
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

        test_embedding_tmr = preload_tmr_text(test_dataloader)

    else:
        train_dataset = MovementDataset('/home/mbicchierai/Tesi Magistrale/new_motion_dataset/data/pos_train.json')
        train_dataloader = DataLoader(train_dataset, batch_size=c.num_prompts_dataset, shuffle=True, drop_last=False,
                                      num_workers=c.num_workers, collate_fn=movement_collate_fn)
        train_embedding_tmr = None

        val_dataset = MovementDataset('/home/mbicchierai/Tesi Magistrale/new_motion_dataset/data/pos_val.json')
        val_dataloader = DataLoader(val_dataset, batch_size=c.val_batch_size, shuffle=False, drop_last=False,
                                    num_workers=c.num_workers, collate_fn=movement_collate_fn)
        val_embedding_tmr = None

        test_dataset = MovementDataset('/home/mbicchierai/Tesi Magistrale/new_motion_dataset/data/pos_test.json')
        test_dataloader = DataLoader(test_dataset, batch_size=c.val_batch_size, shuffle=False, drop_last=False,
                                     num_workers=c.num_workers, collate_fn=movement_collate_fn)
        test_embedding_tmr = None

    file_path = "ResultRL/VAL/"
    os.makedirs(file_path, exist_ok=True)

    if c.lora:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, diffusion_rl.denoiser.parameters()), lr=c.lr,
                                      betas=(c.beta1, c.beta2), eps=c.eps,
                                      weight_decay=c.weight_decay)
    else:
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
