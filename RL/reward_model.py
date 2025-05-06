import numpy as np
import torch
from colorama import Fore, Style, init
from src.tools.guofeats.motion_representation import joints_to_guofeats
from src.tools.extract_joints import extract_joints
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.model.tmr import get_sim_matrix
from TMR.mtt.load_tmr_model import load_tmr_model_easy
from TM2T.load_tm2t_model import load_tm2t_model_easy

tmr_forward_plus_plus = load_tmr_model_easy(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")
tmr_forward = load_tmr_model_easy(device="cpu", dataset="humanml3d")
guo_forward = load_tm2t_model_easy(device="cpu", dataset="humanml3d") # humanml3d OR humanml3d_kitml_augmented_and_hn OR tmr_humanml3d_kitml_guoh3dfeats


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


def calc_eval_stats(x, smplh, forward):
    x_guofeats = smpl_to_guofeats(x, smplh)
    x_latents = forward(x_guofeats)# tensor(N, 256)
    return x_latents


def is_list_of_strings(var):
    return isinstance(var, list) and all(isinstance(item, str) for item in var)


def print_matrix_nicely(matrix: np.ndarray, mmax=True):
    init(autoreset=True)
    for row in matrix:
        if mmax:
            max_val = np.max(row)
        else:
            max_val = np.min(row)
        line = ""
        for val in row:
            truncated = int(val * 1000) / 1000
            formatted = f"{truncated:.3f}"
            if val == max_val:
                line += f"{Fore.GREEN}{formatted}{Style.RESET_ALL}  "
            else:
                line += f"{formatted}  "
        print(line)


def tmr_reward_special(sequences, infos, smplh, real_texts, all_embedding_tmr, c):

    texts_plus_plus = tmr_forward_plus_plus(real_texts)
    texts = tmr_forward(real_texts)

    motions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]
        motions.append(x_start.detach().cpu())

    x_latents = calc_eval_stats(motions, smplh, tmr_forward)
    x_latents_plus_plus = calc_eval_stats(motions, smplh, tmr_forward_plus_plus)

    sim_matrix = get_sim_matrix(x_latents, texts.detach().cpu().type(x_latents.dtype)).numpy()
    # print_matrix_nicely(sim_matrix)

    sim_matrix_plus_plus = get_sim_matrix(x_latents_plus_plus, texts_plus_plus.detach().cpu().type(x_latents.dtype)).numpy()
    # print_matrix_nicely(sim_matrix)

    sim_matrix = torch.tensor(sim_matrix)
    sim_matrix = (sim_matrix + 1) / 2
    tmr = sim_matrix.diagonal()

    sim_matrix_plus_plus = torch.tensor(sim_matrix_plus_plus)
    sim_matrix_plus_plus = (sim_matrix_plus_plus + 1) / 2
    tmr_plus_plus = sim_matrix_plus_plus.diagonal()

    if c.tmr_reward:

        metrics = {
            "tmr": tmr,
            "tmr++": tmr_plus_plus,
            "reward": tmr_plus_plus * c.reward_scale if c.tmr_plus_plus else tmr * c.reward_scale
        }

        return metrics
    else:

        sim_matrix_tmp = get_sim_matrix(x_latents, all_embedding_tmr.detach().cpu().type(x_latents.dtype)).numpy()
        # print_matrix_nicely(sim_matrix_tmp)

        sim_matrix_tmp = (sim_matrix_tmp + 1) / 2
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

            # Calculate mean without the excluded element
            row_mean = np.nanmean(row_copy)
            # Calculate special value for this row (real - mean of row of all emb)
            special_value = diagonal_values[i] - row_mean
            special.append(special_value)

        special = torch.tensor(special)

    return special * c.reward_scale


def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def guo_reward(sequences, infos, smplh, real_texts, all_embedding_tmr, c):

    motions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]
        motions.append(x_start.detach().cpu())

    motions_guofeats = smpl_to_guofeats(motions, smplh=smplh)
    motions_latents, texts_latents = guo_forward(motions=motions_guofeats, texts=real_texts)

    sim_matrix = euclidean_distance_matrix(motions_latents.cpu().numpy(), texts_latents.cpu().numpy())
    # print_matrix_nicely(sim_matrix, mmax=False)

    sim_matrix = torch.tensor(sim_matrix)
    # Normalization (not needed but I wanted):
    # the Trace of the matrix is between [0, inf], so I multiply it by (-1) and add 1, Now is between [-inf, 1]. I divide by 10 for better visualization.
    guo = (sim_matrix.diagonal() * (-1) + 1) / 10 

    metrics = {
        "guo": guo,
        "reward": guo * c.reward_scale
    }

    return metrics


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
