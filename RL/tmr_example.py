import numpy as np
import torch
import re

from colorama import Fore, Style, init
from src.tools.smpl_layer import SMPLH
from src.tools.extract_joints import extract_joints
from src.tools.guofeats.motion_representation import joints_to_guofeats, guofeats_to_joints

from TMR.mtt.load_tmr_model import load_tmr_model_easy
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.model.tmr import get_sim_matrix


def is_list_of_strings(var):
    return isinstance(var, list) and all(isinstance(item, str) for item in var)


def print_matrix_nicely(matrix: np.ndarray):
    """
    Stampa una matrice 2D con valori troncati a 3 decimali e colora in verde
    il massimo per ogni riga.

    Args:
        matrix (np.ndarray): Matrice 2D di float.
    """
    init(autoreset=True)  # per ripristinare i colori automaticamente

    if len(matrix.shape) != 2:
        raise ValueError("La matrice deve essere 2D")

    for row in matrix:
        max_val = np.max(row)
        line = ""
        for val in row:
            # Troncamento a 3 decimali (non arrotondamento)
            truncated = int(val * 1000) / 1000
            formatted = f"{truncated:.3f}"

            # Colore verde se massimo della riga
            if val == max_val:
                line += f"{Fore.GREEN}{formatted}{Style.RESET_ALL}  "
            else:
                line += f"{formatted}  "
        print(line)


def count_diagonal_max(matrix):
    matrix = np.array(matrix)  # Converti in array NumPy se non lo è già
    count = 0
    for i in range(len(matrix)):
        if matrix[i, i] == np.max(matrix[i]):  # Controlla se l'elemento diagonale è il massimo della riga
            count += 1
    return count


def smpl_to_guofeats(smpl):
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
        # convert to guofeats
        # first, make sure to revert the axis
        # as guofeats have gravity axis in Y
        x, y, z = i_joints.T
        i_joints = np.stack((x, z, -y), axis=0).T
        i_guofeats = joints_to_guofeats(i_joints)
        i_joints = guofeats_to_joints(torch.tensor(i_guofeats))
        # joints_renderer(i_joints_.numpy(), title="", output= "/andromeda/personal/lmandelli/MotionDiffusionBase/joint_pose_2.mp4", canonicalize=False)
        guofeats.append(i_guofeats)

    return guofeats

tmr_forward = load_tmr_model_easy(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")  # humanml3d OR humanml3d_kitml_augmented_and_hn OR tmr_humanml3d_kitml_guoh3dfeats

smplh = SMPLH(
    path="deps/smplh",
    jointstype='both',
    input_pose_rep="axisangle",
    gender='male',
)

def calc_eval_stats(X, Y):
    """
        Calculate Motion2Motion (m2m) and the Motion2Text (m2t) between the recostructed motion, the gt motion and the gt text.
    """
    if is_list_of_strings(X):
        X_latents = tmr_forward(X)  # tensor(N, 256)
    else:
        X_guofeats = smpl_to_guofeats(X)
        X_latents = tmr_forward(X_guofeats)  # tensor(N, 256)
    if is_list_of_strings(Y):
        Y_latents = tmr_forward(Y)  # tensor(N, 256)
    else:
        Y_guofeats = smpl_to_guofeats(Y)
        Y_latents = tmr_forward(Y_guofeats)

    sim_matrix = get_sim_matrix(X_latents, Y_latents).numpy()
    return sim_matrix


motion_paths = [
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/0/0_smpl.npy",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/1/1_smpl.npy",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/2/2_smpl.npy",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/3/3_smpl.npy",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/4/4_smpl.npy",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/5/5_smpl.npy",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/6/6_smpl.npy",
]

text_paths = [
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/0/0.txt",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/1/1.txt",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/2/2.txt",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/3/3.txt",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/4/4.txt",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/5/5.txt",
    "/home/mbicchierai/Tesi Magistrale/ResultRL/VAL/OLD/batch_0/6/6.txt",
]

motions = []
for i in motion_paths:
    motions.append(torch.tensor(np.load(i)))

texts = []
for i in text_paths:
    with open(i, "r") as f:
        content = f.read()
        #match = re.search(r"Motion:\s*-\s*(.*?)- duration:", content, re.DOTALL)
        #if match:
        #    content = match.group(1).strip()
        texts.append(content)

with torch.no_grad():
    simm_matrix = calc_eval_stats(motions, texts)
    print(f"\nSimmetria motions - texts (score: {count_diagonal_max(simm_matrix)})")
    print_matrix_nicely(simm_matrix)

    simm_matrix = calc_eval_stats(motions, motions)
    print(f"\nSimmetria motions - motions (score: {count_diagonal_max(simm_matrix)})")
    print_matrix_nicely(simm_matrix)

    simm_matrix = calc_eval_stats(texts, texts)
    print(f"\nSimmetria texts - texts (score: {count_diagonal_max(simm_matrix)})")
    print_matrix_nicely(simm_matrix)