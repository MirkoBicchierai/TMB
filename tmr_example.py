import numpy as np
import torch

from src.tools.smpl_layer import SMPLH
from src.tools.extract_joints import extract_joints

from TMR.mtt.load_tmr_model import load_tmr_model_easy
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.model.tmr import get_sim_matrix, get_score_matrix
from TMR.src.data.motion import Normalizer
# carico  il modello TMR
tmr_forward = load_tmr_model_easy(device="cpu", dataset="humanml3d")

smplh = SMPLH(
    path="deps/smplh",
    jointstype='both',
    input_pose_rep="axisangle",
    gender='male',
)

def calc_eval_stats(x, texts):
    """
    Calculate Motion2Motion (m2m) and the Motion2Text (m2t) between the recostructed motion, the gt motion and the gt text.
    """

    text_latents_gt = tmr_forward(texts)[0] #  tensor(N, 256)
    normalizer = Normalizer(base_dir="/home/mirko/PycharmProjects/Tesi Magistrale/TMR/stats/humanml3d/guoh3dfeats")


    x_rec_output = extract_joints(
        x,
        'smplrifke',
        fps=20,
        value_from='smpl',
        smpl_layer=smplh,
    )
    x_rec_joints = x_rec_output["joints"]
    x_rec_guofeats = joints_to_guofeats(x_rec_joints)
    motion = normalizer(torch.tensor(x_rec_guofeats))
    motion_x_dict = {"x": motion, "length": len(motion)}
    motion_latents = tmr_forward([np.array(motion)])[0]  # tensor(N, 256)

    score = get_score_matrix(text_latents_gt, motion_latents).cpu()

    return score


x = np.load("/home/mirko/PycharmProjects/Tesi Magistrale/ResultRL/1/1_smpl.npy")
x = torch.tensor(x)

texts = [
    ["a man claps his hands a few times then returns his arms to his sides."],
    ["a person walks up stairs."],
    ["a person sitting down in place."]
]

for t in texts:
    m2t = calc_eval_stats(x, t)
    print(f"score : {m2t}, text {t}")
