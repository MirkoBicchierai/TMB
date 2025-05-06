import argparse
import os
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from mainRL3 import fast_extract_pelvis_xy_batch, compute_reach_reward, render, tmr_reward_special
from new_motion_dataset.loader import MovementDataset, movement_collate_fn
from src.tools.smpl_layer import SMPLH
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import read_config
from src.tools.extract_joints import extract_joints
from src.model.text_encoder import TextToEmb
from colorama import Fore, Style, init
from TMR.mtt.load_tmr_model import load_tmr_model_easy
from src.tools.guofeats.motion_representation import joints_to_guofeats
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.model.tmr import get_sim_matrix
from peft import LoraModel, LoraConfig
import pytorch_lightning as pl

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

tmr_forward = load_tmr_model_easy(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")


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
        # convert to guofeats, first, make sure to revert the axis, as guofeats have gravity axis in Y\
        x, y, z = i_joints.T
        i_joints_flipped = np.stack((x, z, -y), axis=0).T
        i_guofeats = joints_to_guofeats(i_joints_flipped)
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


def test(model, dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer, c, all_embedding_tmr,
         path):
    os.makedirs(path, exist_ok=True)

    model.eval()

    generate_bar = tqdm(enumerate(dataloader), leave=False, desc=f"[Validation/Test Generations]")

    total_reward, total_tmr = 0, 0
    batch_count_reward, batch_count_tmr = 0, 0

    for batch_idx, batch in generate_bar:
        tmp_path = path + "batch_" + str(batch_idx) + "/"
        os.makedirs(tmp_path, exist_ok=True)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)


            infos["all_lengths"] = batch["length"]

            sequences, _ = model.diffusionRL(tx_emb, tx_emb_uncond, infos, p=batch["positions"])


            _, tmr = tmr_reward_special(sequences, infos, smplh, batch["tmr_text"], all_embedding_tmr,
                                        c)

            render(sequences, infos, smplh, joints_renderer, smpl_renderer, batch["text"], tmp_path, "",
                   video_log=False, p=batch["positions"], tmr=tmr)
            #
  # shape [batch_size]

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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training configuration parser")

    # sequence parameters
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--time", type=float, default=2.5, help="Duration in seconds")
    parser.add_argument("--joint_stype", type=str, default="both", choices=["both", "smpljoints"],
                        help="Joint style type")

    # Validation/Test parameters
    parser.add_argument("--val_iter", type=int, default=25, help="Validation iterations")
    parser.add_argument("--val_batch_size", type=int, default=256, help="Validation batch size")
    parser.add_argument("--val_num_batch", type=int, default=1,
                        help="Validation number of batch used (0 for all batch)")

    # pretrained model parameters
    parser.add_argument("--run_dir", type=str, default='pretrained_models/mdm-smpl_clip_smplrifke_humanml3d',
                        help="Run directory")

    parser.add_argument("--ckpt_name", type=str, default='logs/checkpoints/last.ckpt', help="Checkpoint file name")

    return parser.parse_args()


@hydra.main(config_path="configs", config_name="TrainRL", version_base="1.3")
def main(c: DictConfig):
    args = parse_arguments()

    cfg = read_config(args.run_dir)
    pl.seed_everything(1534)
    cfg.diffusion.denoiser._target_ = "src.model.mdm_smpl_controllable.TransformerDenoiserControllable"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    joints_renderer = instantiate(c.joints_renderer)
    smpl_renderer = instantiate(c.smpl_renderer)

    text_model = TextToEmb(
        modelpath=cfg.data.text_encoder.modelname, mean_pooling=cfg.data.text_encoder.mean_pooling, device=device
    )

    smplh = SMPLH(
        path="deps/smplh",
        jointstype=args.joint_stype,
        input_pose_rep="axisangle",
        gender=c.gender,
    )

    print("Loading the models")
    normalizer_dir = "pretrained_models/mdm-smpl_clip_smplrifke_humanml3d" if cfg.dataset == "humanml3d" else "pretrained_models/mdm-smpl_clip_smplrifke_kitml"
    cfg.diffusion.motion_normalizer.base_dir = os.path.join(normalizer_dir, "motion_stats")
    cfg.diffusion.text_normalizer.base_dir = os.path.join(normalizer_dir, "text_stats")

    diffusion_rl = instantiate(cfg.diffusion)

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

    # Apply LoRA configuration to the model first
    diffusion_rl.denoiser = LoraModel(diffusion_rl.denoiser, lora_config, "sus")
    diffusion_rl.load_state_dict(torch.load('/home/mbicchierai/Tesi Magistrale/RL_Model/better_no_tmr.pth'))

    diffusion_rl = diffusion_rl.to(device)

    val_dataset = MovementDataset('/home/mbicchierai/Tesi Magistrale/new_motion_dataset/data/pos_val.json')
    val_dataloader = DataLoader(val_dataset, batch_size=c.val_batch_size, shuffle=False, drop_last=False,
                                num_workers=c.num_workers, collate_fn=movement_collate_fn)

    infos = {
        # "all_lengths": torch.tensor(np.full(2048, int(args.time * args.fps))).to(device),
        "featsname": cfg.motion_features,
        "fps": args.fps,
        "guidance_weight": 1.0
    }

    avg_reward, avg_tmr = test(diffusion_rl, val_dataloader, device, infos, text_model, smplh, joints_renderer,
                               smpl_renderer, c, None, path="ResultRL/TEST_RL_MODEL/")
    print(avg_reward, avg_tmr)

    # tmr_forward = load_tmr_model_easy(device="cpu", dataset="humanml3d")
    # GUIDANCE 7: OLD-0.801, NEW- 0.786
    # GUIDANCE 1: OLD-0.737 , NEW- 0.7187

    # tmr_forward = load_tmr_model_easy(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")
    # GUIDANCE 7: OLD - 0.753, NEW - 0.743
    # GUIDANCE 1: OLD - 0.695, NEW - 0.732


if __name__ == "__main__":
    main()
