import itertools
import os
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from RL.reward_model import tmr_reward_special, guo_reward
from src.tools.smpl_layer import SMPLH
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import read_config
from src.model.text_encoder import TextToEmb
from peft import LoraModel, LoraConfig
from RL.utils import render, get_embeddings, freeze_normalization_layers, create_folder_results


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def preload_tmr_text(dataloader):
    all_embeddings = []
    for batch_idx, batch in enumerate(dataloader):
        all_embeddings.append(batch["tmr_text"])
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def test(model, dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer, c, all_embedding_tmr, path):
    os.makedirs(path, exist_ok=True)
    out_formats = ['txt', 'smpl', 'joints', 'videojoints']

    ty_log = ""
    if "VAL" in path:
        ty_log = "Validation"
    else:
        ty_log = "Test"

    model.eval()

    if c.val_num_batch == 0:
        generate_bar = tqdm(enumerate(dataloader), leave=False, desc=f"[Validation/Test Generations]", total=len(dataloader))
    else:
        generate_bar = tqdm(enumerate(itertools.islice(itertools.cycle(dataloader), c.val_num_batch)),
                            total=c.val_num_batch, leave=False, desc=f"[Validation/Test Generations]")

    total_reward, total_tmr, total_tmr_plus_plus, total_guo = 0, 0, 0, 0
    batch_count_reward, batch_count_tmr, batch_count_tmr_plus_plus = 0, 0, 0

    for batch_idx, batch in generate_bar:
        tmp_path = path + "batch_" + str(batch_idx) + "/"
        os.makedirs(tmp_path, exist_ok=True)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)

            if not c.sequence_fixed:
                infos["all_lengths"] = batch["length"]

            sequences, _ = model.diffusionRL(tx_emb, tx_emb_uncond, infos)

            if False:
               render(sequences, infos, smplh, joints_renderer, smpl_renderer, batch["text"], tmp_path, ty_log, out_formats, video_log=True)

            metrics = tmr_reward_special(sequences, infos, smplh, batch["text"], all_embedding_tmr, c)  # shape [batch_size]
            metrics_guo = guo_reward(sequences, infos, smplh, batch["text"], all_embedding_tmr, c)

            total_reward += metrics["reward"].sum().item()
            batch_count_reward += metrics["reward"].shape[0]

            total_tmr += metrics["tmr"].sum().item()
            batch_count_tmr += metrics["tmr"].shape[0]

            total_tmr_plus_plus += metrics["tmr++"].sum().item()
            batch_count_tmr_plus_plus += metrics["tmr++"].shape[0]

            total_guo += metrics_guo["guo"].sum().item()

    avg_reward = total_reward / batch_count_reward
    avg_tmr = total_tmr / batch_count_tmr
    avg_tmr_plus_plus = total_tmr_plus_plus / batch_count_tmr_plus_plus
    total_guo = total_guo / batch_count_tmr

    return avg_reward, avg_tmr, avg_tmr_plus_plus, total_guo

@hydra.main(config_path="configs", config_name="TrainRL", version_base="1.3")
def main(c: DictConfig):
    config_dict = OmegaConf.to_container(c, resolve=True)

    create_folder_results("ResultRL")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = read_config(c.run_dir)

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

    # lora_config = LoraConfig(
    #     r=c.lora_rank,
    #     lora_alpha=c.lora_alpha,
    #     target_modules=[
    #         "to_skel_layer",
    #         "skel_embedding",
    #         "tx_embedding.0",
    #         "tx_embedding.2",
    #         "seqTransEncoder.layers.0.self_attn.out_proj",
    #         "seqTransEncoder.layers.1.self_attn.out_proj",
    #         "seqTransEncoder.layers.2.self_attn.out_proj",
    #         "seqTransEncoder.layers.3.self_attn.out_proj",
    #         "seqTransEncoder.layers.4.self_attn.out_proj",
    #         "seqTransEncoder.layers.5.self_attn.out_proj",
    #         "seqTransEncoder.layers.6.self_attn.out_proj",
    #         "seqTransEncoder.layers.7.self_attn.out_proj",
    #     ],
    #     lora_dropout=c.lora_dropout,
    #     bias=c.lora_bias,
    # )

    # Apply LoRA configuration to the model first
    # diffusion_rl.denoiser = LoraModel(diffusion_rl.denoiser, lora_config, "sus")
    # diffusion_rl.load_state_dict(torch.load('/home/mbicchierai/Tesi Magistrale/RL_Model/checkpoint_2500.pth'))

    ckpt = torch.load("/home/mbicchierai/Tesi Magistrale/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d/logs/checkpoints/last.ckpt", map_location="cuda")
    diffusion_rl.load_state_dict(ckpt["state_dict"])

    diffusion_rl = diffusion_rl.to(device)

    val_dataset = instantiate(cfg.data, split=str(c.dataset_name) +"val")
    test_dataset = instantiate(cfg.data, split=str(c.dataset_name) +"test")

    infos = {
        "featsname": cfg.motion_features,
        "fps": c.fps,
        "guidance_weight":  c.guidance_weight
    }

    if c.sequence_fixed:
        infos["all_lengths"] = torch.tensor(np.full(2048, int(c.time * c.fps))).to(device)

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

    file_path = "ResultRL/TEST/"
    os.makedirs(file_path, exist_ok=True)

    file_path = "ResultRL/VAL/"
    os.makedirs(file_path, exist_ok=True)

    avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo = test(diffusion_rl, val_dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer, c, val_embedding_tmr, path="ResultRL/VAL/OLD/")
    print("Validation-reward:", avg_reward,"Validation-tmr:", avg_tmr, "Validation-tmr++:", avg_tmr_plus_plus, "Validation-guo:", avg_guo)

    avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo = test(diffusion_rl, test_dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer, c, test_embedding_tmr, path="ResultRL/TEST/")
    print("Test-reward:", avg_reward, "Test-tmr:", avg_tmr, "Test-tmr++:", avg_tmr_plus_plus, "Test-guo:", avg_guo)

    # GT val (short) [tmr:0.897 tmr++: 0.892]
    # GT test (short) [tmr: 0.836 tmr++: 0.837]
    # Validation set (short) | guidance 7 -- OLD: [tmr: 0.841 tmr++: 0.849] NEW: [] checkpoint_2500
    #                        | guidance 1 -- OLD: [tmr: 0.772 tmr++: 0.780] NEW: [tmr: 0.827 tmr++: 0.841] checkpoint_2500
    # Test set (short)       | guidance 7 -- OLD: [tmr: 0.801 tmr++: 0.812] NEW: [] checkpoint_2500
    #                        | guidance 1 -- OLD: [tmr: 0.736 tmr++: 0.750] NEW: [tmr: 0.793 Test-tmr++: 0.810] checkpoint_2500

if __name__ == "__main__":
    main()
