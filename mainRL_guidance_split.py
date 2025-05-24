import itertools
import os
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from RL.reward_model import all_metrics, reward_model
from src.tools.smpl_layer import SMPLH
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import read_config
from src.model.text_encoder import TextToEmb
import wandb
from peft import LoraModel, LoraConfig
from RL.utils import render, get_embeddings, get_embeddings_2, freeze_normalization_layers, create_folder_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def preload_tmr_text(dataloader):
    all_embeddings = []
    for batch_idx, batch in enumerate(dataloader):
        all_embeddings.append(batch["tmr_text"])
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


@torch.no_grad()
def generate(model, train_dataloader, iteration, c, device, infos, text_model, smplh,
             train_embedding_tmr, target_model):  # , generation_iter
    model.train()

    dataset = {

        "r": [],
        "xt_1": [],
        "xt": [],
        "t": [],
        "log_like": [],

        "mask": [],
        "length": [],
        "tx_x": [],
        "tx_mask": [],
        "tx_length": [],
        "tx_uncond_x": [],
        "tx_uncond_mask": [],
        "tx_uncond_length": [],
        "tx_uncond_output": [],

    }

    generate_bar = tqdm(enumerate(itertools.islice(itertools.cycle(train_dataloader), 1)),
                        desc=f"Iteration {iteration + 1}/{c.iterations} [Generate new dataset]",
                        total=1, leave=False)

    for batch_idx, batch in generate_bar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        tx_emb, tx_emb_uncond = get_embeddings_2(text_model, batch, c.num_gen_per_prompt, device)

        if not c.sequence_fixed:
            infos["all_lengths"] = batch["length"].repeat(c.num_gen_per_prompt)

        sequences, results_by_timestep = model.diffusionRL_guidance_split(tx_emb=tx_emb, tx_emb_uncond=tx_emb_uncond,
                                                                          infos=infos, target_model=target_model)

        metrics_reward = reward_model(sequences, infos, smplh, batch["text"] * c.num_gen_per_prompt, c)

        has_nan = (
                any(torch.isnan(t.cpu()).any() for t in metrics_reward.values())
                or torch.isnan(sequences.cpu()).any()
        )

        if has_nan:
            print("Found NaN in masked_tmr")
            save_dir = "NaN_folder"
            os.makedirs(save_dir, exist_ok=True)
            # Do something if there is at least one NaN
            try:
                masked_tmr_np = metrics_reward["tmr"].cpu().numpy()
                np.save(os.path.join(save_dir, "masked_tmr_with_nan.npy"), masked_tmr_np)

            except:
                masked_tmr_plus_np = metrics_reward["tmr++"].cpu().numpy()
                np.save(os.path.join(save_dir, "masked_tmr_plus_with_nan.npy"), masked_tmr_plus_np)
            # Save to .npy file
            np.save(os.path.join(save_dir, "sequences.npy"), sequences.cpu().numpy())
            import json
            with open(os.path.join(save_dir, "infos.json"), "w") as f:
                infos["all_lengths"] = infos["all_lengths"].tolist()
                json.dump(infos, f, indent=4)
            exit()

        timesteps = sorted(results_by_timestep.keys(), reverse=True)
        diff_step = len(timesteps)

        batch_size = sequences.shape[0]
        seq_len = results_by_timestep[0]["xt_new"].shape[1]

        # Store text embeddings just once, with repeat handling during concatenation
        all_rewards = []
        all_tmr = []
        all_tmr_plus_plus = []
        all_guo = []

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
        all_tx_uncond_output = []

        for t in timesteps:
            experiment = results_by_timestep[t]
            experiment = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in experiment.items()}

            if t == 0:
                all_rewards.append(metrics_reward["reward"].cpu())
            else:
                all_rewards.append(torch.zeros_like(metrics_reward["reward"]).cpu())

            all_xt_new.append(experiment["xt_new"])
            all_xt_old.append(experiment["xt_old"])
            all_t.append(torch.full((batch_size,), t, device=metrics_reward["reward"].device).cpu())
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
            all_tx_uncond_output.append(experiment["tx_uncond-output"])

        # Concatenate all the results for this batch
        dataset["r"].append(torch.cat(all_rewards, dim=0).view(diff_step, batch_size).T.clone())
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
        dataset["tx_uncond_output"].append(
            torch.cat(all_tx_uncond_output, dim=0).view(diff_step, batch_size, seq_len, 205).permute(1, 0, 2, 3))

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
    output_uncond = dataset["tx_uncond_output"][i: i + minibatch_size].to(device)

    return y, r, xt_1, xt, t, log_like, output_uncond


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

    wandb.log({"Train": {"Mean Reward": mean_r.item(), "Std Reward": std_r.item(), "iterations": iteration}})

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
        mb_counter = 0
        optimizer.zero_grad()
        for batch_idx in minibatch_bar:

            # with torch.autocast(device_type="cuda"):
            advantage = dataset["advantage"][batch_idx: batch_idx + c.train_batch_size].to(device)
            real_batch_size = advantage.shape[0]
            y, r, xt_1, xt, t, log_like, output_uncond = get_batch(dataset, batch_idx, real_batch_size, infos,
                                                                   diff_step, device)

            new_log_like, rl_pred = model.diffusionRL_guidance_split(y=y, infos=infos, t=t.view(diff_step * real_batch_size),
                                                      xt=xt.view(diff_step * real_batch_size, *xt.shape[2:]),
                                                      A=xt_1.view(diff_step * real_batch_size, *xt_1.shape[2:]),
                                                      output_uncond=output_uncond.view(diff_step * real_batch_size,
                                                                                       *output_uncond.shape[2:]))

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

            mb_counter += 1

            if mb_counter == 4:
                mb_counter = 0
                grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None))
                wandb.log({"Train": {"Gradient Norm": grad_norm.item(),
                                     "real_step": (iteration * c.train_epochs + e) * num_minibatches + (
                                             batch_idx // c.train_batch_size)}})

                torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            minibatch_bar.set_postfix(batch_loss=f"{combined_loss.item():.4f}")

        epoch_loss = tot_loss / num_minibatches
        epoch_policy_loss = tot_policy_loss / num_minibatches
        clipping_percentage = 100 * epoch_clipped_elements / epoch_total_elements

        train_bar.set_postfix(epoch_loss=f"{epoch_loss:.4f}")

        if c.betaL > 0:
            epoch_kl = tot_kl / num_minibatches
            wandb.log({"Train": {"gloss": epoch_loss, "epochs": iteration * c.train_epochs + e,
                                 "policy_loss": epoch_policy_loss, "kl_loss": epoch_kl,
                                 "trigger-clip": clipping_percentage}})
        else:
            wandb.log({"Train": {"loss": epoch_loss, "epochs": iteration * c.train_epochs + e,
                                 "policy_loss": epoch_policy_loss, "trigger-clip": clipping_percentage}})


def test(model, dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer, c, all_embedding_tmr,
         path, target_model):
    os.makedirs(path, exist_ok=True)
    out_formats = ['txt', 'smpl', 'joints', 'videojoints']

    ty_log = ""
    if "VAL" in path:
        ty_log = "Validation"
    else:
        ty_log = "Test"

    model.eval()

    if c.val_num_batch == 0:
        generate_bar = tqdm(enumerate(dataloader), leave=False, desc=f"[Validation/Test Generations]",
                            total=len(dataloader))
    else:
        generate_bar = tqdm(enumerate(itertools.islice(itertools.cycle(dataloader), c.val_num_batch)),
                            total=c.val_num_batch, leave=False, desc=f"[Validation/Test Generations]")

    total_reward, total_tmr, total_tmr_plus_plus, total_tmr_guo = 0, 0, 0, 0
    batch_count_reward, batch_count_tmr, batch_count_tmr_plus_plus, batch_count_guo = 0, 0, 0, 0

    for batch_idx, batch in generate_bar:
        tmp_path = path + "batch_" + str(batch_idx) + "/"
        os.makedirs(tmp_path, exist_ok=True)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)

            if not c.sequence_fixed:
                infos["all_lengths"] = batch["length"]

            sequences, _ = model.diffusionRL_guidance_split(tx_emb=tx_emb,
                                                                              tx_emb_uncond=tx_emb_uncond,
                                                                              infos=infos, target_model=target_model)



            if ((ty_log == "Validation" and batch_idx == 0) or ty_log == "Test") and c.render_videos:
                render(sequences, infos, smplh, joints_renderer, smpl_renderer, batch["text"], tmp_path, ty_log,
                       out_formats, video_log=True)

            metrics_reward = reward_model(sequences, infos, smplh, batch["text"], c)
            metrics = all_metrics(sequences, infos, smplh, batch["text"], c)

            has_nan = (
                    any(torch.isnan(t.cpu()).any() for t in metrics_reward.values())
                    or torch.isnan(sequences.cpu()).any()
            )

            if has_nan:
                print("Found NaN in masked_tmr")
                save_dir = "NaN_folder"
                os.makedirs(save_dir, exist_ok=True)
                # Do something if there is at least one NaN
                try:
                    masked_tmr_np = metrics_reward["tmr"].cpu().numpy()
                    np.save(os.path.join(save_dir, "masked_tmr_with_nan.npy"), masked_tmr_np)

                except:
                    masked_tmr_plus_np = metrics_reward["tmr++"].cpu().numpy()
                    np.save(os.path.join(save_dir, "masked_tmr_plus_with_nan.npy"), masked_tmr_plus_np)
                # Save to .npy file
                np.save(os.path.join(save_dir, "sequences.npy"), sequences.cpu().numpy())
                import json
                with open(os.path.join(save_dir, "infos.json"), "w") as f:
                    infos["all_lengths"] = infos["all_lengths"].tolist()
                    json.dump(infos, f, indent=4)

            total_reward += metrics_reward["reward"].sum().item()
            batch_count_reward += metrics_reward["reward"].shape[0]

            total_tmr += metrics["tmr"].sum().item()
            batch_count_tmr += metrics["tmr"].shape[0]

            total_tmr_plus_plus += metrics["tmr++"].sum().item()
            batch_count_tmr_plus_plus += metrics["tmr++"].shape[0]

            total_tmr_guo += metrics["guo"].sum().item()
            batch_count_guo += metrics["guo"].shape[0]

    avg_reward = total_reward / batch_count_reward
    avg_tmr = total_tmr / batch_count_tmr
    avg_tmr_plus_plus = total_tmr_plus_plus / batch_count_tmr_plus_plus
    avg_guo = total_tmr_guo / batch_count_guo

    return avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo


@hydra.main(config_path="configs", config_name="TrainRL", version_base="1.3")
def main(c: DictConfig):
    config_dict = OmegaConf.to_container(c, resolve=True)
    wandb.login(key="686f740320175b422861147930c51baba0e47fe6")

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

    diffusion_rl = instantiate(cfg.diffusion)
    diffusion_rl.load_state_dict(ckpt["state_dict"])
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

        # Check trainable parameters
        trainable_params = [name for name, param in diffusion_rl.denoiser.named_parameters() if param.requires_grad]
        print("Trainable LorA layer:", trainable_params)
        total_trainable_params = sum(p.numel() for p in diffusion_rl.denoiser.parameters() if p.requires_grad)
        print(f"Trainable parameters LorA: {total_trainable_params:,}")

        """END LORA"""

    # if c.betaL > 0:
    # diffusion_old = instantiate(cfg.diffusion)
    # diffusion_old.load_state_dict(ckpt["state_dict"])
    # diffusion_old = diffusion_old.to(device)
    # diffusion_old.train()
    # else:
    #     diffusion_old = None



    train_dataset = instantiate(cfg.data, split=str(c.dataset_name) + "val")
    val_dataset = instantiate(cfg.data, split=str(c.dataset_name) + "test")
    # test_dataset = instantiate(cfg.data, split=str(c.dataset_name) + "test")

    infos = {
        "featsname": cfg.motion_features,
        "fps": c.fps,
        "guidance_weight": c.guidance_weight
    }

    if c.sequence_fixed:
        infos["all_lengths"] = torch.tensor(np.full(2048, int(c.time * c.fps))).to(device)

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

    # test_dataloader = DataLoader(
    #    test_dataset,
    #    batch_size=c.val_batch_size,
    #    shuffle=False,
    #    drop_last=False,
    #    num_workers=c.num_workers,
    #    collate_fn=test_dataset.collate_fn
    # )

    # test_embedding_tmr = preload_tmr_text(val_dataloader)

    file_path = "../ResultRL/VAL/"
    os.makedirs(file_path, exist_ok=True)

    if c.lora:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, diffusion_rl.denoiser.parameters()), lr=c.lr,
                                      betas=(c.beta1, c.beta2), eps=c.eps,
                                      weight_decay=c.weight_decay)
    else:
        optimizer = torch.optim.AdamW(diffusion_rl.parameters(), lr=c.lr, betas=(c.beta1, c.beta2), eps=c.eps,
                                      weight_decay=c.weight_decay)

    avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo = test(diffusion_rl, val_dataloader, device, infos, text_model,
                                                           smplh, joints_renderer, smpl_renderer, c, val_embedding_tmr,
                                                           path="../ResultRL/VAL/OLD/", target_model=diffusion_rl)
    wandb.log({"Validation": {"Reward": avg_reward, "TMR": avg_tmr, "TMR++": avg_tmr_plus_plus, "Guo": avg_guo,
                              "iterations": 0}})

    iter_bar = tqdm(range(c.iterations), desc="Iterations", total=c.iterations)
    for iteration in iter_bar:

        train_datasets_rl = generate(diffusion_rl, train_dataloader, iteration, c, device, infos, text_model, smplh,
                                     train_embedding_tmr, diffusion_rl)  # , generation_iter
        train(diffusion_rl, optimizer, train_datasets_rl, iteration, c, infos, device, old_model=diffusion_rl)

        if (iteration + 1) % c.val_iter == 0:
            avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo = test(diffusion_rl, val_dataloader, device, infos,
                                                                   text_model, smplh, joints_renderer,
                                                                   smpl_renderer, c, val_embedding_tmr,
                                                                   path="ResultRL/VAL/" + str(iteration + 1) + "/", target_model=diffusion_rl)
            wandb.log({"Validation": {"Reward": avg_reward, "TMR": avg_tmr, "TMR++": avg_tmr_plus_plus, "Guo": avg_guo,
                                      "iterations": iteration + 1}})
            torch.save(diffusion_rl.state_dict(), 'RL_Model/checkpoint_' + str(iteration + 1) + '.pth')
            iter_bar.set_postfix(val_tmr=f"{avg_tmr:.4f}")

    # file_path = "ResultRL/TEST/"
    # os.makedirs(file_path, exist_ok=True)
    # avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo = test(diffusion_rl, test_dataloader, device, infos, text_model,smplh, joints_renderer, smpl_renderer, c, test_embedding_tmr, path="ResultRL/TEST/")
    # wandb.log({"Test": {"Reward": avg_reward, "TMR": avg_tmr, "TMR++": avg_tmr_plus_plus, "Guo": avg_guo}})

    torch.save(diffusion_rl.state_dict(), 'RL_Model/model_final.pth')


if __name__ == "__main__":
    main()
    wandb.finish()