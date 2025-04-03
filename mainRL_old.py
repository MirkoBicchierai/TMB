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
import gc
from TMR.mtt.load_tmr_model import load_tmr_model_easy
from src.tools.guofeats.motion_representation import joints_to_guofeats, guofeats_to_joints
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.model.tmr import get_sim_matrix

tmr_forward = load_tmr_model_easy(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")

wandb.init(
    project="TM-BM",
    name="experiment_mobile",
    config={
        "learning_rate": 1e-3,
        "epochs": 4,
    })

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def render(x_starts, infos, smplh, joints_renderer, smpl_renderer, texts, file_path):
    out_formats = ['txt', 'videojoints', 'txt', 'smpl']  # 'joints', 'txt', 'smpl', 'videojoints', 'videosmpl'
    tmp = file_path
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


def calc_eval_stats(X, Y, smplh):
    """
        Calculate Motion2Motion (m2m) and the Motion2Text (m2t) between the recostructed motion, the gt motion and the gt text.
    """
    if is_list_of_strings(X):
        X_latents = tmr_forward(X)  # tensor(N, 256)
    else:
        X_guofeats = smpl_to_guofeats(X, smplh)
        X_latents = tmr_forward(X_guofeats)  # tensor(N, 256)
    if is_list_of_strings(Y):
        Y_latents = tmr_forward(Y)  # tensor(N, 256)
    else:
        Y_guofeats = smpl_to_guofeats(Y, smplh)
        Y_latents = tmr_forward(Y_guofeats)

    sim_matrix = get_sim_matrix(X_latents, Y_latents).numpy()
    return sim_matrix


def is_list_of_strings(var):
    return isinstance(var, list) and all(isinstance(item, str) for item in var)


def reward_tmr(sequences, infos, smplh, texts):
    reward_scores = torch.zeros(sequences.shape[0])

    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        text = texts[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]

        motion = [x_start.detach().cpu()]
        text = [text]
        sim_matrix = calc_eval_stats(motion, text, smplh)
        reward_scores[idx] = torch.tensor(sim_matrix[0][0])

    return reward_scores


def generate(model, train_dataloader, iteration, iterations, device, infos, text_model, smplh, joints_renderer,
             smpl_renderer, num_examples):
    model.eval()
    generate_bar = tqdm(train_dataloader, desc=f"Iteration {iteration + 1}/{iterations} [Generate new dataset]")

    dataset = {
        "r": [],
        "xt_1": [],
        "xt": [],
        "t": [],
        "log_like": [],

        "tx_emb_x": [],
        "tx_emb_mask": [],
        "tx_emb_length": [],
        "tx_emb_uncond_x": [],
        "tx_emb_uncond_mask": [],
        "tx_emb_uncond_length": [],
    }

    for batch_idx, batch in enumerate(generate_bar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        for i in range(num_examples):
            with torch.no_grad():
                tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)
                sequences, results_by_timestep = model.diffusionRL(tx_emb, tx_emb_uncond, infos, guidance_weight=1.0)
                r = reward_tmr(sequences, infos, smplh, batch["text"])

            timesteps = results_by_timestep.keys()
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

            for t in timesteps:
                experiment = results_by_timestep[t]

                if t == 0:
                    all_rewards.append(r.cpu())
                else:
                    all_rewards.append(torch.zeros_like(r).cpu())

                all_xt_new.append(experiment["xt_new"])
                all_xt_old.append(experiment["xt_old"])
                all_t.append(torch.full((batch_size,), t, device=r.device).cpu())
                all_log_probs.append(experiment["log_prob"])

            # Concatenate all the results for this batch
            dataset["r"].append(torch.cat(all_rewards, dim=0).clone())
            dataset["xt_1"].append(torch.cat(all_xt_new, dim=0))
            dataset["xt"].append(torch.cat(all_xt_old, dim=0))
            dataset["t"].append(torch.cat(all_t, dim=0))
            dataset["log_like"].append(torch.cat(all_log_probs, dim=0))

        break

    for key in dataset:
        dataset[key] = torch.cat(dataset[key], dim=0)

    return dataset


def get_batch(dataset, i, minibatch_size, device):
    tx_emb_x = dataset["tx_emb_x"][i: i + minibatch_size]
    tx_emb_mask = dataset["tx_emb_mask"][i: i + minibatch_size]
    tx_emb_length = dataset["tx_emb_length"][i: i + minibatch_size]
    tx_emb_uncond_x = dataset["tx_emb_uncond_x"][i: i + minibatch_size]
    tx_emb_uncond_mask = dataset["tx_emb_uncond_mask"][i: i + minibatch_size]
    tx_emb_uncond_length = dataset["tx_emb_uncond_length"][i: i + minibatch_size]

    tx_emb = {
        "x": tx_emb_x.to(device),
        "mask": tx_emb_mask.to(device),
        "length": tx_emb_length.to(device)
    }

    tx_emb_uncond = {
        "x": tx_emb_uncond_x.to(device),
        "mask": tx_emb_uncond_mask.to(device),
        "length": tx_emb_uncond_length.to(device)
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

    mask = dataset["r"] != 0
    mean_r = torch.mean(dataset["r"][mask], dim=0)
    std_r = torch.std(dataset["r"][mask], dim=0)

    #dataset["advantage"] = torch.zeros_like(dataset["r"])
    #dataset["advantage"][mask] = (dataset["r"][mask] - mean_r) / (std_r + delta)

    dataset["advantage"] = (dataset["r"] - mean_r) / (std_r + delta)

    num_minibatches = (dataset["r"].shape[0] + batch_size - 1) // batch_size

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

            tx_emb, tx_emb_uncond = get_batch(dataset, batch_idx, batch_size, device)

            new_log_like = model.diffusionRL(tx_emb, tx_emb_uncond, infos, t=t, xt=xt, A=xt_1, guidance_weight=1.0)

            # print(f"log_like: {log_like}")
            # print(f"new_log_like: {new_log_like}")
            # print(f"Difference: {(new_log_like - log_like).abs().mean()}")

            #print(new_log_like, log_like)
            ratio = torch.exp(new_log_like - log_like)
            #print(ratio)

            # adv_per_ratio = -advantage * ratio

            epsilon = 0.2
            # clipped_adv_per_ratio = -torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
            # final_advantage = torch.max(adv_per_ratio, clipped_adv_per_ratio).mean()

            clip_adv = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
            policy_loss = -torch.min(ratio * advantage, clip_adv).mean()
            #print(policy_loss)

            tot_loss += policy_loss.item()

            policy_loss.backward()

            """ 
            
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
            print(f"Total gradient norm: {grad_norm}")
            
            """

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            minibatch_bar.set_postfix(advantage=f"{policy_loss.item():.4f}")

        epoch_loss = tot_loss / num_minibatches
        train_bar.set_postfix(avg_advantage=f"{epoch_loss:.4f}")
        wandb.log({"Train": {"loss": epoch_loss}})


def create_folder_results(name):
    results_dir = name
    os.makedirs(results_dir, exist_ok=True)

    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def test(model, dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer, path):
    model.eval()
    generate_bar = tqdm(dataloader, desc=f"[Validation/Test Generations]")

    total_reward = 0
    batch_count = 0

    for batch in generate_bar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)
            sequences, _ = model.diffusionRL(tx_emb, tx_emb_uncond, infos, guidance_weight=7.0)
            render(sequences, infos, smplh, joints_renderer, smpl_renderer, batch["text"], path)
            r = reward_tmr(sequences, infos, smplh, batch["text"])  # shape [batch_size]
            total_reward += r.sum().item()
            batch_count += r.shape[0]

        break

    avg_reward = total_reward / batch_count

    return avg_reward



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
    val_dataset = instantiate(cfg.data, split="val")
    test_dataset = instantiate(cfg.data, split="val")

    iterations = 10000

    num_examples = 4
    batch_size = 32

    val_batch_size = 2

    train_batch_size = 2
    train_epochs = 5

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

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=12,
        collate_fn=val_dataset.collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=12,
        collate_fn=test_dataset.collate_fn
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

    lr = 2 * 1e-6
    optimizer = torch.optim.AdamW(diffusion_rl.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    # optimizer = torch.optim.SGD(diffusion_rl.parameters(), lr=lr)

    file_path = "ResultRL/VAL/"
    os.makedirs(file_path, exist_ok=True)
    file_path = "ResultRL/VAL/OLD/"
    os.makedirs(file_path, exist_ok=True)
    avg_reward = test(diffusion_rl, val_dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer,
                      file_path)
    wandb.log({"Validation": {"Reward": avg_reward}})
    print("Avg Reward OLD model:", avg_reward)
    gc.collect()

    for iteration in range(iterations):
        train_datasets_rl = generate(diffusion_rl, train_dataloader, iteration, iterations, device, infos, text_model,
                                     smplh, joints_renderer, smpl_renderer, num_examples)
        gc.collect()

        train(diffusion_rl, optimizer, train_datasets_rl, iteration, iterations, infos, device, train_batch_size,
              train_epochs)

        file_path = "ResultRL/VAL/" + str(iteration) + "/"
        os.makedirs(file_path, exist_ok=True)
        avg_reward = test(diffusion_rl, val_dataloader, device, infos, text_model, smplh, joints_renderer,
                          smpl_renderer, file_path)
        wandb.log({"Validation": {"Reward": avg_reward}})
        print("Avg Reward:", avg_reward, " at iteration:", iteration)
        gc.collect()

    file_path = "ResultRL/TEST/"
    os.makedirs(file_path, exist_ok=True)
    avg_reward = test(diffusion_rl, test_dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer,
                      file_path)
    wandb.log({"Test": {"Reward": avg_reward}})
    print("Avg Reward Test Set:", avg_reward)
    gc.collect()

    torch.save(diffusion_rl.state_dict(), 'RL_Model/model_state.pth')


if __name__ == "__main__":
    main()
    wandb.finish()
