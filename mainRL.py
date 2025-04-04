import argparse
import itertools
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

def render(x_starts, infos, smplh, joints_renderer, smpl_renderer, texts, file_path):
    out_formats = ['txt', 'videojoints']  # 'joints', 'txt', 'smpl', 'videojoints', 'videosmpl'
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


def stillness_reward(sequences, infos, smplh, texts):
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

    joints = torch.stack(joint_positions)  # (batch_size, N_frames, 22, 3)
    dt = 1.0 / 200

    velocities = torch.diff(joints, dim=1) / dt
    velocity_loss = torch.mean(velocities.pow(2), dim=(1, 2, 3))

    reward = velocity_loss
    return - reward


def generate(model, train_dataloader, iteration, args, device, infos, text_model, smplh):
    model.eval()

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

    }

    generate_bar = tqdm(enumerate(itertools.islice(train_dataloader, args.n_batch)),
                        desc=f"Iteration {iteration + 1}/{args.iterations} [Generate new dataset]", total=args.n_batch)
    for batch_idx, batch in generate_bar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        for i in range(args.num_examples):

            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)

            sequences, results_by_timestep = model.diffusionRL(tx_emb=tx_emb, tx_emb_uncond=tx_emb_uncond, infos=infos,
                                                               guidance_weight=args.guidance_weight_generation)

            r = stillness_reward(sequences, infos, smplh, batch["text"])

            timesteps = sorted(results_by_timestep.keys(), reverse=True)
            diff_step = len(timesteps)

            batch_size = r.shape[0]
            seq_len = results_by_timestep[0]["xt_new"].shape[1]

            # Store text embeddings just once, with repeat handling during concatenation
            all_rewards = []
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
                    all_rewards.append(r.cpu())
                else:
                    all_rewards.append(torch.zeros_like(r).cpu())

                all_xt_new.append(experiment["xt_new"])
                all_xt_old.append(experiment["xt_old"])
                all_t.append(torch.full((batch_size,), t, device=r.device).cpu())
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


def train(model, optimizer, dataset, iteration, args, infos, device):
    model.train()

    delta = 1e-7
    mask = dataset["r"] != 0
    mean_r = torch.mean(dataset["r"][mask], dim=0)
    std_r = torch.std(dataset["r"][mask], dim=0)
    wandb.log({"Train": {"Mean Reward": mean_r.item(), "Std Reward": std_r.item(), "iterations": iteration}})

    dataset["advantage"] = torch.zeros_like(dataset["r"])
    dataset["advantage"][mask] = (dataset["r"][mask] - mean_r) / (std_r + delta)
    dataset["advantage"] = (dataset["r"] - mean_r) / (std_r + delta)

    num_minibatches = (dataset["r"].shape[0] + args.train_batch_size - 1) // args.train_batch_size

    diff_step = dataset["xt_1"][0].shape[0]

    train_bar = tqdm(range(args.train_epochs), desc=f"Iteration {iteration + 1}/{args.iterations} [Train]")
    for e in train_bar:
        tot_loss = 0
        minibatch_bar = tqdm(range(0, dataset["r"].shape[0], args.train_batch_size), leave=False, desc="Minibatch")

        dataset = prepare_dataset(dataset)
        for batch_idx in minibatch_bar:
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                advantage = dataset["advantage"][batch_idx: batch_idx + args.train_batch_size].to(device)
                real_batch_size = args.train_batch_size
                y, r, xt_1, xt, t, log_like = get_batch(dataset, batch_idx, real_batch_size, infos, diff_step, device)

                new_log_like = model.diffusionRL(y=y, infos=infos, t=t.view(diff_step * real_batch_size),
                                                 xt=xt.view(diff_step * real_batch_size, *xt.shape[2:]),
                                                 A=xt_1.view(diff_step * real_batch_size, *xt_1.shape[2:]),
                                                 guidance_weight=args.guidance_weight_train).view(real_batch_size, diff_step)

            ratio = torch.exp(new_log_like - log_like)

            real_adv = advantage[:, -1:]  # r[:,-1:]

            clip_adv = torch.clamp(ratio, 1.0 - args.advantage_clip_epsilon, 1.0 + args.advantage_clip_epsilon) * real_adv
            policy_loss = -torch.min(ratio * real_adv, clip_adv).sum(1).mean()

            policy_loss.backward()
            tot_loss += policy_loss.item()

            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None))
            wandb.log({"Train": {"Gradient Norm": grad_norm.item(),
                                 "real_step": (iteration * args.train_epochs + e) * num_minibatches + (batch_idx // args.train_batch_size)}})

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            minibatch_bar.set_postfix(policy_loss=f"{policy_loss.item():.4f}")

        epoch_loss = tot_loss / num_minibatches
        train_bar.set_postfix(epoch_loss=f"{epoch_loss:.4f}")
        wandb.log({"Train": {"loss": epoch_loss, "epochs": iteration * args.train_epochs + e}})


def test(model, dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer,args, path):

    os.makedirs(path, exist_ok=True)

    model.eval()
    generate_bar = tqdm(dataloader, desc=f"[Validation/Test Generations]")

    total_reward = 0
    batch_count = 0

    for batch_idx, batch in enumerate(generate_bar):
        tmp_path = path + "batch_" + str(batch_idx) + "/"
        os.makedirs(tmp_path, exist_ok=True)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            tx_emb, tx_emb_uncond = get_embeddings(text_model, batch, device)
            sequences, _ = model.diffusionRL(tx_emb, tx_emb_uncond, infos, guidance_weight=args.guidance_weight_valid)
            render(sequences, infos, smplh, joints_renderer, smpl_renderer, batch["text"], tmp_path)
            r = stillness_reward(sequences, infos, smplh, batch["text"])  # shape [batch_size]
            total_reward += r.sum().item()
            batch_count += r.shape[0]

        break

    avg_reward = total_reward / batch_count

    return avg_reward


def create_folder_results(name):
    results_dir = name
    os.makedirs(results_dir, exist_ok=True)

    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training configuration parser")

    # General parameters
    parser.add_argument("--num_examples", type=int, default=4, help="Number of examples")
    parser.add_argument("--n_batch", type=int, default=256 // (4 * 12), help="Number of batches")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Num workers dataloader")

    # Training parameters
    parser.add_argument("--iterations", type=int, default=10000, help="Number of iterations")
    parser.add_argument("--train_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=48, help="Training batch size")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--advantage_clip_epsilon", type=float, default=1e-4, help="Advantage clipping epsilon (1e-4, 0.2)")

    # Guidance Weight
    parser.add_argument("--guidance_weight_train", type=float, default=1.0, help="Guidance weight at training time")
    parser.add_argument("--guidance_weight_generation", type=float, default=1.0, help="Guidance weight at dataset generation time")
    parser.add_argument("--guidance_weight_valid", type=float, default=7.0, help="Guidance weight at test generation time")

    # sequence parameters
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--time", type=float, default=2.5, help="Duration in seconds")
    parser.add_argument("--joint_stype", type=str, default="both", choices=["both", "smpljoints"], help="Joint style type")

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Betas for optimizer (Beta1)")
    parser.add_argument("--beta2", type=float, default=0.999, help="Betas for optimizer (Beta2)")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon value for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # Validation/Test parameters
    parser.add_argument("--val_iter", type=int, default=25, help="Validation iterations")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Validation batch size")

    # pretrained model parameters
    parser.add_argument("--run_dir", type=str, default='pretrained_models/mdm-smpl_clip_smplrifke_humanml3d', help="Run directory")
    parser.add_argument("--ckpt_name", type=str, default='logs/checkpoints/last.ckpt', help="Checkpoint file name")

    return parser.parse_args()


@hydra.main(config_path="configs", config_name="TrainRL", version_base="1.3")
def main(c: DictConfig):

    args = parse_arguments()

    wandb.init(
        project="TM-BM",
        name="New_Experiment",
        config={
            "args": vars(args)
        }
    )

    create_folder_results("ResultRL")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = read_config(args.run_dir)

    ckpt_path = os.path.join(args.run_dir, args.ckpt_name)
    print("Loading the checkpoint")
    ckpt = torch.load(str(ckpt_path), map_location=c.device)

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
    diffusion_rl.load_state_dict(ckpt["state_dict"])
    diffusion_rl = diffusion_rl.to(device)

    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split="val")
    test_dataset = instantiate(cfg.data, split="val")

    infos = {
        "all_lengths": torch.tensor(np.full(2048, int(args.time * args.fps))).to(device),
        "featsname": cfg.motion_features,
        "fps": args.fps,
        "guidance_weight": c.guidance
    }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn
    )

    file_path = "ResultRL/VAL/"
    os.makedirs(file_path, exist_ok=True)

    optimizer = torch.optim.AdamW(diffusion_rl.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)

    avg_reward = test(diffusion_rl, val_dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer,args, path="ResultRL/VAL/OLD/")
    wandb.log({"Validation": {"Reward": avg_reward, "iterations": 0}})
    print("Avg Reward OLD model:", avg_reward)

    for iteration in range(args.iterations):

        train_datasets_rl = generate(diffusion_rl, train_dataloader, iteration, args, device, infos, text_model, smplh)
        train(diffusion_rl, optimizer, train_datasets_rl, iteration, args, infos, device)

        if (iteration + 1) % args.val_iter == 0:
            avg_reward = test(diffusion_rl, val_dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer,args, path="ResultRL/VAL/" + str(iteration + 1) + "/")
            wandb.log({"Validation": {"Reward": avg_reward, "iterations": iteration + 1}})
            print("Avg Reward:", avg_reward, " at iteration:", iteration)

    avg_reward = test(diffusion_rl, test_dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer,args, path="ResultRL/TEST/")
    wandb.log({"Test": {"Reward": avg_reward}})
    print("Avg Reward Test Set:", avg_reward)

    torch.save(diffusion_rl.state_dict(), 'RL_Model/model_state.pth')


if __name__ == "__main__":
    main()
    wandb.finish()
