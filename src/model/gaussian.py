import copy
import logging
from collections import defaultdict
from matplotlib.pyplot import barbs
from tqdm import tqdm
import torch
from .diffusion_base import DiffuserBase
from ..data.collate import length_to_mask, collate_tensor_with_padding
from src.stmc import combine_features_intervals, interpolate_intervals, bada_bim_core
import numpy as np
import os
import itertools
from src.tools.extract_joints import extract_joints
from src.tools.smpl_layer import SMPLH

smplh = SMPLH(
    path="deps/smplh",
    jointstype="both",
    input_pose_rep="axisangle",
    gender="male",
)

# Inplace operator: return the original tensor
# work with a list of tensor as well
def masked(tensor, mask):
    if isinstance(tensor, list):
        return [masked(t, mask) for t in tensor]
    tensor[~mask] = 0.0
    return tensor

def nan_masked(tensor, mask):
    if isinstance(tensor, list):
        return [masked(t, mask) for t in tensor]
    tensor[~mask] =  float('nan')
    return tensor



logger = logging.getLogger(__name__)


def remove_padding_to_numpy(x, length):
    x = x.detach().cpu().numpy()
    return [d[:l] for d, l in zip(x, length)]


class GaussianDiffusion(DiffuserBase):
    name = "gaussian"

    def __init__(
            self,
            denoiser,
            schedule,
            timesteps,
            motion_normalizer,
            text_normalizer,
            prediction: str = "x",
            lr: float = 2e-4,
            weight=1,
            mcd=False
    ):
        super().__init__(schedule, timesteps)

        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.lr = lr
        self.prediction = prediction

        self.reconstruction_loss = torch.nn.MSELoss(reduction="mean")

        # normalization
        self.motion_normalizer = motion_normalizer
        self.text_normalizer = text_normalizer

        self.weight = weight
        self.mcd = mcd
        self.original_timeline = True

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

    def activate_composition_model(self):
        print(f"Num of trainable parametrers {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def configure_optimizers(self) -> None:
        return {"optimizer": torch.optim.AdamW(lr=self.lr, params=self.parameters())}

    def prepare_tx_emb(self, tx_emb):
        # Text embedding normalization
        if "mask" not in tx_emb:
            tx_emb["mask"] = length_to_mask(tx_emb["length"], device=self.device)
        tx = {
            "x": masked(self.text_normalizer(tx_emb["x"]), tx_emb["mask"]),
            "length": tx_emb["length"],
            "mask": tx_emb["mask"],
        }
        return tx

    def move_to_device(self, obj, device):
        """
        Recursively moves all tensors in obj to the specified device.
        Works with tensors, lists, tuples, and dictionaries containing tensors.
        """
        if obj is None:
            return None
        elif isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.move_to_device(item, device) for item in obj)
        elif isinstance(obj, dict):
            return {k: self.move_to_device(v, device) for k, v in obj.items()}
        else:
            return obj

    def diffusion_step(self, batch, batch_idx, training=False):
        mask = batch["mask"]

        # normalization
        x = masked(self.motion_normalizer(batch["x"]), mask)
        y = {
            "length": batch["length"],
            "mask": mask,
            "tx": self.prepare_tx_emb(batch["tx"]),
            # the condition is already dropped sometimes in the dataloader
        }

        bs = len(x)
        # Sample a diffusion step between 0 and T-1
        # 0 corresponds to noising from x0 to x1
        # T-1 corresponds to noising from xT-1 to xT
        t = torch.randint(0, self.timesteps, (bs,), device=x.device)

        # Create a noisy version of x
        # no noise for padded region
        noise = masked(torch.randn_like(x), mask)
        xt = self.q_sample(xstart=x, t=t, noise=noise)
        xt = masked(xt, mask)

        # denoise it
        # no drop cond -> this is done in the training dataloader already
        # give "" instead of the text
        # denoise it
        output = masked(self.denoiser(xt, y, t), mask)

        # Predictions
        xstart = masked(self.output_to("x", output, xt, t), mask)

        xloss = self.reconstruction_loss(xstart, x)
        loss = {"loss": xloss}
        return loss

    def log_likelihood(self, x, mu, sigma):
        var = sigma ** 2 + 1e-8  # Ensure variance is > 0
        log_prob = -0.5 * (torch.log(2 * torch.pi * var) + ((x - mu) ** 2) / var)
        return log_prob

    def training_step(self, batch, batch_idx):
        bs = len(batch["x"])
        loss = self.diffusion_step(batch, batch_idx, training=True)
        for loss_name in sorted(loss):
            loss_val = loss[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        bs = len(batch["x"])
        loss = self.diffusion_step(batch, batch_idx)
        for loss_name in sorted(loss):
            loss_val = loss[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )

        return loss["loss"]

    def on_train_epoch_end(self):
        dico = {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.global_step),
        }
        # reset losses
        self._saved_losses = defaultdict(list)
        self.losses = []
        self.log_dict(dico)

    # dispatch
    def forward(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        ff = self.text_forward
        return ff(tx_emb, tx_emb_uncond, infos, progress_bar=progress_bar)

    def text_forward(
            self,
            tx_emb,
            tx_emb_uncond,
            infos,
            progress_bar=tqdm,
    ):
        # normalize text embeddings first
        device = self.device

        lengths = infos["all_lengths"]

        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        duration = max(lengths)
        nfeats = self.denoiser.nfeats

        shape = bs, duration, nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)  # range(self.timesteps - 1, -2, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample(xt, y, t)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart

    def p_sample(self, xt, y, t):
        # guided forward
        output_cond = self.denoiser(xt, y, t)

        guidance_weight = y["infos"].get("guidance_weight", 1.0)

        if guidance_weight == 1.0:
            output = output_cond
        else:
            y_uncond = y.copy()  # not a deep copy
            y_uncond["tx"] = y_uncond["tx_uncond"]

            output_uncond = self.denoiser(xt, y_uncond, t)
            # classifier-free guidance
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(output, xt, t)

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise
        xstart = output
        return x_out, xstart

    def p_sample_split(self, xt, y, t, guidance_weight,output_uncond ):
        output_cond = masked(self.denoiser(xt, y, t), y["mask"])

        if guidance_weight == 1.0:
            output = output_cond
        else:

            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(output, xt, t)
        mean = masked(mean, y["mask"])
        sigma = torch.max(sigma, torch.tensor(0.1, device=sigma.device))

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise
        xstart = output
        return x_out, xstart, mean, sigma

    def p_sample_2(self, xt, y, t, guidance_weight):
        output_cond = masked(self.denoiser(xt, y, t), y["mask"])

        if guidance_weight == 1.0:
            output = output_cond
        else:
            y_uncond = y.copy()
            y_uncond["tx"] = y_uncond["tx_uncond"]
            output_uncond = masked(self.denoiser(xt, y_uncond, t), y["mask"])
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(output, xt, t)
        mean = masked(mean, y["mask"])
        sigma = torch.max(sigma, torch.tensor(0.1, device=sigma.device))

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise
        xstart = output
        return x_out, xstart, mean, sigma

    def p_sample_controllable(self, xt, y, t, guidance_weight, p):
        output_cond = masked(self.denoiser(xt, y, p, t), y["mask"])

        if guidance_weight == 1.0:
            output = output_cond
        else:
            y_uncond = y.copy()
            y_uncond["tx"] = y_uncond["tx_uncond"]
            output_uncond = masked(self.denoiser(xt, y_uncond, p, t), y["mask"])
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(output, xt, t)
        mean = masked(mean, y["mask"])

        sigma = torch.max(sigma, torch.tensor(0.1, device=sigma.device))

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise
        xstart = output
        return x_out, xstart, mean, sigma

    def diffusionRL_controllable(self, tx_emb=None, tx_emb_uncond=None, infos=None, y=None, t=None, xt=None, A=None, p=None):

        device = self.device

        if A is None:

            lengths = infos["all_lengths"][0:tx_emb["x"].shape[0]]
            mask = length_to_mask(lengths, device=device)

            y = {
                "length": lengths,
                "mask": mask,
                "tx": self.prepare_tx_emb(tx_emb),
                "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
                "infos": infos,
            }

            bs = len(lengths)
            duration = max(lengths)

            if isinstance(duration, torch.Tensor):
                duration = int(duration.item())

            nfeats = self.denoiser.nfeats

            shape = bs, duration, nfeats

            xt = masked(torch.randn(shape, device=device), mask)

            # iterator = range(self.timesteps - 1, -1, -1)

            iterator = list(range(self.timesteps - 1, -1, -2)) + [0]

            results = {}

            for diffusion_step in iterator:
                t = torch.full((bs,), diffusion_step, device=device)
                xt_old = xt.clone()

                xt, x_start, mean, sigma = self.p_sample_controllable(xt, y, t, infos["guidance_weight"], p=p)

                x_start = masked(x_start, mask)
                xt = masked(xt, mask)

                log_likelihood = self.log_likelihood(xt, mean, sigma)
                log_likelihood = nan_masked(log_likelihood, mask)
                log_prob = log_likelihood.nanmean(dim=[1, 2])

                results[diffusion_step] = {

                    "t": diffusion_step,  # begin the t
                    "xt_old": xt_old.detach().cpu(),  # begin the xt
                    "xt_new": xt.clone().detach().cpu(),  # begin the A when train PPO
                    "log_prob": log_prob.detach().cpu(),
                    "positions": p.detach().cpu(),

                    "length": lengths.detach().cpu(),
                    "mask": mask.detach().cpu(),
                    "tx-x": y["tx"]["x"],
                    "tx-length": y["tx"]["length"],
                    "tx-mask": y["tx"]["mask"],

                    "tx_uncond-x": y["tx_uncond"]["x"],
                    "tx_uncond-length": y["tx_uncond"]["length"],
                    "tx_uncond-mask": y["tx_uncond"]["mask"],

                }

                results = {k: v.detach().to("cpu") if isinstance(v, torch.Tensor) else v for k, v in results.items()}

            x_start = self.motion_normalizer.inverse(x_start)

            return x_start, results

        else:

            xt_pred, _, mean, sigma = self.p_sample_controllable(xt, y, t, infos["guidance_weight"], p=p)
            xt_pred = masked(xt_pred, y["mask"])
            log_likelihood = self.log_likelihood(A, mean, sigma)
            log_likelihood = nan_masked(log_likelihood, y["mask"])
            log_probs = log_likelihood.nanmean(dim=[1, 2])

            return log_probs, xt_pred


    def diffusionRL(self,tx_emb=None, tx_emb_uncond=None, infos=None, y=None, t=None, xt=None,A=None):

        device = self.device

        if A is None:

            lengths = infos["all_lengths"][0:tx_emb["x"].shape[0]]
            mask = length_to_mask(lengths, device=device)

            y = {
                "length": lengths,
                "mask": mask,
                "tx": self.prepare_tx_emb(tx_emb),
                "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
                "infos": infos,
            }

            bs = len(lengths)
            duration = max(lengths)

            if isinstance(duration, torch.Tensor):
                duration = int(duration.item())

            nfeats = self.denoiser.nfeats

            shape = bs, duration, nfeats

            xt = masked(torch.randn(shape, device=device), mask)

            # iterator = range(self.timesteps - 1, -1, -1)

            iterator = list(range(self.timesteps - 1, -1, -2)) + [0]

            results = {}

            for diffusion_step in iterator:
                t = torch.full((bs,), diffusion_step, device=device)
                xt_old = xt.clone()

                xt, x_start, mean, sigma = self.p_sample_2(xt, y, t, infos["guidance_weight"])

                x_start = masked(x_start, mask)
                xt = masked(xt, mask)

                log_likelihood = self.log_likelihood(xt, mean, sigma)
                log_likelihood = nan_masked(log_likelihood, mask)
                log_prob = log_likelihood.nanmean(dim=[1, 2])

                results[diffusion_step] = {

                    "t": diffusion_step,  # begin the t
                    "xt_old": xt_old.detach().cpu(),  # begin the xt
                    "xt_new": xt.clone().detach().cpu(),  # begin the A when train PPO
                    "log_prob": log_prob.detach().cpu(),

                    "length": lengths.detach().cpu(),
                    "mask": mask.detach().cpu(),
                    "tx-x": y["tx"]["x"],
                    "tx-length": y["tx"]["length"],
                    "tx-mask": y["tx"]["mask"],

                    "tx_uncond-x": y["tx_uncond"]["x"],
                    "tx_uncond-length": y["tx_uncond"]["length"],
                    "tx_uncond-mask": y["tx_uncond"]["mask"],

                }

                results = {k: v.detach().to("cpu") if isinstance(v, torch.Tensor) else v for k, v in results.items()}

            x_start = self.motion_normalizer.inverse(x_start)

            return x_start, results

        else:

            xt_pred, _, mean, sigma = self.p_sample_2(xt, y, t, infos["guidance_weight"])
            xt_pred = masked(xt_pred, y["mask"])
            log_likelihood = self.log_likelihood(A, mean, sigma)
            log_likelihood = nan_masked(log_likelihood, y["mask"])
            log_probs = log_likelihood.nanmean(dim=[1, 2])

            return log_probs, xt_pred

    def diffusionRL_guidance_split(self,tx_emb=None, tx_emb_uncond=None, infos=None, y=None, t=None, xt=None,A=None, target_model=None, output_uncond=None):

        device = self.device

        if A is None:

            lengths = infos["all_lengths"][0:tx_emb["x"].shape[0]]
            mask = length_to_mask(lengths, device=device)

            y = {
                "length": lengths,
                "mask": mask,
                "tx": self.prepare_tx_emb(tx_emb),
                "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
                "infos": infos,
            }

            bs = len(lengths)
            duration = max(lengths)

            if isinstance(duration, torch.Tensor):
                duration = int(duration.item())

            nfeats = self.denoiser.nfeats

            shape = bs, duration, nfeats

            xt = masked(torch.randn(shape, device=device), mask)

            # iterator = range(self.timesteps - 1, -1, -1)

            iterator = list(range(self.timesteps - 1, -1, -2)) + [0]

            results = {}

            for diffusion_step in iterator:
                t = torch.full((bs,), diffusion_step, device=device)
                xt_old = xt.clone()

                y_uncond = y.copy()
                y_uncond["tx"] = y_uncond["tx_uncond"]
                output_uncond = masked(target_model.denoiser(xt, y_uncond, t), y["mask"])

                xt, x_start, mean, sigma = self.p_sample_split(xt, y, t, infos["guidance_weight"], output_uncond)

                x_start = masked(x_start, mask)
                xt = masked(xt, mask)

                log_likelihood = self.log_likelihood(xt, mean, sigma)
                log_likelihood = nan_masked(log_likelihood, mask)
                log_prob = log_likelihood.nanmean(dim=[1, 2])

                results[diffusion_step] = {

                    "t": diffusion_step,  # begin the t
                    "xt_old": xt_old.detach().cpu(),  # begin the xt
                    "xt_new": xt.clone().detach().cpu(),  # begin the A when train PPO
                    "log_prob": log_prob.detach().cpu(),

                    "length": lengths.detach().cpu(),
                    "mask": mask.detach().cpu(),
                    "tx-x": y["tx"]["x"],
                    "tx-length": y["tx"]["length"],
                    "tx-mask": y["tx"]["mask"],

                    "tx_uncond-x": y["tx_uncond"]["x"],
                    "tx_uncond-length": y["tx_uncond"]["length"],
                    "tx_uncond-mask": y["tx_uncond"]["mask"],
                    "tx_uncond-output": output_uncond.detach().cpu(),

                }

                results = {k: v.detach().to("cpu") if isinstance(v, torch.Tensor) else v for k, v in results.items()}

            x_start = self.motion_normalizer.inverse(x_start)

            return x_start, results

        else:

            xt_pred, _, mean, sigma = self.p_sample_split(xt, y, t, infos["guidance_weight"], output_uncond)
            xt_pred = masked(xt_pred, y["mask"])
            log_likelihood = self.log_likelihood(A, mean, sigma)
            log_likelihood = nan_masked(log_likelihood, y["mask"])
            log_probs = log_likelihood.nanmean(dim=[1, 2])

            return log_probs, xt_pred