import torch

from TMR.src.data.collate import collate_x_dict
from TMR.src.config import read_config
from TMR.src.load import load_model_from_cfg
from hydra.utils import instantiate


def load_tmr_model_easy(device="cpu", dataset="humanml3d"):
    if dataset == "humanml3d":
        run_dir = "TMR/models/tmr_humanml3d_guoh3dfeats"
    elif dataset == "kitml":
        run_dir = "TMR/models/tmr_kitml_guoh3dfeats"
    elif dataset == "humanml3d_kitml_augmented_and_hn":
        run_dir = "TMR/models/tmr_humanml3d_kitml_augmented_and_hn"
    elif dataset == "tmr_humanml3d_kitml_guoh3dfeats":
        run_dir = "TMR/models/tmr_humanml3d_kitml_guoh3dfeats"

    ckpt_name = "last"
    cfg = read_config(run_dir)

    print("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    normalizer = instantiate(cfg.data.motion_loader.normalizer)

    print("Loading the text model")
    text_model = instantiate(cfg.data.text_to_token_emb, preload=False, device=device)

    def easy_forward(motions_or_texts):
        if isinstance(motions_or_texts[0], str):
            texts = motions_or_texts
            x_dict = collate_x_dict(text_model(texts))
        else:
            motions = motions_or_texts
            motions = [
                normalizer(torch.from_numpy(motion).to(torch.float)).to(device)
                for motion in motions
            ]
            x_dict = collate_x_dict(
                [
                    {
                        "x": motion,
                        "length": len(motion),
                    }
                    for motion in motions
                ]
            )

        with torch.inference_mode():
            latents = model.encode(x_dict, sample_mean=True).cpu()
        return latents

    return easy_forward
