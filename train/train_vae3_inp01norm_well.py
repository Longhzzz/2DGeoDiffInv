import argparse
import copy
import json
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Sequence

os.environ["WANDB_MODE"] = "disabled"

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_tensorboard_available, is_wandb_available
from packaging import version
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import opt
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed

from mydatasets import create_dataset
from vae2d import SimpleConditionalDenoisingVAE

if is_wandb_available():
    import wandb

    wandb.init(mode="disabled")

logger = get_logger(__name__, log_level="INFO")


def validate_seismic_directories(seismic_source_dir, train_data_dir):
    """Validate source and output paths before allowing old 2D seismic data to be cleared."""
    if seismic_source_dir is None:
        raise ValueError("Specify --seismic_source_dir when seismic refresh is enabled.")
    source = Path(seismic_source_dir).expanduser().resolve()
    output = Path(train_data_dir).expanduser().resolve()
    if not source.is_dir() or not any(p.is_file() for p in source.glob("*.npz")):
        raise ValueError("The seismic source directory must exist and contain raw .npz files.")
    if source == output:
        raise ValueError("The raw NPZ source and 2D training output directories must differ.")
    return source, output


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple conditional denoising VAE baseline.")
    parser.add_argument("--dataset_name", type=str, default="Seis")
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument(
        "--seismic_source_dir", "--seismic-source-dir", type=str, default=None,
        help="Raw impedance NPZ directory used when refreshing the 2D seismic dataset.",
    )
    parser.add_argument(
        "--validation_data_dir",
        type=str,
        default=None,
        help="Optional fixed validation directory when reset_dataset only rebuilds training prefixes.",
    )
    parser.add_argument("--output_dir", type=str, default="output/vae3_inp01norm_well")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--block_out_channels", type=str, default="128,128,256,256,512,512,512")
    parser.add_argument("--layers_per_block", type=int, default=3)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--channel_multipliers", type=str, default=None)
    parser.add_argument("--recon_loss", type=str, default="l1", choices=["l1", "mse", "l1_mse"])
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--kl_warmup_steps", type=int, default=2000)
    parser.add_argument("--noise_std_min", type=float, default=0.0)
    parser.add_argument("--noise_std_max", type=float, default=0.3)
    parser.add_argument(
        "--noise_sampling_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "stratified"],
        help="Sample one Gaussian noise standard deviation per image.",
    )
    parser.add_argument(
        "--noise_strata_edges",
        type=str,
        default="0.0,0.1,0.4,0.6",
        help="Comma-separated bin edges used by stratified noise sampling.",
    )
    parser.add_argument(
        "--noise_strata_probs",
        type=str,
        default="0.25,0.5,0.25",
        help="Comma-separated probabilities for the stratified noise bins.",
    )
    parser.add_argument(
        "--noise_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append sigma/noise_std_max as a spatial condition channel.",
    )
    parser.add_argument(
        "--well_mask_conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the well mask as a condition channel; disable to match the diffusion ControlNet inputs.",
    )
    parser.add_argument(
        "--clamp_noisy_input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clamp corrupted inputs to [-1, 1]. Keep disabled for Gaussian denoising.",
    )
    parser.add_argument(
        "--validation_noise_std",
        type=float,
        default=None,
        help="Fixed validation noise level; defaults to noise_std_max.",
    )
    parser.add_argument("--min_wells", type=int, default=1)
    parser.add_argument("--max_wells", type=int, default=5)
    parser.add_argument("--well_interval", type=int, default=1)
    parser.add_argument("--well_drop_prob", type=float, default=0.1)
    parser.add_argument("--well_loss_weight", type=float, default=0.0)
    parser.add_argument("--save_png_steps", type=int, default=500)
    parser.add_argument("--loss_plot_steps", type=int, default=100)
    parser.add_argument(
        "--refresh_seismic_before_training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete/rebuild the seismic dataset before constructing the initial DataLoader.",
    )
    parser.add_argument(
        "--refresh_seismic_each_epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete/rebuild training seismic files from epoch 1 onward, matching diffusion training.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify --train_data_dir.")
    if args.refresh_seismic_before_training or args.refresh_seismic_each_epoch:
        # Validate the actual refresh paths before constructing the model or rewriting data.
        validate_seismic_directories(args.seismic_source_dir, args.train_data_dir)

    if args.noise_std_min < 0 or args.noise_std_max < args.noise_std_min:
        raise ValueError(
            f"Invalid noise range [{args.noise_std_min}, {args.noise_std_max}]."
        )
    args.noise_strata_edges = _parse_float_list(args.noise_strata_edges)
    args.noise_strata_probs = _parse_float_list(args.noise_strata_probs)
    if args.noise_sampling_strategy == "stratified":
        _validate_noise_strata(
            args.noise_strata_edges,
            args.noise_strata_probs,
            args.noise_std_min,
            args.noise_std_max,
        )
    if args.validation_noise_std is None:
        args.validation_noise_std = args.noise_std_max
    if args.validation_noise_std < 0:
        raise ValueError("validation_noise_std must be non-negative.")
    return args


def _parse_float_list(values: str | Sequence[float]) -> list[float]:
    if isinstance(values, str):
        return [float(item.strip()) for item in values.split(",") if item.strip()]
    return [float(item) for item in values]


def _validate_noise_strata(edges, probabilities, noise_std_min, noise_std_max):
    if len(edges) != len(probabilities) + 1:
        raise ValueError("noise_strata_edges must contain exactly one more value than noise_strata_probs.")
    if any(right <= left for left, right in zip(edges[:-1], edges[1:])):
        raise ValueError("noise_strata_edges must be strictly increasing.")
    if abs(edges[0] - noise_std_min) > 1e-8 or abs(edges[-1] - noise_std_max) > 1e-8:
        raise ValueError(
            "For stratified sampling, the first/last edges must equal noise_std_min/noise_std_max."
        )
    if any(value < 0 for value in probabilities) or sum(probabilities) <= 0:
        raise ValueError("noise_strata_probs must be non-negative and have a positive sum.")


def copy_files(save_path: str):
    code_dir = os.path.join(save_path, "code")
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy(__file__, os.path.join(code_dir, os.path.basename(__file__)))
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "vae2d.py"),
        os.path.join(code_dir, "vae2d.py"),
    )


def remove_old_checkpoints(output_dir: str, checkpoints_total_limit: int | None):
    if checkpoints_total_limit is None or not os.path.isdir(output_dir):
        return
    checkpoints = os.listdir(output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    if len(checkpoints) < checkpoints_total_limit:
        return
    num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
    for checkpoint in checkpoints[:num_to_remove]:
        shutil.rmtree(os.path.join(output_dir, checkpoint))


def build_random_wells(
    clean_images: torch.Tensor,
    min_num: int,
    max_num: int,
    interval: int,
    generator: torch.Generator | None = None,
):
    batch_size, channels, height, width_dim = clean_images.shape
    well_images = torch.zeros_like(clean_images)
    well_mask = torch.zeros_like(clean_images)

    stripe_width = 2 * interval + 1
    max_start = max(height - stripe_width, 0)
    num_slots = max(height // stripe_width - 1, 1)
    max_num = min(max_num, num_slots)
    min_num = min(min_num, max_num)
    if max_num <= 0:
        return well_images, well_mask

    random_values = torch.rand(
        batch_size, num_slots, device=clean_images.device, generator=generator
    )
    slot_ids = random_values.argsort(dim=1)[:, :max_num]
    offsets = torch.randint(
        0,
        stripe_width,
        (batch_size, max_num),
        device=clean_images.device,
        generator=generator,
    )
    starts = (slot_ids * stripe_width + offsets).clamp(0, max_start).long()
    num_wells = torch.randint(
        min_num,
        max_num + 1,
        (batch_size,),
        device=clean_images.device,
        generator=generator,
    )
    local_offsets = torch.arange(stripe_width, device=clean_images.device)

    for index in range(batch_size):
        n_item = int(num_wells[index].item())
        if n_item == 0:
            continue
        pos = (starts[index, :n_item].unsqueeze(1) + local_offsets).flatten()
        selected = clean_images[index, :, pos, :]
        selected = selected.reshape(channels, n_item, stripe_width, width_dim)
        selected = selected.mean(2, keepdim=True).repeat(1, 1, stripe_width, 1)
        well_images[index, :, pos, :] = selected.reshape(channels, -1, width_dim)
        well_mask[index, :, pos, :] = 1

    return well_images, well_mask


def sample_noise_std(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    noise_std_min: float,
    noise_std_max: float,
    strategy: str = "uniform",
    strata_edges: Sequence[float] | None = None,
    strata_probs: Sequence[float] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    shape = (batch_size, 1, 1, 1)
    if noise_std_max <= 0:
        return torch.zeros(shape, device=device, dtype=dtype)
    if strategy == "uniform":
        return torch.empty(shape, device=device, dtype=dtype).uniform_(
            noise_std_min, noise_std_max, generator=generator
        )
    if strategy != "stratified":
        raise ValueError(f"Unsupported noise sampling strategy: {strategy}")

    edges = torch.as_tensor(strata_edges, device=device, dtype=dtype)
    probabilities = torch.as_tensor(strata_probs, device=device, dtype=torch.float32)
    probabilities = probabilities / probabilities.sum()
    bin_ids = torch.multinomial(
        probabilities,
        num_samples=batch_size,
        replacement=True,
        generator=generator,
    )
    lower = edges[bin_ids].view(shape)
    upper = edges[bin_ids + 1].view(shape)
    unit_uniform = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    return lower + (upper - lower) * unit_uniform


def add_input_noise(
    clean_images: torch.Tensor,
    noise_std_min: float,
    noise_std_max: float,
    strategy: str = "uniform",
    strata_edges: Sequence[float] | None = None,
    strata_probs: Sequence[float] | None = None,
    fixed_noise_std: float | None = None,
    clamp_noisy_input: bool = False,
    generator: torch.Generator | None = None,
):
    if fixed_noise_std is None:
        noise_std = sample_noise_std(
            clean_images.shape[0],
            clean_images.device,
            clean_images.dtype,
            noise_std_min,
            noise_std_max,
            strategy=strategy,
            strata_edges=strata_edges,
            strata_probs=strata_probs,
            generator=generator,
        )
    else:
        noise_std = torch.full(
            (clean_images.shape[0], 1, 1, 1),
            float(fixed_noise_std),
            device=clean_images.device,
            dtype=clean_images.dtype,
        )
    noise = torch.randn(
        clean_images.shape,
        device=clean_images.device,
        dtype=clean_images.dtype,
        generator=generator,
    )
    signal_scale = torch.sqrt(torch.clamp(1.0 - noise_std.square(), min=0.0))
    noisy_images = signal_scale * clean_images + noise_std * noise
    if clamp_noisy_input:
        noisy_images = noisy_images.clamp(-1, 1)
    return noisy_images, noise_std


def build_noise_condition(
    noise_std: torch.Tensor,
    spatial_shape: Sequence[int],
    noise_condition_scale: float,
) -> torch.Tensor:
    if noise_condition_scale <= 0:
        raise ValueError("noise_condition_scale must be positive.")
    height, width = spatial_shape
    return (noise_std / noise_condition_scale).expand(-1, 1, height, width)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str):
    pred = pred.float()
    target = target.float()
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    return F.l1_loss(pred, target) + F.mse_loss(pred, target)


def init_loss_dict():
    return {
        "loss": [],
        "recon": [],
        "kl": [],
        "beta": [],
        "well_loss": [],
        "noise_std": [],
        "n_step": 0,
        "accu": {
            "loss": 0.0,
            "recon": 0.0,
            "kl": 0.0,
            "beta": 0.0,
            "well_loss": 0.0,
            "noise_std": 0.0,
        },
    }


def normalize_loss_dict(loss_dic):
    default = init_loss_dict()
    if not isinstance(loss_dic, dict):
        return default
    for key in ["loss", "recon", "kl", "beta", "well_loss", "noise_std"]:
        loss_dic.setdefault(key, [])
    loss_dic.setdefault("n_step", 0)
    loss_dic.setdefault("accu", default["accu"].copy())
    if "accu_loss" in loss_dic:
        loss_dic["accu"]["loss"] = loss_dic.pop("accu_loss")
    for key, value in default["accu"].items():
        loss_dic["accu"].setdefault(key, value)
    return loss_dic


def update_loss_dict(loss_dic, logs):
    for key in ["loss", "recon", "kl", "beta", "well_loss", "noise_std"]:
        loss_dic["accu"][key] += float(logs[key])


def flush_loss_dict(loss_dic, interval):
    for key in ["loss", "recon", "kl", "beta", "well_loss", "noise_std"]:
        loss_dic[key].append(loss_dic["accu"][key] / interval)
        loss_dic["accu"][key] = 0.0


def save_loss_plots(loss_dic, save_path):
    plot_keys = ["loss", "recon", "kl", "beta", "well_loss", "noise_std"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, key in zip(axes, plot_keys):
        ax.plot(loss_dic.get(key, []))
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


@torch.no_grad()
def sample_validation(model, val_item, args, device, generator=None):
    dtype = next(model.parameters()).dtype
    clean_images = val_item["image"].unsqueeze(0).repeat(args.eval_batch_size, 1, 1, 1).to(device=device, dtype=dtype)
    cond_images = val_item["mask"].unsqueeze(0).repeat(args.eval_batch_size, 1, 1, 1).to(device=device, dtype=dtype)
    well_image_single, well_mask_single = build_random_wells(
        clean_images[:1],
        min_num=args.min_wells,
        max_num=args.max_wells,
        interval=args.well_interval,
        generator=generator,
    )
    well_images = well_image_single.repeat(args.eval_batch_size, 1, 1, 1)
    well_mask = well_mask_single.repeat(args.eval_batch_size, 1, 1, 1)
    noisy_images, noise_std = add_input_noise(
        clean_images,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        fixed_noise_std=args.validation_noise_std,
        clamp_noisy_input=args.clamp_noisy_input,
        generator=generator,
    )
    condition_parts = [cond_images, well_images]
    if args.well_mask_conditioning:
        condition_parts.append(well_mask)
    if args.noise_conditioning:
        condition_parts.append(
            build_noise_condition(
                noise_std,
                clean_images.shape[-2:],
                noise_condition_scale=max(args.noise_std_max, 1e-8),
            )
        )
    condition = torch.cat(condition_parts, dim=1)
    samples = model.predict_clean(noisy_images, condition, use_posterior_mean=True)
    return samples, clean_images, cond_images, well_images, well_mask, noisy_images, noise_std


@torch.no_grad()
def quick_save_validation(
    samples,
    target,
    cond_images,
    well_images,
    well_mask,
    noisy_images,
    noise_std,
    save_dir,
    n_step,
):
    os.makedirs(save_dir, exist_ok=True)
    samples = samples.detach().float().cpu()
    target = target.detach().float().cpu()
    cond_images = cond_images.detach().float().cpu()
    well_images = well_images.detach().float().cpu()
    well_mask = well_mask.detach().float().cpu()
    noisy_images = noisy_images.detach().float().cpu()
    noise_std = noise_std.detach().float().cpu()

    num_samples = min(samples.shape[0], 4)
    plt.figure(figsize=(13, 25))
    plt.subplot(5, 2, 1)
    plt.imshow(cond_images[0, 0].T, cmap="seismic", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("seismic")

    plt.subplot(5, 2, 2)
    plt.imshow(target[0, 0].T, cmap="jet", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("target")

    plt.subplot(5, 2, 3)
    plt.imshow(well_images[0, 0].T, cmap="jet", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("well")

    plt.subplot(5, 2, 4)
    plt.imshow(well_mask[0, 0].T, cmap="gray", vmin=0, vmax=1)
    plt.colorbar()
    plt.title("well mask")

    plt.subplot(5, 2, 5)
    noisy_limit = max(1.0, float(noisy_images[0, 0].abs().max()))
    plt.imshow(noisy_images[0, 0].T, cmap="jet", vmin=-noisy_limit, vmax=noisy_limit)
    plt.colorbar()
    plt.title(f"noisy input, sigma={float(noise_std[0]):.3f}")

    plt.subplot(5, 2, 6)
    row_ids = torch.where(well_mask[0, 0].sum(dim=-1) > 0)[0]
    row_index = int(row_ids[len(row_ids) // 2].item()) if len(row_ids) > 0 else target.shape[2] // 2
    plt.plot(samples[:, 0, row_index, :].T)
    plt.plot(target[0, 0, row_index, :], "--", color="black", linewidth=1)
    if len(row_ids) > 0:
        plt.plot(well_images[0, 0, row_index, :], ":", color="red", linewidth=1)
    plt.title(f"line {row_index}")

    for idx in range(num_samples):
        plt.subplot(5, 2, 7 + idx)
        plt.imshow(samples[idx, 0].T, cmap="jet", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"posterior-mean denoising {idx}")

    plt.savefig(f"{save_dir}/inference_{n_step}.png", bbox_inches="tight")
    plt.close()


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[init_kwargs, ddp_kwargs],
    )
    if args.logger == "tensorboard" and not is_tensorboard_available():
        raise ImportError("Make sure to install tensorboard if you want to use it for logging.")
    if args.logger == "wandb" and not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging.")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "vae"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                load_model = SimpleConditionalDenoisingVAE.from_pretrained(
                    os.path.join(input_dir, "vae"),
                    map_location="cpu",
                )
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            if args.overwrite_output_dir and os.path.isdir(args.output_dir):
                shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2)

    if args.refresh_seismic_before_training:
        reset_dataset(
            accelerator, accelerator.device, args.seismic_source_dir,
            args.train_data_dir, initial=True,
        )

    model = SimpleConditionalDenoisingVAE(
        sample_size=args.resolution,
        in_channels=1,
        condition_channels=2 + int(args.well_mask_conditioning) + int(args.noise_conditioning),
        out_channels=1,
        latent_dim=args.latent_dim,
        block_out_channels=args.block_out_channels,
        layers_per_block=args.layers_per_block,
        base_channels=args.base_channels,
        channel_multipliers=args.channel_multipliers,
        noise_condition_scale=max(args.noise_std_max, 1e-8),
    )
    print(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = create_dataset(args.dataset_name, args)
    validation_args = copy.copy(args)
    if args.validation_data_dir is not None:
        validation_args.train_data_dir = args.validation_data_dir
    val_dataset = create_dataset(args.dataset_name, validation_args, split="val")
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=math.ceil(len(train_dataloader) * args.num_epochs / args.gradient_accumulation_steps),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running VAE baseline training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(
        "  Input noise = %s VP-style, range=[%.3f, %.3f], clamp=%s, noise_conditioning=%s, well_mask_conditioning=%s",
        args.noise_sampling_strategy,
        args.noise_std_min,
        args.noise_std_max,
        args.clamp_noisy_input,
        args.noise_conditioning,
        args.well_mask_conditioning,
    )

    global_step = 0
    first_epoch = 0
    resume_step = 0
    loss_dic = init_loss_dict()

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            loss_path = os.path.join(args.output_dir, path, "loss.json")
            if os.path.exists(loss_path):
                with open(loss_path, "r", encoding="utf-8") as f:
                    loss_dic = json.load(f)
                loss_dic = normalize_loss_dict(loss_dic)
                for key in loss_dic["accu"]:
                    loss_dic["accu"][key] = 0.0
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    for epoch in range(first_epoch, args.num_epochs):
        if args.refresh_seismic_each_epoch and epoch > 0:
            reset_dataset(
                accelerator, accelerator.device, args.seismic_source_dir, args.train_data_dir,
            )
            dataset = create_dataset(args.dataset_name, args)
            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.dataloader_num_workers,
                drop_last=True,
            )
            train_dataloader = accelerator.prepare(train_dataloader)
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["image"].to(weight_dtype)
            cond_images = batch["mask"].to(weight_dtype)
            well_images, well_mask = build_random_wells(
                clean_images,
                min_num=args.min_wells,
                max_num=args.max_wells,
                interval=args.well_interval,
            )

            if args.well_drop_prob > 0:
                keep = (torch.rand(clean_images.shape[0], 1, 1, 1, device=clean_images.device) >= args.well_drop_prob)
                keep = keep.to(dtype=clean_images.dtype)
                well_images = well_images * keep
                well_mask = well_mask * keep

            noisy_images, noise_std = add_input_noise(
                clean_images,
                noise_std_min=args.noise_std_min,
                noise_std_max=args.noise_std_max,
                strategy=args.noise_sampling_strategy,
                strata_edges=args.noise_strata_edges,
                strata_probs=args.noise_strata_probs,
                clamp_noisy_input=args.clamp_noisy_input,
            )
            condition_parts = [cond_images, well_images]
            if args.well_mask_conditioning:
                condition_parts.append(well_mask)
            if args.noise_conditioning:
                condition_parts.append(
                    build_noise_condition(
                        noise_std,
                        clean_images.shape[-2:],
                        noise_condition_scale=max(args.noise_std_max, 1e-8),
                    )
                )
            condition = torch.cat(condition_parts, dim=1)

            # plt.figure(figsize=(15,15))
            # plt.subplot(221)
            # plt.imshow(clean_images[0,0].detach().cpu().T, cmap='jet')
            # plt.title(noise_std[0,0,0,0].detach().cpu())
            # plt.subplot(222)
            # plt.imshow(noisy_images[0,0].detach().cpu().T, cmap='jet')
            # plt.subplot(223)
            # plt.imshow(cond_images[0,0].detach().cpu().T, cmap='jet')
            # plt.subplot(224)
            # plt.imshow(well_images[0,0].detach().cpu().T, cmap='jet')
            # plt.savefig('input.png')
            
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    output = model(noisy_images, condition)
                    pred_images = output["sample"]
                    recon = reconstruction_loss(pred_images, clean_images, args.recon_loss)
                    kl = SimpleConditionalDenoisingVAE.kl_loss(output["mu"].float(), output["logvar"].float())
                    if args.kl_warmup_steps > 0:
                        beta = args.kl_weight * min(1.0, global_step / args.kl_warmup_steps)
                    else:
                        beta = args.kl_weight
                    loss = recon + beta * kl
                    well_loss = torch.zeros((), device=clean_images.device, dtype=loss.dtype)
                    if args.well_loss_weight > 0:
                        well_loss = ((pred_images.float() - well_images.float()) ** 2 * well_mask.float()).sum()
                        well_loss = well_loss / (well_mask.float().sum() + 1e-8)
                        loss = loss + args.well_loss_weight * well_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss_dic["n_step"] += 1
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step == 1:
                        copy_files(args.output_dir)
                    if global_step % args.checkpointing_steps == 0:
                        remove_old_checkpoints(args.output_dir, args.checkpoints_total_limit)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        with open(save_path + "/loss.json", "w", encoding="utf-8") as f:
                            json.dump(loss_dic, f, indent=4)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "recon": recon.detach().item(),
                "kl": kl.detach().item(),
                "beta": beta,
                "well_loss": well_loss.detach().item(),
                "noise_std": noise_std.mean().detach().item(),
                "input_outside_fraction": (noisy_images.detach().abs() > 1).float().mean().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "sum_step": loss_dic["n_step"],
            }
            update_loss_dict(loss_dic, logs)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                if loss_dic["n_step"] % args.loss_plot_steps == 0 and loss_dic["n_step"] > 0:
                    flush_loss_dict(loss_dic, args.loss_plot_steps)
                    save_loss_plots(loss_dic, f"loss_{os.path.basename(args.output_dir)}.png")

                if (loss_dic["n_step"] % args.save_png_steps == 0 or loss_dic["n_step"] == 20) and loss_dic["n_step"] > 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.eval()
                    generator = torch.Generator(device=clean_images.device).manual_seed(0)
                    with accelerator.autocast():
                        (
                            samples,
                            target,
                            cond_val,
                            well_val,
                            well_mask_val,
                            noisy_val,
                            noise_std_val,
                        ) = sample_validation(
                            unwrapped_model,
                            val_dataset[-1],
                            args,
                            clean_images.device,
                            generator=generator,
                        )
                    quick_save_validation(
                        samples,
                        target,
                        cond_val,
                        well_val,
                        well_mask_val,
                        noisy_val,
                        noise_std_val,
                        save_dir=os.path.join("png_inference", os.path.basename(args.output_dir)),
                        n_step=loss_dic["n_step"],
                    )
                    unwrapped_model.train()

        progress_bar.close()
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        remove_old_checkpoints(args.output_dir, args.checkpoints_total_limit)
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        with open(save_path + "/loss.json", "w", encoding="utf-8") as f:
            json.dump(loss_dic, f, indent=4)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()

def process_file(input_path, output_folder, device):
    """
    Process a single file: load, compute, and save
    """
    # Load data.
    # print('Begin:',input_path)

    data = np.load(input_path,allow_pickle=True)
    dt_base = 0.0005
    # dt = np.round(np.random.uniform(0.2, 0.5),2) / 1000
    dt = 0.0005
    f0 = np.round(np.random.uniform(20, 60), 2)
    # f0 = 30
    ricker_wavelet, _ = opt.ricker_torch(f0, dt_base, 3/f0)
    imp = data['imp']
    imp[imp<0] *= -1
    imp_interp = opt.interp_torch(torch.from_numpy(imp).to(device),original_sampling_rate=dt,target_sampling_rate=dt_base)
    # Process data.

    ref = opt.imp2ref_torch(imp_interp, dim=0)
    
    seis = opt.ref2seis_torch(ref, ricker_wavelet.to(device))
    
    
    
    seis = seis.permute(2,1,0)
    imp_interp = imp_interp.permute(2,1,0)
    
    # mean = torch.FloatTensor(1).uniform_(-0.01, 0.01)[0]
    mean = 0
    std = torch.FloatTensor(1).uniform_(0.005, 0.03)[0]
    noise = torch.normal(mean, std, size=seis.shape, device=device)
    seis = seis + noise

    nx, ny, nt = imp_interp.shape
    
    # cube_i_mn = imp_interp.min()
    # cube_i_mx = imp_interp.max()
    # seis_i_mn = seis.min()
    # seis_i_mx = seis.max()
    # imp_interp = (imp_interp - cube_i_mn) / (cube_i_mx-cube_i_mn)
    # seis = (seis - seis_i_mn) / (seis_i_mx-seis_i_mn) 
    number = random.choice([0, 1, 2, 3])
    for ix in range(number,nx,4):
        slice_seis = seis[ix].cpu()
        slice_imp = imp_interp[ix].cpu()
        slice_seis_max = slice_seis.max()
        slice_seis_min = slice_seis.min()
        slice_imp_max = slice_imp.max()
        slice_imp_min = slice_imp.min()
        
        slice_seis = (slice_seis - slice_seis_min) / (slice_seis_max - slice_seis_min)
        slice_imp = (slice_imp - slice_imp_min) / (slice_imp_max - slice_imp_min)
        dic = {
                'seis':slice_seis.to(torch.float16),
                'imp':slice_imp.to(torch.float16),
            }
        torch.save(dic, Path(output_folder) / f'{os.path.basename(input_path).split(".")[0]}_{slice_imp_min:.2f}_{slice_imp_max:.2f}_{slice_seis_min:.2f}_{slice_seis_max:.2f}_{dt*1000}_{f0}_ix-{ix}.pth')
    
    for iy in range(number,ny,4):
        slice_seis = seis[:,iy,:].cpu()
        slice_imp = imp_interp[:,iy,:].cpu()
        slice_seis_max = slice_seis.max()
        slice_seis_min = slice_seis.min()
        slice_imp_max = slice_imp.max()
        slice_imp_min = slice_imp.min()
        
        slice_seis = (slice_seis - slice_seis_min) / (slice_seis_max - slice_seis_min)
        slice_imp = (slice_imp - slice_imp_min) / (slice_imp_max - slice_imp_min)
        dic = {
                'seis':slice_seis.to(torch.float16),
                'imp':slice_imp.to(torch.float16),
            }
        torch.save(dic, Path(output_folder) / f'{os.path.basename(input_path).split(".")[0]}_{slice_imp_min:.2f}_{slice_imp_max:.2f}_{slice_seis_min:.2f}_{slice_seis_max:.2f}_{dt*1000}_{f0}_iy-{iy}.pth')

def reset_dataset(accelerator, device, seismic_source_dir, train_data_dir, initial=False):
    process_index = accelerator.process_index
    num_processes = accelerator.num_processes
    # Use only user-specified paths and validate them before creating or clearing output directories.
    input_folder, output_folder = validate_seismic_directories(seismic_source_dir, train_data_dir)
    validation_ids = range(950, 1000)  # Match the training/validation split used by SeisDataset.
    npz_files = sorted(
        str(path) for path in input_folder.iterdir()
        if path.is_file() and path.suffix == ".npz"
    )
    if not initial:
        # Filter by numeric IDs so a subset does not treat validation files as the first 950 training samples.
        npz_files = [
            path for path in npz_files
            if int(Path(path).stem.split("_", 1)[0]) not in validation_ids
        ]
        if not npz_files:
            raise ValueError("No training NPZ files remain after excluding validation IDs 950-999.")

    os.makedirs(output_folder, exist_ok=True)
    if process_index == 0:
        print('Delete original data')
        for path in output_folder.iterdir():
            if not path.is_file() or path.suffix != ".pth":
                continue
            try:
                sample_id = int(path.stem.split("_", 1)[0])
            except ValueError:
                continue
            # Rebuild all slices at initialization; refresh only training data in later epochs, preserving validation paths and contents.
            if initial or sample_id not in validation_ids:
                path.unlink()
    accelerator.wait_for_everyone()
    print('len(os.listdir(output_folder)):', len(os.listdir(output_folder)))
    print()

    # Split evenly across the total number of processes.
    split_files = np.array_split(npz_files, num_processes)
    local_npz_files = split_files[process_index].tolist()

    print(
        f'[Process {process_index}/{num_processes}] Files to process: {len(local_npz_files)}'
    )

    if local_npz_files:
        print(
            f'[Process {process_index}] File range: {os.path.basename(local_npz_files[0])} -> {os.path.basename(local_npz_files[-1])}'
        )
        
    # Use a thread pool for parallel processing within each process.
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                process_file,
                input_path,
                output_folder,
                device,
            )
            for input_path in local_npz_files
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Process {process_index}",
            position=process_index,
        ):
            try:
                future.result()
            except Exception as e:
                print(f'[Process {process_index}] File processing failed: {e}')

    # Wait for all processes to finish.
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("All workers have finished processing.")

if __name__ == "__main__":
    main(parse_args())
