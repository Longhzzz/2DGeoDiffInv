import argparse
import inspect
import logging
import math
import os
os.environ["WANDB_MODE"] = "disabled"
import shutil
from datetime import timedelta
from pathlib import Path
import random
import numpy as np
from accelerate.utils import DistributedDataParallelKwargs
from shutil import copy
import time

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from mydatasets import create_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json
from torchvision.transforms import GaussianBlur
import kornia
from torchmetrics import StructuralSimilarityIndexMeasure
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import opt
from scipy.signal import hilbert, find_peaks, windows
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel, ControlNetModel, MultiControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
# from scipy.interpolate import griddata
from scipy.interpolate import Rbf


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")
if is_wandb_available():
    import wandb
wandb.init(mode="disabled")
logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=1, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=1, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample", "v_prediction"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def imp2ref_torch(x, dim=-1):
    """
    Seismic impedance to reflectivity.
    x: impedance, tensor (can be 1D, 2D, 3D, or more dimensions)
    dim: the dimension along which reflectivity is calculated
    """
    diff = torch.diff(x, dim=dim,)
    avg = (x + torch.roll(x, shifts=-1, dims=dim))
    ref = torch.zeros_like(x)
    
    slices = [slice(None)] * x.ndimension()
    slices[dim] = slice(0, -1)
    ref[tuple(slices)] = diff / (avg[tuple(slices)] + 1e-6)
    return ref


def ref2seis_torch(x, wavelet):
    """
    Reflectivity to seismic data.

    x: tensor-like (can be 1D/2D/3D tensor)
    wavelet: 1D tensor (wavelet function)
    dim: the dimension of time/depth to apply convolution
    """
    
    if isinstance(x, torch.Tensor):
        wavelet = wavelet.to(x.device)  # Ensure wavelet is on the same device as x
        x = x.to(torch.float32)  # Ensure input is float32
        # Unsqueeze wavelet to match the format for conv1d: (out_channels, in_channels, kernel_size)
        wavelet = wavelet.unsqueeze(1).to(torch.float32) 
        # x: 2,512,512
        # We need to permute the dimensions so the convolution happens on the `dim` dimension
        x = x.permute(1,0,2)
        # Perform 1D convolution to simulate seismic data generation
        out = F.conv1d(x, wavelet, padding=wavelet.shape[2] // 2, groups=x.shape[1])
        out = out.permute(1,0,2)
        

        return out
    else:
        raise ValueError("Unsupported data type")
    
def _get_time_torch(duration: float, dt: float, tc: float, sym: bool = True):
    """
    Helper function to generate time series.
    """
    n = int((duration / dt).max())
    odd = n % 2
    if (odd and sym):
        t = torch.arange(n)
    elif (not odd and sym):
        t = torch.arange(n + 1, )
    elif (odd and not sym):
        t = torch.arange(n, )
    elif (not odd and not sym):
        t = torch.arange(n, ) - 1
    t -= t[-1] // 2
    t  = t.unsqueeze(0).repeat(dt.shape[0],1).to(duration.device)
    return t * dt.unsqueeze(1) + tc
def ricker_torch(f, dt, duration, sym=True):
    t = _get_time_torch(duration, dt, 0, sym)  # Get the time axis
    
    # Reshape f and t for broadcasting
    f = f.view(-1, 1)  # Shape: (n_frequencies, 1)
    # t = t.reshape(1, -1)  # Shape: (1, n_time_samples)
    
    # Compute the Ricker wavelet for each frequency
    w = (1 - 2 * (torch.pi * f * t) ** 2) * torch.exp(-(torch.pi * f * t) ** 2)
    
    return w, t.squeeze()

def forward(imp, f0, dt, dim4=False):
    # imp: bs * 256 * 256 , f0: (bs, ), dt: (bs, )
    if len(imp.shape) == 4:
        # imp: bs * C * 256 * 256 , f0: (bs, ), dt: (bs, )
        B, C, H, W = imp.shape
        imp = imp.view(-1, H, W)
        f0 = f0.unsqueeze(1).repeat(1,C).view(-1)
        dt = dt.unsqueeze(1).repeat(1,C).view(-1) 
        dim4 = True     
    ricker_wavelet, _ = ricker_torch(f0, dt, 3/f0)
    ref = imp2ref_torch(imp, dim=2)
    seis = ref2seis_torch(ref, ricker_wavelet.to(imp.device))
    if dim4:
        seis = seis.view(B, C, H, W)
    return seis


@torch.no_grad()
def inference(
              min_num,
              max_num,
              unet,
              controlnet,
              interval = 1,
              control_guidance_start=0.0,
              control_guidance_end=1.0,
              controlnet_conditioning_scale=1.0,
              scheduler=None,
              generator=None,
              num_inference_steps=None,
              device=None,
              cond_dataset=None,
              kernels=None):
    # Adjust the control-guidance timing parameters.
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [control_guidance_end]
        
    image_shape = (
                4,
                unet.config.in_channels,
                unet.config.sample_size,
                unet.config.sample_size,
            )
    if cond_dataset is not None:
        f0 = torch.tensor(cond_dataset["f0"]).view(-1).repeat(image_shape[0])
        dt = torch.tensor(cond_dataset["dt"]).view(-1).repeat(image_shape[0])
        cond_images=cond_dataset['mask'].unsqueeze(0).repeat(image_shape[0],1,1,1).to(device)
        clean_images=cond_dataset['image'].unsqueeze(0).repeat(image_shape[0],1,1,1).to(device)
        # well_images=cond_dataset['well'].unsqueeze(0).repeat(image_shape[0],1,1,1).to(device)
        # interval = 1
        # max_num = max(5, int((global_step / max_train_steps) * 50))
        # min_num = max(0, int(max_num - 10))
        width = 2 * interval + 1
        rdm = torch.randint(0, clean_images.shape[2], (clean_images.shape[0], 1 + max_num,))
        # rdm2 = torch.randint(0, (image.shape[1] // width) - 1, (max_num,))
        random_values = torch.rand(clean_images.shape[0], (clean_images.shape[2] // width) - 1)
        rdm2 = random_values.argsort(dim=1)[:, :max_num]
        N = (rdm[:,0] / (clean_images.shape[2] / ((max_num - min_num) + 1))).int() + min_num
        F = (rdm[:,1:] / (clean_images.shape[2] / width)).int()
        pos = (rdm2 * width + F).clamp(0, clean_images.shape[2]).int()
        pos2 = torch.randint(0, clean_images.shape[-1], (clean_images.shape[0], max_num,))
        pos3 = torch.randint(64, clean_images.shape[2], (clean_images.shape[0], max_num,))
        well_images = torch.zeros_like(clean_images, device=clean_images.device)
        mask = torch.zeros_like(clean_images, device=clean_images.device).float()
        
        for ind in range(clean_images.shape[0]):
            pos_f = (pos[0,:N[0]].unsqueeze(1) + torch.arange(width)).flatten()
            pos_f2 = (pos2[0,:N[0]].unsqueeze(1).repeat(1, width)).flatten()
            pos_f3 = (pos3[0,:N[0]].unsqueeze(1).repeat(1, width)).flatten()
            well_images[ind, :, pos_f] = clean_images[ind, :, pos_f].clone().view(clean_images.shape[1], N[0], width, clean_images.shape[-1]).mean(2,keepdim=True).repeat(1,1,width,1).view(clean_images.shape[1],-1,clean_images.shape[-1])
            row_indices = torch.cat([torch.arange(max(0, p2-p3), min(p2+p3, clean_images.shape[-1])) for p2, p3 in zip(pos_f2, pos_f3)])
            col_indices = torch.cat([torch.full((min(p2+p3, clean_images.shape[-1]) - max(0, p2-p3),), p1, dtype=torch.long) for p1, p2, p3 in zip(pos_f, pos_f2, pos_f3)])
            mask[ind, :, col_indices, row_indices] = 1
        
        well_images = well_images * mask
        kernel_size = torch.randint(low=10, high=50, size=(1,))[0].item()
        shift = torch.rand((clean_images.shape[0], 1, 1, 1), dtype=clean_images.dtype, device=clean_images.device) * 0
        initial_well_images = gaussian_smoothing(clean_images, kernel_size=2 * kernel_size + 1, sigma=kernel_size / 3).to(clean_images.device) + shift
        # well_images = torch.cat([well_images, initial_well_images], dim=1)
        
        imp_min, imp_max = torch.tensor(cond_dataset["imp_min"]).view(-1, 1, 1, 1).repeat(image_shape[0],1,1,1).to(device), torch.tensor(cond_dataset["imp_max"]).view(-1, 1, 1, 1).repeat(image_shape[0],1,1,1).to(device)
        seis_min, seis_max = torch.tensor(cond_dataset["seis_min"]).view(-1, 1, 1, 1).repeat(image_shape[0],1,1,1).to(device), torch.tensor(cond_dataset["seis_max"]).view(-1, 1, 1, 1).repeat(image_shape[0],1,1,1).to(device)
        # norm to truely
        cond_images_denorm = cond_images * 0.5 + 0.5
        cond_images_denorm = cond_images_denorm * (seis_max - seis_min) + seis_min
        # cond_images_norm = mean_filter(cond_images_norm, kernel_size=3)
    
    rand_float = torch.rand(1).item()
    well_images = well_images if rand_float<1 else (torch.zeros_like(well_images, device=clean_images.device) )
    scale_well = torch.ones((image_shape[0],1,1,1)).to(device) if rand_float<1 else torch.zeros((image_shape[0],1,1,1)).to(device)
    # scale_well = torch.ones((image_shape[0],1,1,1)).to(device)
      
    image = randn_tensor(image_shape, generator=generator, device=device, dtype=unet.dtype)
    scheduler.set_timesteps(num_inference_steps)
    noise_weight = (1 - scheduler.alphas_cumprod) ** 0.5
    
    controlnet_keep = []
    for i in range(len(scheduler.timesteps)):
        keeps = [
            1.0 - float(i / len(scheduler.timesteps) < s or (i + 1) / len(scheduler.timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps) 
    i = 0
    image_pred_ori = randn_tensor(image_shape, generator=generator, device=device, dtype=unet.dtype)
    
    for t in scheduler.timesteps:
        image = scheduler.scale_model_input(image, t)
        
        if isinstance(controlnet_keep[i], list):
            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
        else:
            cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
        
        # scales = (t < 1501).float().reshape(1, 1, 1, 1).repeat(image.shape[0], 1, 1, 1).to(image.device)
        
        down_samples, mid_sample = controlnet(
                image,
                t,
                encoder_hidden_states=None,  # Disable text conditioning.
                controlnet_cond=[cond_images, well_images],
                conditioning_scale=[torch.ones((image_shape[0],1,1,1)).to(image.device), 
                                    scale_well],
                guess_mode=False,
                return_dict=False,
            )
        
        # 1. predict noise model_output
        model_output = unet(
                image,
                t, 
                None,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
            ).sample
        

        # 2. compute previous image: x_t -> x_t-1
        image = scheduler.step(model_output, t, image, generator=generator).prev_sample
        i += 1
    # image = (image / 2 + 0.5).clamp(0, 1)
    image = image.permute(0, 2, 3, 1)
    return image, well_images

@torch.no_grad()
def quick_save_4images(images, well_images, val_dataset, save_dir, n_step):
    f0 = torch.tensor(val_dataset["f0"]).view(-1).repeat(images.shape[0]).to(images.device)
    dt = torch.tensor(val_dataset["dt"]).view(-1).repeat(images.shape[0]).to(images.device)
    cond_image=val_dataset['mask'].to(images.device)
    real_image=val_dataset['image'].to(images.device)
    well_image= well_images.to(images.device)
    imp_min, imp_max = torch.tensor(val_dataset["imp_min"]).to(images.device), torch.tensor(val_dataset["imp_max"]).to(images.device)
    seis_min, seis_max = torch.tensor(val_dataset["seis_min"]).to(images.device), torch.tensor(val_dataset["seis_max"]).to(images.device)
    """Simplified helper for quickly saving four images."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 35))
    plt.subplot(7, 2, 1)
    plt.imshow(cond_image[0,:,:].T.cpu(), cmap='seismic', vmin=-1, vmax=1)
    plt.colorbar()
    plt.subplot(7, 2, 2)
    plt.imshow(real_image[0,:,:].T.cpu(), cmap='jet', vmin=-1, vmax=1)
    plt.colorbar()
    plt.subplot(7, 2, 3)
    plt.imshow(well_image[0,0,:,:].T.cpu(), cmap='jet', vmin=-1, vmax=1)
    plt.colorbar()
    if well_image.shape[1]>1:
        plt.subplot(7, 2, 4)
        plt.imshow(well_image[0,1,:,:].T.cpu(), cmap='jet', vmin=-1, vmax=1)
        plt.colorbar()
    for i in range(4):
        plt.subplot(7, 2, 4+i+1)
        plt.imshow(images[i,:,:,0].T.cpu(), cmap='jet', vmin=-1, vmax=1)
        plt.colorbar()
        
    image_denorm = images * 0.5 + 0.5
    image_denorm = image_denorm * (imp_max - imp_min) + imp_min
    seis_forward = forward(image_denorm[:,:,:,0], f0, dt)
    seis_forward = (seis_forward - seis_min) / (seis_max - seis_min)
    seis_forward = (seis_forward - 0.5) / 0.5
    for i in range(4):
        plt.subplot(7, 2, 8+i+1)
        plt.imshow(seis_forward[i,:,:].T.cpu(), cmap='seismic', vmin=-1, vmax=1)
        plt.colorbar()
        
    if well_image.max() > 0 + 1e-4:
        index = torch.where((well_image[0,0].sum(dim=-1)!=0*well_image.shape[-1]).cpu())[0].sort().values[1].item()
    else:
        index = 100
    plt.subplot(7, 2, 13)
    plt.plot(seis_forward[:,index,:].T.cpu())
    plt.plot(cond_image[0,index,:].cpu(),'--')
    plt.subplot(7, 2, 14)
    plt.plot(images[:,index,:,0].T.cpu())
    plt.plot(well_image[0,0,index,:].cpu(),'--')
    plt.savefig(f'{save_dir}/inference_{n_step}.png', bbox_inches='tight')
    plt.close()
    
def main(args):

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # 1. Configure DDP find_unused_parameters.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    # 2. Set the process-group initialization timeout.
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))

    # 3. Initialize Accelerator.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[init_kwargs, ddp_kwargs],  # Pass both configuration objects.
    )
    
    # holder = reserve_gpu_memory(81, accelerator.device)
    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks for both UNet and ControlNet models
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    # Save EMA versions if using EMA
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
                    if hasattr(args, 'use_controlnet_ema') and args.use_controlnet_ema:
                        ema_model_controlnet.save_pretrained(os.path.join(output_dir, "controlnet_ema"))

                for i, model in enumerate(models):
                    print(type(model).__name__)  
                    # Determine if the model is UNet or ControlNet
                    if isinstance(model, UNet2DConditionModel):
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif isinstance(model, MultiControlNetModel):
                        model.save_pretrained(os.path.join(output_dir, "controlnet", "controlnet"))
                    
                    weights.pop()
            
                noise_scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))    
                    

        def load_model_hook(models, input_dir):
            if args.use_ema:
                # Load EMA UNet
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
                
                # Load EMA ControlNet if exists
                if hasattr(args, 'use_controlnet_ema') and args.use_controlnet_ema:
                    load_controlnet = EMAModel.from_pretrained(os.path.join(input_dir, "controlnet_ema"), MultiControlNetModel)
                    ema_model_controlnet.load_state_dict(load_controlnet.state_dict())
                    ema_model_controlnet.to(accelerator.device)
                    del load_controlnet

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # Determine model type and load accordingly
                if isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(model, MultiControlNetModel):
                    load_model = MultiControlNetModel.from_pretrained(os.path.join(input_dir, "controlnet", "controlnet"))
                    model.load_state_dict(load_model.state_dict())  # Load only the state dictionary.
                    del load_model
            
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Initialize the model
    if args.pretrained_model_name_or_path is None:
        if args.model_config_name_or_path is None:
            model = UNet2DConditionModel(
                sample_size=args.resolution,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        else:
            config = UNet2DConditionModel.load_config(args.model_config_name_or_path)
            model = UNet2DConditionModel.from_config(config)
    else:
        logger.info(f"Loading existing unet weights from {args.pretrained_model_name_or_path}")
        model = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet"
        )
    
    args.train_unet = False
    # froze unet
    if not args.train_unet:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "up_blocks" in name:  # The decoder is usually contained in up_blocks.
                param.requires_grad = True
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print(name)
      
    if args.controlnet_model_name_or_path is not None:
        logger.info("Loading existing controlnet weights")
        controlnet = MultiControlNetModel.from_pretrained(args.controlnet_model_name_or_path, subfolder="controlnet")
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet1 = ControlNetModel.from_unet(model,conditioning_channels=1,conditioning_embedding_out_channels=[64])
        # controlnet1 = ControlNetModel.from_pretrained('output/ddpm2_lr2e5_squaredcos_deep_st15_mutilcontrolnet_epoch2/checkpoint-7600/controlnet', subfolder="controlnet")
        controlnet2 = ControlNetModel.from_unet(model,conditioning_channels=1,conditioning_embedding_out_channels=[64])
        controlnet = MultiControlNetModel([controlnet1, controlnet2])

    print(controlnet)
    # print('mid_block:',model.mid_block)
    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DConditionModel,
            model_config=model.config,
        )
        if hasattr(args, 'use_controlnet_ema') and args.use_controlnet_ema:
            ema_model_controlnet = EMAModel(
                controlnet.parameters(),
                decay=args.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=args.ema_inv_gamma,
                power=args.ema_power,
                model_cls=MultiControlNetModel,
                model_config=controlnet.config,
            )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # Initialize the optimizer
    if args.train_unet:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate / 10,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
    )
    optimizer_controlnet = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    ###################################################################
    # reset_dataset(accelerator, accelerator.device, True)
    # accelerator.wait_for_everyone()
    ###################################################################
    dataset = create_dataset(args.dataset_name, args)
    val_dataset = create_dataset(args.dataset_name, args, split='val')

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True
    )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers, drop_last=False
    # )

    # Initialize the learning rate scheduler
    if args.train_unet:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=(math.ceil(len(train_dataloader) * args.num_epochs  / args.gradient_accumulation_steps)),
        )
    lr_scheduler_controlnet = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_controlnet,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=(math.ceil(len(train_dataloader) * args.num_epochs  / args.gradient_accumulation_steps)),
    )
    # Prepare everything with our `accelerator`.
    if args.train_unet:
        model, controlnet, optimizer, optimizer_controlnet, train_dataloader, lr_scheduler, lr_scheduler_controlnet= accelerator.prepare(
            model, controlnet, optimizer, optimizer_controlnet, train_dataloader, lr_scheduler, lr_scheduler_controlnet
        )
    else:
        model, controlnet, optimizer_controlnet, train_dataloader, lr_scheduler_controlnet= accelerator.prepare(
            model, controlnet, optimizer_controlnet, train_dataloader, lr_scheduler_controlnet
        )

    if args.use_ema:
        ema_model.to(accelerator.device)
        if hasattr(args, 'use_controlnet_ema') and args.use_controlnet_ema:
            ema_model_controlnet.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume = 0

    loss_dic = {}
    loss_dic['loss'] = []
    loss_dic['n_step'] = 0
    loss_dic['accu_loss'] = 0
    
    
        
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            noise_scheduler = noise_scheduler.from_pretrained(os.path.join(args.output_dir, path, "scheduler"))
            with open(os.path.join(args.output_dir, path, 'loss.json'), "r", encoding="utf-8") as f:
                loss_dic = json.load(f)
            loss_dic['accu_loss'] = 0
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            resume = 1

    max_ks = 3#39
    min_ks = 3
    kernels = {
        int(ks) : torch.ones(1, 1, ks, ks, device=accelerator.device, dtype=weight_dtype) / (ks ** 2)
        for ks in range(min_ks, max_ks + 1, 2)
    }
    kernels_2 = {
        int(ks) : torch.ones(1, 1, ks, ks, device=accelerator.device, dtype=torch.float32) / (ks ** 2)
        for ks in range(min_ks, max_ks + 1, 2)
    }
    noise_weight = (1 - noise_scheduler.alphas_cumprod) ** 0.5
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0,kernel_size=11,sigma=1.5).to(accelerator.device)
    
    mask_processor = StripMaskProcessor(
        kernel_size=41, sigma=7.0,contrast_factor=40
        )
    boundary_loss = StripBoundaryLoss().to(accelerator.device)
    
    # ssim = kornia.losses.SSIM(
    #             window_size=11,  # Equivalent to kernel_size=11.
    #             max_val=2.0,     # Equivalent to data_range=2.0.
    #             reduction='none' # Do not reduce; preserve the output dimensions.
    #         ).to(accelerator.device)
    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        if epoch>0:
            if resume != 1:
                reset_dataset(accelerator, accelerator.device)
                accelerator.wait_for_everyone()
                dataset = create_dataset(args.dataset_name, args)
                # val_dataset = create_dataset(args.dataset_name, args, split='val')
                train_dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True
                )
                train_dataloader = accelerator.prepare(train_dataloader)
            else:    
                resume = 0
        if args.train_unet:
            model.train()
        controlnet.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
                
                
            clean_images = batch["image"].to(weight_dtype)
            cond_images = batch["mask"].to(weight_dtype)
            
            interval = 1
            # interval = max(1, int((1 - 3 * global_step / max_train_steps) * 50))
            # max_num = max(5, int((1 - global_step / max_train_steps) * 50))
            # min_num = max(0, int(max_num - 10))
            max_num = 3
            min_num = 1
            width = 2 * interval + 1
            rdm = torch.randint(0, clean_images.shape[2], (clean_images.shape[0], 1 + max_num,))
            # rdm2 = torch.randint(0, (image.shape[1] // width) - 1, (max_num,))
            random_values = torch.rand(clean_images.shape[0], (clean_images.shape[2] // width) - 1)
            rdm2 = random_values.argsort(dim=1)[:, :max_num]
            N = (rdm[:,0] / (clean_images.shape[2] / ((max_num - min_num) + 1))).int() + min_num
            # print('N:', N, min_num, max_num)
            First = (rdm[:,1:] / (clean_images.shape[2] / width)).int()
            pos = (rdm2 * width + First).clamp(0, clean_images.shape[2]).int()
            pos2 = torch.randint(0, clean_images.shape[-1], (clean_images.shape[0], max_num,))
            pos3 = torch.randint(64, clean_images.shape[2], (clean_images.shape[0], max_num,))
            well_images = torch.zeros_like(clean_images, device=clean_images.device).to(weight_dtype)
            mask = torch.zeros_like(clean_images, device=clean_images.device).float()
            
            for ind in range(clean_images.shape[0]):
                pos_f = (pos[0,:N[0]].unsqueeze(1) + torch.arange(width)).flatten()
                pos_f2 = (pos2[0,:N[0]].unsqueeze(1).repeat(1, width)).flatten()
                pos_f3 = (pos3[0,:N[0]].unsqueeze(1).repeat(1, width)).flatten()
                well_images[ind, :, pos_f] = clean_images[ind, :, pos_f].clone().view(clean_images.shape[1], N[0], width, clean_images.shape[-1]).mean(2,keepdim=True).repeat(1,1,width,1).view(clean_images.shape[1],-1,clean_images.shape[-1])
                row_indices = torch.cat([torch.arange(max(0, p2-p3), min(p2+p3, clean_images.shape[-1])) for p2, p3 in zip(pos_f2, pos_f3)])
                col_indices = torch.cat([torch.full((min(p2+p3, clean_images.shape[-1]) - max(0, p2-p3),), p1, dtype=torch.long) for p1, p2, p3 in zip(pos_f, pos_f2, pos_f3)])
                mask[ind, :, col_indices, row_indices] = 1
            
            well_images = well_images * mask
            # kernel_size = torch.randint(low=10, high=50, size=(1,))[0].item()
            # shift = torch.rand((clean_images.shape[0], 1, 1, 1), dtype=clean_images.dtype, device=clean_images.device) * 0
            # initial_well_images = gaussian_smoothing(clean_images, kernel_size=2 * kernel_size + 1, sigma=kernel_size / 3).to(clean_images.device) + shift
            # well_images = torch.cat([well_images, initial_well_images], dim=1)
            
            # rand_float = torch.rand(1).item()
            # if rand_float < 0.5:
            #     kernel_size = torch.randint(low=10, high=50, size=(1,))[0].item()
            #     shift = torch.rand((clean_images.shape[0], 1, 1, 1), dtype=clean_images.dtype, device=clean_images.device) * 0.2
            #     initial_well_images = gaussian_smoothing(clean_images, kernel_size=2 * kernel_size + 1, sigma=kernel_size / 3).to(clean_images.device) + shift
            # else:
            #     initial_well_images = torch.zeros_like(clean_images).to(clean_images.device)
            # well_images = torch.cat([initial_well_images, well_images], dim=1)
            
            # rand_float = torch.rand(1).item()
            # well_images = well_images * (rand_float < 0.5)
            # mask = mask * (rand_float < 0.5)
            
            # if accelerator.is_main_process:
            #     plt.figure(figsize=(15,4))  
            #     plt.subplot(131)
            #     plt.imshow(clean_images[0,0].detach().T.cpu(),cmap='jet',vmin=-1,vmax=1)
            #     plt.colorbar()
            #     plt.subplot(132)
            #     plt.imshow(cond_images[0,0].detach().T.cpu(),cmap='seismic',vmin=-1,vmax=1)
            #     plt.colorbar()
            #     plt.subplot(133)
            #     plt.imshow(well_images[0,0].detach().T.cpu(),cmap='jet',vmin=-1,vmax=1)
            #     plt.colorbar()
            #     plt.savefig('input_well.png')
            #     plt.close()
            # print(sss)
            
            # f0 = batch["f0"].to(weight_dtype)
            # dt = batch["dt"].to(weight_dtype)
            # imp_min, imp_max = batch["imp_min"].to(weight_dtype).view(-1, 1, 1, 1), batch["imp_max"].to(weight_dtype).view(-1, 1, 1, 1)
            # seis_min, seis_max = batch["seis_min"].to(weight_dtype).view(-1, 1, 1, 1), batch["seis_max"].to(weight_dtype).view(-1, 1, 1, 1)
            
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device) * 1

            bsz = clean_images.shape[0]
            
            random_item = random.choice([1, 2, 3])
            # random_item = 2
            
            if random_item == 1:
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()

                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                with accelerator.accumulate(model, controlnet):
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_images,
                        timesteps,
                        conditioning_scale = [torch.ones((noisy_images.shape[0],1,1,1)).to(noisy_images.device),
                                                torch.ones((noisy_images.shape[0],1,1,1)).to(noisy_images.device),],
                        encoder_hidden_states=None,
                        controlnet_cond=[cond_images, well_images],
                        # controlnet_cond=[well_images],
                        return_dict=False,
                    )
                    # for i in down_block_res_samples:
                    #     print(i.shape)
                    # Predict the noise residual
                    model_output = model(
                        noisy_images,
                        timesteps, 
                        encoder_hidden_states=None,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample
                    
                    if args.prediction_type == "epsilon":
                        loss_noise = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                    elif args.prediction_type == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss_noise = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                        loss_noise = loss.mean()
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                        loss_noise_shaped = F.mse_loss(model_output.float(), target.float(), reduction="none")
                        loss_noise = loss_noise_shaped.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {args.prediction_type}")
                    
                    # loss_well = (mask * loss_noise_shaped).sum() / (mask.sum() + 1e-8)
                    
                    # """
                    # Rewarding ControlNet
                    # """
                    # noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
                    # pred_original_sample_ = [
                    #     noise_scheduler.step(noise_, t_.item(), noisy_latent_).pred_original_sample.to(weight_dtype) \
                    #         for (noise_, t_, noisy_latent_) in zip(model_output, timesteps, noisy_images)
                    # ]
                    # # mask[timesteps>1400] = 0
                    # pred_original_sample = torch.stack(pred_original_sample_)
                    # loss_well = ((pred_original_sample - well_images) ** 2 * mask).sum() / (mask.sum() + 1e-8)
                    
            
                    # pred_original_sample = mean_filter(pred_original_sample, min_ks=min_ks, max_ks=max_ks, t=noise_weight[timesteps.cpu()], num_steps=noise_scheduler.config.num_train_timesteps, kernels=kernels).clamp(-1,1)
                    # pred_original_sample_denorm = pred_original_sample * 0.5 + 0.5
                    # pred_original_sample_denorm = pred_original_sample_denorm * (imp_max - imp_min) + imp_min
                    # seis_forward = forward(pred_original_sample_denorm, f0, dt).to(weight_dtype)
                    # seis_forward = (seis_forward - seis_min) / (seis_max - seis_min)
                    # seis_forward = (seis_forward - 0.5) / 0.5
                    # loss_forward = 0.5 * 0.1 * ssim(seis_forward.float(), cond_images.float()) + 0.5 * 10 * F.mse_loss(seis_forward.float(), cond_images.float())
                    
                    
                    # loss_smooth = smooth_loss(pred_original_sample[:,0].mean(2), window=10, threshold=1.5)
                    soft_mask = mask_processor(mask)
                    boundary_mask = mask_processor.get_strip_boundary_mask(mask)
                    b_loss = boundary_loss(model_output.float(), target.float(), soft_mask, boundary_mask)
                    mse_loss = (F.mse_loss(model_output.float(), target.float(), reduction='none') * soft_mask).sum() / (soft_mask.sum() + 1e-8)
                    
                    loss = loss_noise + (b_loss + mse_loss)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if args.train_unet:
                            all_parameters = list(model.parameters()) + list(controlnet.parameters())
                        else:
                            all_parameters = list(controlnet.parameters())
                        accelerator.clip_grad_norm_(all_parameters, 1.0)
        
                    if args.train_unet:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    optimizer_controlnet.step()
                    lr_scheduler_controlnet.step()
                    optimizer_controlnet.zero_grad()
                    
            elif random_item == 2:
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()

                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                with accelerator.accumulate(model, controlnet):
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_images,
                        timesteps,
                        conditioning_scale = [torch.zeros((noisy_images.shape[0],1,1,1)).to(noisy_images.device),
                                              torch.ones((noisy_images.shape[0],1,1,1)).to(noisy_images.device),],
                        encoder_hidden_states=None,
                        controlnet_cond=[cond_images, well_images],
                        return_dict=False,
                    )

                    # Predict the noise residual
                    model_output = model(
                        noisy_images,
                        timesteps, 
                        encoder_hidden_states=None,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample

                    
                    if args.prediction_type == "epsilon":
                        loss_noise = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                    elif args.prediction_type == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss_noise = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                        loss_noise = loss.mean()
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                        loss_noise_shaped = F.mse_loss(model_output.float(), target.float(), reduction="none")
                        loss_noise = loss_noise_shaped.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {args.prediction_type}")
                    
                    # """
                    # Rewarding ControlNet
                    # """
                    # noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
                    # pred_original_sample_ = [
                    #     noise_scheduler.step(noise_, t_.item(), noisy_latent_).pred_original_sample.to(weight_dtype) \
                    #         for (noise_, t_, noisy_latent_) in zip(model_output, timesteps, noisy_images)
                    # ]
                    # # mask[timesteps>1400] = 0
                    # pred_original_sample = torch.stack(pred_original_sample_)
                    # loss_well = ((pred_original_sample - well_images) ** 2 * mask).sum() / (mask.sum() + 1e-8)
                    
            
                    # pred_original_sample = mean_filter(pred_original_sample, min_ks=min_ks, max_ks=max_ks, t=noise_weight[timesteps.cpu()], num_steps=noise_scheduler.config.num_train_timesteps, kernels=kernels).clamp(-1,1)
                    # pred_original_sample_denorm = pred_original_sample * 0.5 + 0.5
                    # pred_original_sample_denorm = pred_original_sample_denorm * (imp_max - imp_min) + imp_min
                    # seis_forward = forward(pred_original_sample_denorm, f0, dt).to(weight_dtype)
                    # seis_forward = (seis_forward - seis_min) / (seis_max - seis_min)
                    # seis_forward = (seis_forward - 0.5) / 0.5
                    # loss_forward = 0.5 * 0.1 * ssim(seis_forward.float(), cond_images.float()) + 0.5 * 10 * F.mse_loss(seis_forward.float(), cond_images.float())
                    # loss_smooth = smooth_loss(pred_original_sample[:,0].mean(2), window=10, threshold=1.5)
                    
                    # loss_well = (mask * loss_noise_shaped).sum() / (mask.sum() + 1e-8)
                    soft_mask = mask_processor(mask)
                    boundary_mask = mask_processor.get_strip_boundary_mask(mask)
                    b_loss = boundary_loss(model_output.float(), target.float(), soft_mask, boundary_mask)
                    mse_loss = (F.mse_loss(model_output.float(), target.float(), reduction='none') * soft_mask).sum() / (soft_mask.sum() + 1e-8)
                    
                    loss = loss_noise + (b_loss + mse_loss)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if args.train_unet:
                            all_parameters = list(model.parameters()) + list(controlnet.parameters())
                        else:
                            all_parameters = list(controlnet.parameters())
                        accelerator.clip_grad_norm_(all_parameters, 1.0)
        
                    if args.train_unet:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    optimizer_controlnet.step()
                    lr_scheduler_controlnet.step()
                    optimizer_controlnet.zero_grad()
                    
            elif random_item == 3:
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()

                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                with accelerator.accumulate(model, controlnet):
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_images,
                        timesteps,
                        conditioning_scale = [torch.ones((noisy_images.shape[0],1,1,1)).to(noisy_images.device),
                                              torch.zeros((noisy_images.shape[0],1,1,1)).to(noisy_images.device),],
                        encoder_hidden_states=None,
                        controlnet_cond=[cond_images, well_images],
                        return_dict=False,
                    )
                    # Predict the noise residual
                    model_output = model(
                        noisy_images,
                        timesteps, 
                        encoder_hidden_states=None,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample
                    
                    if args.prediction_type == "epsilon":
                        loss_noise = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                    elif args.prediction_type == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss_noise = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                        loss_noise = loss.mean()
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                        loss_noise_shaped = F.mse_loss(model_output.float(), target.float(), reduction="none")
                        loss_noise = loss_noise_shaped.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {args.prediction_type}")
                
                    loss = loss_noise
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if args.train_unet:
                            all_parameters = list(model.parameters()) + list(controlnet.parameters())
                        else:
                            all_parameters = list(controlnet.parameters())
                        accelerator.clip_grad_norm_(all_parameters, 1.0)
        
                    if args.train_unet:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    optimizer_controlnet.step()
                    lr_scheduler_controlnet.step()
                    optimizer_controlnet.zero_grad()
                 
            loss_dic['n_step'] += 1
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                    if hasattr(args, 'use_controlnet_ema') and args.use_controlnet_ema:
                        ema_model_controlnet.step(controlnet.parameters())
                    
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        with open(save_path + "/loss.json", "w", encoding="utf-8") as f:
                            json.dump(loss_dic, f, indent=4)
                        copy_Files(save_path)
                        logger.info(f"Saved state to {save_path}")
            if args.train_unet:
                logs = {"item":random_item, "loss": loss.detach().item(), "lr_u": lr_scheduler.get_last_lr()[0],"lr_c": lr_scheduler_controlnet.get_last_lr()[0], "step": global_step,'sum_step':loss_dic['n_step']}
            else:
                logs = {"item":random_item, "loss": loss.detach().item(), "lr_c": lr_scheduler_controlnet.get_last_lr()[0], "step": global_step,'sum_step':loss_dic['n_step']}
            loss_dic['accu_loss'] += loss.detach().item()
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
                if hasattr(args, 'use_controlnet_ema') and args.use_controlnet_ema:
                    logs["ema_decay_controlnet"] = ema_model_controlnet.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            
            if accelerator.is_main_process:
                if loss_dic['n_step']%100 == 0 and loss_dic['n_step']>0:
                    loss_dic['loss'].append(loss_dic['accu_loss']/100)
                    plt.figure()
                    plt.plot(loss_dic['loss'])
                    plt.savefig('loss_{}.png'.format(os.path.basename(args.output_dir)))
                    plt.close()
                    loss_dic['accu_loss'] = 0
                    
                if  (loss_dic['n_step']%2000 == 0 or loss_dic['n_step'] == 2) and (loss_dic['n_step']>0):
                    unet = accelerator.unwrap_model(model)
                    controlnet_ = accelerator.unwrap_model(controlnet)
                    generator = torch.Generator(device=clean_images.device).manual_seed(0)
                    scheduler=noise_scheduler
                    
                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())
                        if hasattr(args, 'use_controlnet_ema') and args.use_controlnet_ema:
                            ema_model_controlnet.store(controlnet_.parameters())
                            ema_model_controlnet.copy_to(controlnet_.parameters())
                    
                    # test_id = random.randint(0, len(val_dataset))  
                    test_id = -100
                    images, well_images_ = inference(
                        min_num=min_num,
                        max_num=max_num,
                        interval=interval,
                        unet=unet,
                        controlnet=controlnet_,
                        control_guidance_start=[0.0, 0.0],
                        control_guidance_end=[1.0, 1.0],
                        controlnet_conditioning_scale=[torch.zeros((noisy_images.shape[0],1,1,1)).to(noisy_images.device), 
                                                       torch.ones((noisy_images.shape[0],1,1,1)).to(noisy_images.device),],
                        scheduler=noise_scheduler,
                        generator=generator,
                        num_inference_steps=300,
                        device=clean_images.device,
                        cond_dataset=val_dataset[test_id],
                        kernels=kernels_2
                    )
                    if args.use_ema:
                        ema_model.restore(unet.parameters())
                        if hasattr(args, 'use_controlnet_ema') and args.use_controlnet_ema:
                            ema_model_controlnet.restore(controlnet_.parameters())
                            
                    # Save images
                    quick_save_4images(images, well_images_, val_dataset[test_id], save_dir=os.path.join('png_inference', os.path.basename(args.output_dir)), n_step=loss_dic["n_step"])
            
        progress_bar.close()

        accelerator.wait_for_everyone()

        # # Generate sample images for visual inspection
        # if accelerator.is_main_process:
        #     if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
        #         unet = accelerator.unwrap_model(model)

        #         if args.use_ema:
        #             ema_model.store(unet.parameters())
        #             ema_model.copy_to(unet.parameters())

        #         pipeline = DDPMPipeline(
        #             unet=unet,
        #             scheduler=noise_scheduler,
        #         )

        #         generator = torch.Generator(device=pipeline.device).manual_seed(0)
        #         # run pipeline in inference (sample random noise and denoise)
        #         images = pipeline(
        #             generator=generator,
        #             batch_size=1,
        #             num_inference_steps=args.ddpm_num_inference_steps,
        #             output_type="np",
        #         ).images

        #         if args.use_ema:
        #             ema_model.restore(unet.parameters())
        #         plt.figure()  
        #         plt.imshow(images[0,0,:,:,0].T.cpu())
        #         plt.savefig('inference.png')
        #         plt.close()

        #     if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
        #         # save the model
        #         unet = accelerator.unwrap_model(model)

        #         if args.use_ema:
        #             ema_model.store(unet.parameters())
        #             ema_model.copy_to(unet.parameters())

        #         pipeline = DDPMPipeline(
        #             unet=unet,
        #             scheduler=noise_scheduler,
        #         )

        #         pipeline.save_pretrained(args.output_dir)

        #         if args.use_ema:
        #             ema_model.restore(unet.parameters())

        #         if args.push_to_hub:
        #             upload_folder(
        #                 repo_id=repo_id,
        #                 folder_path=args.output_dir,
        #                 commit_message=f"Epoch {epoch}",
        #                 ignore_patterns=["step_*", "epoch_*"],
        #             )
    if accelerator.is_main_process:
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        if args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        with open(save_path + "/loss.json", "w", encoding="utf-8") as f:
            json.dump(loss_dic, f, indent=4)
        copy_Files(save_path)
        logger.info(f"Saved state to {save_path}")
    
    accelerator.end_training()

class StripMaskProcessor:
    def __init__(self, kernel_size=21, sigma=3.0, contrast_factor=6.0):
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure an odd size.
        self.sigma = sigma
        self.contrast_factor = contrast_factor
        self.gaussian_kernel = self._create_horizontal_gaussian_kernel()
        
    def _create_horizontal_gaussian_kernel(self):
        """Create a horizontal Gaussian kernel without vertical blurring."""
        # Generate the 1D Gaussian distribution.
        kernel_1d = torch.linspace(-(self.kernel_size // 2), self.kernel_size // 2, self.kernel_size)
        kernel_1d = torch.exp(-0.5 * (kernel_1d / self.sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize.
        
        # Expand to a 2D kernel that blurs only horizontally.
        kernel_2d = kernel_1d  # Shape: [1, kernel_size].
        
        return kernel_2d
        
    def __call__(self, hard_mask):
        kernel = self.gaussian_kernel.to(hard_mask.device, hard_mask.dtype)
        kernel = kernel.view(1, 1, self.kernel_size, 1)
        # Apply horizontal blurring with the larger kernel next.
        soft_mask = F.conv2d(
            hard_mask, 
            kernel, 
            padding=(self.kernel_size // 2, 0),  # Pad only in the horizontal direction.
            groups=hard_mask.size(1)  # Grouped convolution keeps the channels independent.
        )
        
        # Enhance contrast while preserving the mask interior.
        soft_mask = torch.sigmoid((soft_mask - 0.12) * self.contrast_factor)
        
        return soft_mask
    
    def get_strip_boundary_mask(self, hard_mask):
        """Create an elongated-mask boundary map emphasizing the left and right edges."""
        
        # Detect vertical-direction edges at the top and bottom of the strip.
        vertical_kernel = torch.tensor([[-1.], [0.], [1.]], device=hard_mask.device).unsqueeze(0).unsqueeze(0)
        vertical_edges = F.conv2d(hard_mask, vertical_kernel, padding='same')
        vertical_edges = torch.abs(vertical_edges)
        
        # Assign greater weight to horizontal edges, which matter more for a 512x3 strip.
        boundary_mask = 1 * vertical_edges
        return torch.clamp(boundary_mask, 0, 1)

# 2. Boundary loss tailored to elongated masks.
class StripBoundaryLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        # Horizontal and vertical gradient kernels.
        self.grad_kernel_y = torch.tensor([[-1.], [0.], [1.]], device='cpu').unsqueeze(0).unsqueeze(0)  # Vertical gradient.
        
    def forward(self, v_pred, v_true, mask, boundary_mask):
        """
        Boundary loss tailored to elongated masks,
        emphasizing horizontal edge continuity.
        """
        # Ensure the convolution kernel is on the correct device.
        self.grad_kernel_y = self.grad_kernel_y.to(v_pred.device)
        
        # Compute horizontal and vertical gradients.
        pred_grad_y = F.conv2d(v_pred, self.grad_kernel_y, padding='same')
        
        true_grad_y = F.conv2d(v_true, self.grad_kernel_y, padding='same')
        
        # Compute horizontal and vertical losses separately and apply their respective weights.
        loss_y = F.mse_loss(pred_grad_y, true_grad_y, reduction='none')
        
        # Compute the loss only inside the mask.
        return (loss_y * boundary_mask).sum() / (boundary_mask.sum() + 1e-8)
      
def gaussian_kernel(size=3, sigma=1.0):
    """Generate a 2D Gaussian kernel."""
    kernel = np.zeros((size, size))
    center = size // 2
    for x in range(size):
        for y in range(size):
            dx, dy = x - center, y - center
            kernel[x, y] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize.
    return kernel
def gaussian_kernel_1d(size=3, sigma=1.0):
    """Generate a 1D Gaussian kernel."""
    kernel = np.zeros(size)
    center = size // 2
    for x in range(size):
        dx = x - center
        kernel[x] = np.exp(-(dx**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize.
    return kernel

def gaussian_smoothing_1d(signal, kernel_size=3, sigma=1.0):
    # Ensure the input is a torch tensor.
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)
    
    # Handle input dimensions for either a single sample or a batch.
    is_batched = len(signal.shape) == 3
    if not is_batched:
        signal = signal.unsqueeze(0)  # (C, L) -> (1, C, L)
    
    channels = signal.shape[1]
    kernel = gaussian_kernel_1d(kernel_size, sigma)  # Generate the 1D kernel.
    
    # Convert the kernel to PyTorch convolution weights with shape (out_c, in_c/groups, kernel_size).
    kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel = kernel.repeat(channels, 1, 1)  # (C, 1, kernel_size)
    
    # Define a 1D convolution layer without learnable parameters.
    conv = nn.Conv1d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,  # Grouped convolution processes each channel independently.
        bias=False,
        padding=kernel_size // 2,
        padding_mode='replicate'
    )
    conv.weight.data = kernel  # Fix the weights to the Gaussian kernel.
    conv.weight.requires_grad = False
    
    # Apply convolution and restore the original dimensions.
    smoothed = conv(signal)
    return smoothed if is_batched else smoothed.squeeze(0)

def gaussian_smoothing(image, kernel_size=3, sigma=1.0):
    """image: input tensor with shape (C, H, W) or (B, C, H, W)."""
    channels = image.shape[-3]
    kernel = gaussian_kernel(kernel_size, sigma)  # Generate the kernel with the helper above.
    
    # Convert the kernel to PyTorch convolution weights.
    kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel = kernel.repeat(channels, 1, 1, 1).to(image.device).to(image.dtype)  # Expand to a multi-channel convolution kernel with shape (out_c, in_c, H, W).
    
    # Define a convolution layer without learnable parameters.
    conv = nn.Conv2d(
        in_channels=channels, 
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,  # Grouped convolution processes each channel independently.
        bias=False,
        padding=kernel_size // 2,
        padding_mode = 'reflect'
    )
    conv.weight.data = kernel  # Fix the weights to the Gaussian kernel.
    conv.weight.requires_grad = False
    
    return conv(image)
    
def mean_filter(input_batch, t, min_ks, max_ks, num_steps, kernels):
    
    assert input_batch.dim() == 4 and input_batch.size(1) == 1, "Input shape must be [bs, 1, H, W]."
    
    # Map t to an odd kernel size in [min_ks, max_ks].
    t_ratio = t    # Normalize to [0, 1].
    kernel_sizes_float = min_ks + (max_ks - min_ks) * t_ratio
    kernel_sizes = (kernel_sizes_float.round().int() // 2 * 2 + 1).clamp(min_ks, max_ks)
    
    # Process each sample independently.
    batch_size = input_batch.size(0)
    output = torch.empty_like(input_batch)
    
    for i in range(batch_size):
        ks = int(kernel_sizes[i].item())
        padding = ks // 2
        # Create the averaging kernel.
        kernel = kernels[ks]
        
        # Step 1: reflection padding horizontally, on the left and right only.
        padded = F.pad(
            input_batch[i:i+1], 
            (padding, padding, 0, 0),  # Pad only the left and right sides.
            mode='replicate'
        )
        
        # Step 2: replication padding vertically, at the top and bottom only.
        padded = F.pad(
            padded, 
            (0, 0, padding, padding),  # Pad only the top and bottom.
            mode='reflect'
        )
    
        # Reflection padding followed by convolution.
        # padded = F.pad(input_batch[i:i+1], 
        #               (padding, padding, padding, padding), 
        #               mode='reflect')# replicate reflect
        output[i] = F.conv2d(padded, kernel, groups=1)
    
    return output

def median_filter(input_batch, kernel_size=3):
    assert kernel_size % 2 == 1, "kernel_size must be odd."
    padding = kernel_size // 2
    
    # Extract local neighborhoods with unfold.
    unfolded = F.unfold(input_batch, kernel_size=kernel_size, padding=padding)
    
    # Compute medians: (bs*1, kernel_size^2, H*W) -> (bs*1, H*W).
    median = unfolded.median(dim=1, keepdim=True)[0]
    
    # Restore shape (bs, 1, H, W).
    return median.view_as(input_batch)

def gaussian_smooth_batch(input_batch, kernel_size=5, sigma=1.0):
    
    assert input_batch.dim() == 4, "Input must be a 4D tensor with shape (bs, 1, H, W)."
    
    # Create a Gaussian kernel with shared weights.
    kernel = _get_gaussian_kernel(kernel_size, sigma, device=input_batch.device)
    
    # Expand the kernel across the batch: (bs, 1, k, k) -> (bs*1, 1, k, k).
    b, c, *_ = input_batch.shape
    kernel = kernel.repeat(b, 1, 1, 1)  # Use the same kernel for every sample.
    
    # Use grouped convolution for batch filtering.
    smoothed = F.conv2d(
        input=input_batch.view(1, b*c, *input_batch.shape[2:]),  # (1, bs*1, H, W)
        weight=kernel,  # (bs*1, 1, k, k)
        padding=kernel_size//2,
        groups=b*c  # Process each sample independently.
    )
    
    return smoothed.view_as(input_batch)  # Restore the original shape.

def _get_gaussian_kernel(kernel_size, sigma, device='cpu'):
    """Generate a 2D Gaussian kernel."""
    # Create the coordinate grid.
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size//2
    y = x.view(-1, 1)
    
    # Compute the 2D Gaussian distribution.
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize.
    
    # Reshape to (1, 1, kernel_size, kernel_size) for conv2d.
    return kernel.view(1, 1, kernel_size, kernel_size)

def compute_cross_correlation(img1, img2, window_size=3):
    """
    Args:
        img1, img2: (bs, 1, H, W)
        window_size: Custom window size, which must be odd.
    Returns:
        (bs, 1, H, W) Cross-correlation map
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    padding = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=img1.device).to(img1.dtype)
    
    # Compute cross-correlation as the mean of local products.
    cross_map = torch.nn.functional.conv2d(
        (img1 * img2).to(img1.dtype), 
        weight=kernel, 
        padding=padding
    )
    # Compute the local energies of img1 and img2 for normalization.
    img1_sq = torch.nn.functional.conv2d(img1 ** 2, weight=kernel, padding=padding)
    img2_sq = torch.nn.functional.conv2d(img2 ** 2, weight=kernel, padding=padding)
    
    # Normalize to [-1, 1].
    epsilon = 1e-8
    norm = torch.sqrt(img1_sq * img2_sq) + epsilon
    ncc = cross_map / norm
    
    return ncc

def compute_cosine(img1, img2, window_size=3, eps=1e-8):
    """
    Args:
        img1, img2: (bs, 1, H, W)
        window_size: Custom window size, which must be odd.
    Returns:
        (bs, 1, H, W) Cosine similarity map
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    padding = window_size // 2
    unfold = torch.nn.Unfold(kernel_size=window_size, padding=padding)
    
    # Unfold to (bs, window_size*window_size, H*W).
    img1_unfold = unfold(img1)  # (bs, 9, H*W) if window_size=3
    img2_unfold = unfold(img2)
    
    # Compute dot products and norms.
    dot_product = (img1_unfold * img2_unfold).sum(dim=1, keepdim=True)  # (bs, 1, H*W)
    norm = torch.norm(img1_unfold, dim=1, keepdim=True) * torch.norm(img2_unfold, dim=1, keepdim=True)
    
    # Reassemble the spatial layout.
    cosine = dot_product / (norm + eps)
    return cosine.view(img1.shape[0], 1, img1.shape[2], img1.shape[3])



def compute_ssim(img1, img2, window_size=11, reduction='none'):
    """
    Args:
        img1, img2: (bs, 1, H, W)
        window_size: Custom window size, which must be odd.
    Returns:
        (bs, 1, H, W) SSIM map
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    ssim_map = kornia.metrics.ssim(
        img1, img2, 
        window_size=window_size, 
        max_val=1.0, 
    )
    return ssim_map  # Preserve shape bs*1*H*W.

def smooth_loss(line, window=10, threshold=1.5,epoch=0):
    padded = F.pad(line, (window, window), mode='reflect')  # [bs, length + 2*window]
    
    unfolded = padded.unfold(dimension=1, size=2*window+1, step=1)  # [bs, length, 2*window+1]
    stds = torch.std(unfolded, dim=2, unbiased=False)

    # weights = stds * (stds > threshold * stds.mean(dim=1, keepdim=True)).float()
    weights = (stds > threshold * stds.mean(dim=1, keepdim=True)).float()
    if epoch > 1:
        gradients = ((line[:, 1:] - line[:, :-1]) / ( line[:, :-1] + 1e-5)) ** 2   # [bs, length-1]
    else:
        gradients = ((line[:, 1:] - line[:, :-1])) ** 2   # [bs, length-1]
    loss = torch.sum(weights[:, 1:] * gradients) / line.size(0)  # Average over the batch.
    return loss

def identify_lobes_with_params(y, k_list, dt=0.001):
    """Identify main-lobe and sidelobe regions with a separate k parameter for each wavelet."""
    abs_y = np.abs(y)
    batch_size, wave_len = abs_y.shape
    
    # Convert k_list to an array and check matching lengths.
    k_list = np.asarray(k_list)
    if len(k_list) != batch_size:
        raise ValueError("k_list must contain one value per wavelet.")
    
    # Find the main-lobe index of each wavelet.
    main_lobe_idx = np.argmax(abs_y, axis=1)
    
    # Initialize the mask.
    lobe_mask = np.zeros_like(abs_y, dtype=bool)
    indices = np.arange(wave_len)[np.newaxis, :]
    
    # Process the main lobe of each wavelet with vectorized indexing.
    for i in range(batch_size):
        lobe_width = int(k_list[i] / dt)
        # Compute the main-lobe bounds for the current wavelet.
        start = np.maximum(0, main_lobe_idx[i] - lobe_width)
        end = np.minimum(wave_len, main_lobe_idx[i] + lobe_width)
        lobe_mask[i, start:end] = True
        
        # Process the sidelobes of the current wavelet.
        peaks, _ = find_peaks(abs_y[i], height=np.max(abs_y[i])*0.2, distance=int(0.05/dt))
        side_peaks = peaks[peaks != main_lobe_idx[i]]
        
        if len(side_peaks) > 0:
            starts = np.maximum(0, side_peaks - lobe_width)
            ends = np.minimum(wave_len, side_peaks + lobe_width)
            
            # Mark sidelobe regions.
            for s, e in zip(starts, ends):
                lobe_mask[i, s:e] = True
    
    return lobe_mask

def gaussian_smooth_with_params(y, k_list, sigma_list, dt=0.001):
    """Gaussian-smooth the masks using separate k and sigma parameters for each wavelet."""
    # Convert parameters to arrays and check matching lengths.
    k_list = np.asarray(k_list)
    sigma_list = np.asarray(sigma_list)
    batch_size = y.shape[0]
    
    if len(k_list) != batch_size or len(sigma_list) != batch_size:
        raise ValueError("k_list and sigma_list must each contain one value per wavelet.")
    
    # Get the lobe mask for each wavelet.
    lobe_mask = identify_lobes_with_params(y, k_list, dt)
    batch_size, wave_len = lobe_mask.shape
    
    # Initialize the distance field.
    min_distances = np.zeros_like(y, dtype=np.float32)
    indices = np.arange(wave_len)
    
    # Compute a distance field and Gaussian smoothing for each wavelet.
    for i in range(batch_size):
        mask_indices = np.where(lobe_mask[i])[0]
        if len(mask_indices) > 0:
            # Compute the distance matrix and find its minimum values.
            dist_matrix = np.abs(indices[:, np.newaxis] - mask_indices)
            min_dist = dist_matrix.min(axis=1) * dt
            min_dist[lobe_mask[i]] = 0
            min_distances[i] = min_dist
        
        # Apply Gaussian decay with the sigma of the current wavelet.
        sigma = sigma_list[i]
        min_distances[i] = np.exp(-(min_distances[i]**2) / (2 * sigma**2))
        min_distances[i][lobe_mask[i]] = 1.0  # Keep values at 1.0 inside lobe regions.
    
    return min_distances

def add_phase_perturbation_fully_vectorized(y, max_phase_shift=0.2):
    """Fully vectorized batch phase perturbation."""
    batch_size, n = y.shape
    # Generate phase shifts in a fully vectorized operation.
    phase_shift = np.random.uniform(
        -max_phase_shift, 
        max_phase_shift, 
        (batch_size, n//2 + 1)
    )
    
    # Apply batched FFT and phase perturbations with full vectorization.
    y_fft = np.fft.rfft(y, axis=1)
    y_fft_perturbed = y_fft * np.exp(1j * phase_shift)
    
    return np.fft.irfft(y_fft_perturbed, n, axis=1)

def process_wavelets_with_params(ricker_wavelet, dt, k_list, sigma_list, max_phase_shift=1.5):
    """Process a batch with parameter sequences, using the corresponding k and sigma for each wavelet."""
    # Convert to a NumPy array with shape (batch_size, wave_length).
    wavelets_np = np.array([w.numpy() for w in ricker_wavelet])
    batch_size = wavelets_np.shape[0]
    
    # Validate parameter lengths.
    if len(k_list) != batch_size or len(sigma_list) != batch_size:
        raise ValueError("k_list and sigma_list must each contain one value per wavelet.")
    
    # Processing pipeline.
    smooth_mask = gaussian_smooth_with_params(
        wavelets_np, k_list=k_list, sigma_list=sigma_list, dt=dt
    )
    
    perturbed_wave = add_phase_perturbation_fully_vectorized(
        wavelets_np, max_phase_shift=max_phase_shift
    )
    
    # Apply the mask and normalize.
    new_wave = smooth_mask * perturbed_wave
    max_abs = np.max(np.abs(new_wave), axis=1, keepdims=True)
    new_wave = new_wave / max_abs  # Normalize.
    
    return new_wave

def reserve_gpu_memory(gb_size, device):
    """
    device: Pass accelerator.device.
    """
    n_elements = int((gb_size * 1024**3) // 4)
    try:
        # Use torch.empty followed by fill_ to avoid allocating an initialized zeros tensor first.
        dummy_tensor = torch.empty(n_elements, dtype=torch.float32, device=device).fill_(0)
        print(f'Rank {torch.distributed.get_rank()} reserved memory on {device}: {gb_size}GB')
        return dummy_tensor
    except Exception as e:
        print(f'Memory reservation failed: {e}')
        return None

def wavelet_forward(wavelet_mat, window, image):
    image_denorm = image
    B, C, H, W = image_denorm.shape
    n_patch = (wavelet_mat.shape[0], wavelet_mat.shape[1])
    image_pred_ori_shaped = image_denorm.reshape(-1, H, W)
    ref_pred_ori = imp2ref_torch(image_pred_ori_shaped, dim=-1)
    # window = (H//n_patch[0]+1 if H%n_patch[0]!=0 else H//n_patch[0],  W//n_patch[1]+1 if W%n_patch[1]!=0 else W//n_patch[1])
    if H % window[0] != 0:
        ref_pred_ori = torch.cat([ref_pred_ori, torch.zeros((B * C, window[0] - ref_pred_ori.shape[1] % window[0], ref_pred_ori.shape[2]), device=image.device)], dim=1)
    if W % window[0] != 0:
        ref_pred_ori = torch.cat([ref_pred_ori, torch.zeros((B * C, ref_pred_ori.shape[1], window[1] - ref_pred_ori.shape[2] % window[1]), device=image.device)], dim=2) 
    # n_patch = (ref_pred_ori.shape[1]//window[0], ref_pred_ori.shape[2]//window[1])
    # wavelet_mat = ricker_torch(f0_[0:1], dt_[0:1], 3/f0_[0:1])[0].repeat(n_patch[0] * n_patch[1], 1).unsqueeze(1).to(device)
    # wavelet_mat.requires_grad_(True) 
    wavelet_mat = wavelet_mat.view(n_patch[0]*n_patch[1], -1).unsqueeze(1)
    unfold = nn.Unfold(kernel_size=window, stride=window)
    ref_pred_ori = unfold(ref_pred_ori.unsqueeze(1))
    ref_pred_ori = ref_pred_ori.view(1, window[0], window[1], n_patch[0], n_patch[1]).permute(0, 3, 4, 1, 2) # 1, n_patch[0], n_patch[1], window[0], window[1]

    ref_pred_ori = ref_pred_ori.contiguous().view(-1, window[0], window[1])
    ref_pred_ori = ref_pred_ori.permute(1,0,2) # window[0], n_patch[0]*n_patch[1], window[1]
    ref_pred_ori = F.pad(ref_pred_ori, (wavelet_mat.shape[-1], wavelet_mat.shape[-1]))
    seis_pred_ori = F.conv1d(ref_pred_ori, wavelet_mat, padding=0, stride=1, groups=n_patch[0]*n_patch[1])
    seis_pred_ori = seis_pred_ori.permute(1,0,2)
    seis_pred_ori = seis_pred_ori.view(1, n_patch[0], n_patch[1], window[0],seis_pred_ori.shape[-1]).permute(0,2,1,3,4).contiguous().view(1,n_patch[1],-1,seis_pred_ori.shape[-1])
    seis_pred_ori_return = torch.zeros((1, n_patch[0]*window[0], n_patch[1]*window[1]+2*(wavelet_mat.shape[-1]//2+1)),device=image.device)
    for i in range(n_patch[1]):
        seis_pred_ori_return[:,:,i*window[-1]:i*window[-1] + seis_pred_ori.shape[-1]] += seis_pred_ori[0,i,:,:]
    seis_pred_ori_return = seis_pred_ori_return[:,:H,(wavelet_mat.shape[-1]//2+1):(wavelet_mat.shape[-1]//2+1)+W].unsqueeze(0)
    # seis_pred_ori_return = ((seis_pred_ori_return - seis_min) / (seis_max - seis_min) - 0.5) / 0.5
    return seis_pred_ori_return

def gaussian_smoothing(image, kernel_size=3, sigma=1.0):
    """image: input tensor with shape (C, H, W) or (B, C, H, W)."""
    channels = image.shape[-3]
    kernel = gaussian_kernel(kernel_size, sigma)  # Generate the kernel with the helper above.
    
    # Convert the kernel to PyTorch convolution weights.
    kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel = kernel.repeat(channels, 1, 1, 1)  # Expand to a multi-channel convolution kernel with shape (out_c, in_c, H, W).
    
    # Define a convolution layer without learnable parameters.
    conv = nn.Conv2d(
        in_channels=channels, 
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,  # Grouped convolution processes each channel independently.
        bias=False,
        padding=kernel_size // 2,
        padding_mode = 'reflect'
    )
    conv.weight.data = kernel.to(image.device)  # Fix the weights to the Gaussian kernel.
    conv.weight.requires_grad = False
    
    return conv(image)

def process_file2(input_path, output_folder, device):
    """
    Process a single file: load, compute, and save
    """
    # Load data.
    # print('Begin:',input_path)

    data = np.load(input_path,allow_pickle=True)
    dt = 0.0005
    dur = 5
    
    
    imp = torch.from_numpy(data['imp']).permute(1, 2, 0)
    nx, ny, nt = imp.shape
    
    number = random.choice([0, 1, 2, 3])
    for ix in range(number,nx,4):
        imp_slice = imp[ix].float().unsqueeze(0).unsqueeze(0)
        f0 = np.round(np.random.uniform(50, 100), 2)
        addf = np.round(np.random.uniform(0, 10), 2)
        window = round(np.random.uniform(5, 10))
        # f0 = 100
        # addf = 50
        # nf = 10
        window = (window, window)
        n_patch = (imp.shape[-2]//window[0]+1 if imp.shape[-2]%window[0]!=0 else imp.shape[-2]//window[0], imp.shape[-1]//window[1]+1 if imp.shape[-1]%window[1]!=0 else imp.shape[-1]//window[1])
        pos0 = np.linspace(0, n_patch[0], np.random.randint(4,10))
        pos1 = np.linspace(0, n_patch[1], np.random.randint(4,10))
        scale0 = np.random.uniform(0, 0.7, (len(pos0),))
        scale = np.concatenate([np.linspace(1+i, 1-i, len(pos1)) for i in scale0], axis=0)
        scale_noise = scale + scale * np.random.normal(loc=0, scale=0.1, size=scale.shape)
        scale_noise = np.clip(scale_noise, 0.2, 1.8)
        
        f_array = np.linspace(f0, f0-addf, len(pos1)).reshape(1,-1).repeat(len(pos0),axis=0).reshape(-1).round(5)
        f_array = np.clip(f_array, 40, 100)
        k = np.round((0.015-0.005)/(40-100) * f_array + 13/600, 5)
        s = np.round((0.005 - 0.003) / (40 - 100) * f_array + 19/3000, 5)
        ricker_wavelet, t = ricker_torch(torch.from_numpy(f_array),torch.tensor([dt]).repeat(f_array.shape[0]), torch.tensor([dur/(f_array.min())]).repeat(f_array.shape[0]),)
        processed_wavelet = process_wavelets_with_params(ricker_wavelet, dt, k_list=k, sigma_list=s).reshape(len(pos0) * len(pos1), -1)
        x, y = np.meshgrid(pos0, pos1, indexing='ij')
        points = np.column_stack((x.ravel(), y.ravel()))
        grid_x, grid_y = np.mgrid[0:n_patch[0], 0:n_patch[1]]
        
        rbf = Rbf(points[:,0], points[:,1], processed_wavelet, function='multiquadric',mode='N-D')
        processed_wavelet2 = rbf(grid_x, grid_y)
        
        rbf = Rbf(points[:,0], points[:,1], scale_noise, function='multiquadric')
        scale_full = rbf(grid_x, grid_y)
        processed_wavelet2 *= scale_full[:, :, np.newaxis] 
        # processed_wavelet = griddata(points, processed_wavelet, (grid_x, grid_y), method='cubic')
        # processed_wavelet = process_wavelets_with_params(ricker_wavelet, dt, k_list=k, sigma_list=s).reshape(1, n_patch[-1], -1).repeat(n_patch[-2],axis=0)
        # processed_wavelet = ricker_wavelet.numpy().reshape(1, n_patch[-1], -1).repeat(n_patch[-2],axis=0)
        seis_slice = wavelet_forward(torch.from_numpy(processed_wavelet2).float().to(device), window, imp_slice.to(device))
        mean = 0
        std = np.random.uniform(0, 2)
        kernel = np.random.randint(10, 30) * 2 + 1
        noise = mean + std * torch.randn(seis_slice.shape).to(device)
        seis_slice += gaussian_smoothing(noise, kernel, kernel/7)
        
        cube_i_mn = imp_slice.min()
        cube_i_mx = imp_slice.max()
        seis_i_mn = seis_slice.min()
        seis_i_mx = seis_slice.max()
        
        imp_slice = (imp_slice - cube_i_mn) / (cube_i_mx-cube_i_mn)
        seis_slice = seis_slice / max(seis_i_mn.abs(), seis_i_mx.abs()) * (torch.rand(1).to(device) * (1 - 0.3) + 0.3) * 0.5 + 0.5
        
        dic = {
                'seis':seis_slice[0,0].cpu().to(torch.float16),
                'imp':imp_slice[0,0].cpu().to(torch.float16),
            }
        torch.save(dic, output_folder + f'{os.path.basename(input_path).split(".")[0]}_{cube_i_mn:.2f}_{cube_i_mx:.2f}_{seis_i_mn:.2f}_{seis_i_mx:.2f}_{dt*1000}_{f0}_ix:{ix}.pth')
    for iy in range(number,ny,4):
        imp_slice = imp[:, iy].float().unsqueeze(0).unsqueeze(0)
        f0 = np.round(np.random.uniform(50, 100), 2)
        addf = np.round(np.random.uniform(0, 10), 2)
        window = round(np.random.uniform(5, 10))
        # f0 = 100
        # addf = 50
        # nf = 10
        window = (window, window)
        n_patch = (imp.shape[-2]//window[0]+1 if imp.shape[-2]%window[0]!=0 else imp.shape[-2]//window[0], imp.shape[-1]//window[1]+1 if imp.shape[-1]%window[1]!=0 else imp.shape[-1]//window[1])
        pos0 = np.linspace(0, n_patch[0], np.random.randint(4,10))
        pos1 = np.linspace(0, n_patch[1], np.random.randint(4,10))
        scale0 = np.random.uniform(0, 0.7, (len(pos0),))
        scale = np.concatenate([np.linspace(1+i, 1-i, len(pos1)) for i in scale0], axis=0)
        scale_noise = scale + scale * np.random.normal(loc=0, scale=0.1, size=scale.shape)
        scale_noise = np.clip(scale_noise, 0.2, 1.8)
        
        f_array = np.linspace(f0, f0-addf, len(pos1)).reshape(1,-1).repeat(len(pos0),axis=0).reshape(-1).round(5)
        f_array = np.clip(f_array, 40, 100)
        k = np.round((0.015-0.005)/(40-100) * f_array + 13/600, 5)
        s = np.round((0.005 - 0.003) / (40 - 100) * f_array + 19/3000, 5)
        ricker_wavelet, t = ricker_torch(torch.from_numpy(f_array),torch.tensor([dt]).repeat(f_array.shape[0]), torch.tensor([dur/(f_array.min())]).repeat(f_array.shape[0]),)
        processed_wavelet = process_wavelets_with_params(ricker_wavelet, dt, k_list=k, sigma_list=s).reshape(len(pos0) * len(pos1), -1)
        x, y = np.meshgrid(pos0, pos1, indexing='ij')
        points = np.column_stack((x.ravel(), y.ravel()))
        grid_x, grid_y = np.mgrid[0:n_patch[0], 0:n_patch[1]]
        
        rbf = Rbf(points[:,0], points[:,1], processed_wavelet, function='multiquadric',mode='N-D')
        processed_wavelet2 = rbf(grid_x, grid_y)
        
        rbf = Rbf(points[:,0], points[:,1], scale_noise, function='multiquadric')
        scale_full = rbf(grid_x, grid_y)
        processed_wavelet2 *= scale_full[:, :, np.newaxis] 
        # processed_wavelet = griddata(points, processed_wavelet, (grid_x, grid_y), method='cubic')
        # processed_wavelet = process_wavelets_with_params(ricker_wavelet, dt, k_list=k, sigma_list=s).reshape(1, n_patch[-1], -1).repeat(n_patch[-2],axis=0)
        # processed_wavelet = ricker_wavelet.numpy().reshape(1, n_patch[-1], -1).repeat(n_patch[-2],axis=0)
        seis_slice = wavelet_forward(torch.from_numpy(processed_wavelet2).float().to(device), window, imp_slice.to(device))
        mean = 0
        std = np.random.uniform(0, 1)
        kernel = np.random.randint(10, 30) * 2 + 1
        noise = mean + std * torch.randn(seis_slice.shape).to(device)
        seis_slice += gaussian_smoothing(noise, kernel, kernel/7)
        
        cube_i_mn = imp_slice.min()
        cube_i_mx = imp_slice.max()
        seis_i_mn = seis_slice.min()
        seis_i_mx = seis_slice.max()
        
        imp_slice = (imp_slice - cube_i_mn) / (cube_i_mx-cube_i_mn)
        seis_slice = seis_slice / max(seis_i_mn.abs(), seis_i_mx.abs()) * (torch.rand(1).to(device) * (1 - 0.3) + 0.3) * 0.5 + 0.5
        
        dic = {
                'seis':seis_slice[0,0].cpu().to(torch.float16),
                'imp':imp_slice[0,0].cpu().to(torch.float16),
            }
        torch.save(dic, output_folder + f'{os.path.basename(input_path).split(".")[0]}_{cube_i_mn:.2f}_{cube_i_mx:.2f}_{seis_i_mn:.2f}_{seis_i_mx:.2f}_{dt*1000}_{f0}_iy:{iy}.pth')      


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
    k = np.round((0.01-0.03)/(60-20) * f0 + 0.04, 2)
    s = np.round((0.004 - 0.01) / (60 - 20) * f0 + 0.013, 2)
    lobe_mask = identify_lobes(ricker_wavelet.numpy(),k=k, dt=dt)
    smooth_mask = gaussian_smooth_lobe_mask(ricker_wavelet.numpy(), dt=dt, k=k, sigma=0.004)
    perturbed_wave = add_phase_perturbation(ricker_wavelet,max_phase_shift=1.5)
    ricker_wavelet = smooth_mask * perturbed_wave
    ricker_wavelet = torch.from_numpy(ricker_wavelet / np.max(np.abs(ricker_wavelet))).to(torch.float32)  # Normalize.
    
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
    
    cube_i_mn = imp_interp.min()
    cube_i_mx = imp_interp.max()
    seis_i_mn = seis.min()
    seis_i_mx = seis.max()
    imp_interp = (imp_interp - cube_i_mn) / (cube_i_mx-cube_i_mn)
    seis = (seis - seis_i_mn) / (seis_i_mx-seis_i_mn) 
    number = random.choice([0, 1, 2, 3])
    for ix in range(number,nx,4):
        dic = {
                'seis':seis[ix].cpu().to(torch.float16),
                'imp':imp_interp[ix].cpu().to(torch.float16),
            }
        torch.save(dic, output_folder + f'{os.path.basename(input_path).split(".")[0]}_{cube_i_mn:.2f}_{cube_i_mx:.2f}_{seis_i_mn:.2f}_{seis_i_mx:.2f}_{dt*1000}_{f0}_ix:{ix}.pth')
    for iy in range(number,ny,4):
        dic = {
                'seis':seis[:,iy,:].cpu().to(torch.float16),
                'imp':imp_interp[:,iy,:].cpu().to(torch.float16),
            }
        torch.save(dic, output_folder + f'{os.path.basename(input_path).split(".")[0]}_{cube_i_mn:.2f}_{cube_i_mx:.2f}_{seis_i_mn:.2f}_{seis_i_mx:.2f}_{dt*1000}_{f0}_iy:{iy}.pth')      


def reset_dataset(accelerator, device, initial=False):
    process_index = accelerator.process_index
    input_folder = "./data/seis/dataset/"  # Replace with the directory containing the .npz files.
    output_folder = "./data/seis/imp_seis_01norm_nostandwavelet_all_morenoise/"  # Replace with the destination directory for results.
    os.makedirs(output_folder, exist_ok=True)
    if process_index == 0:
        print('Delete original data')
        for filename in os.listdir(output_folder):
            if len(filename) >= 5 and filename[:5].isdigit():
                prefix = filename[:5]  # Take the first five digits.
                if "00000" <= prefix <= "00950":
                    file_path = os.path.join(output_folder, filename)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        pass
    accelerator.wait_for_everyone()     
    print('len(os.listdir(output_folder)):',len(os.listdir(output_folder)))
    print()
    # Find all .npz files.
    delete = ['00012.npz','00023.npz','00025.npz','00034.npz','00037.npz','00041.npz','00048.npz','00084.npz','00092.npz','00095.npz',
    '00109.npz','00123.npz','00152.npz','00198.npz','00230.npz','00248.npz','00257.npz','00273.npz','00274.npz','00281.npz','00291.npz',
    '00309.npz','00311.npz','00322.npz','00328.npz','00428.npz','00451.npz','00456.npz','00462.npz','00464.npz','00468.npz','00478.npz','00489.npz',
    '00495.npz','00536.npz','00540.npz','00548.npz','00555.npz','00573.npz','00575.npz','00579.npz','00600.npz','00604.npz','00630.npz',
    '00638.npz','00641.npz','00658.npz','00685.npz','00686.npz','00687.npz','00698.npz','00705.npz','00706.npz','00711.npz','00717.npz','00740.npz',
    '00757.npz','00774.npz','00785.npz','00801.npz','00824.npz','00846.npz','00876.npz','00922.npz','00938.npz','00942.npz','00948.npz','00982.npz',
    '00995.npz']
    # delete = []
    npz_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npz') and f not in delete]
    npz_files.sort()
    if process_index == 0:
        npz_files = [i for i in npz_files if int(os.path.basename(i).split('.')[0]) in list(range(0,500))]
    elif process_index == 1:
        if len(os.listdir(output_folder))>0:
            npz_files = [i for i in npz_files if int(os.path.basename(i).split('.')[0]) in list(range(500,950))]
        else:
            npz_files = [i for i in npz_files if int(os.path.basename(i).split('.')[0]) in list(range(500,1000))]
     
    with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust max_workers as needed.
        # futures = [executor.submit(process_file, input_path, output_folder) for input_path in npz_files]
        futures = list(tqdm(executor.submit(process_file2, input_path, output_folder, device) for input_path in npz_files))
        # Display progress with tqdm.
        for future in tqdm(as_completed(futures), total=len(futures)):
            # Handle future.result() here if needed.
            pass
def copy_Files(destination):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    for filepath, dirnames, filenames in os.walk(r'./train'):
        for filename in filenames:
            if filename.endswith('.py') or filename.endswith('.ipynb') or filename.endswith('.sh') or filename.endswith('.json') or filename.endswith('.yml'):
                copy(os.path.join(filepath, filename), os.path.join(destination_dir, filename))
    for filepath, dirnames, filenames in os.walk(r'./data'):
        for filename in filenames:
            if filename.endswith('.py') or filename.endswith('.ipynb') or filename.endswith('.sh') or filename.endswith('.json') or filename.endswith('.yml'):
                copy(os.path.join(filepath, filename), os.path.join(destination_dir, filename))

def identify_lobes(y, k=0.03, dt=0.001):
    """Identify main-lobe and sidelobe regions."""
    # Find the main lobe at the largest peak.
    main_lobe_idx = np.argmax(np.abs(y))
    
    # Find sidelobes at the other significant peaks.
    peaks, _ = find_peaks(np.abs(y), height=np.max(np.abs(y))*0.2, distance=int(0.05/dt))
    
    # Create a mask that is True in the main-lobe and sidelobe regions.
    lobe_mask = np.zeros_like(y, dtype=bool)
    lobe_width = int(k/dt)  # Width of each lobe in samples.
    
    # Mark the main lobe.
    lobe_mask[max(0, main_lobe_idx-lobe_width):min(len(y), main_lobe_idx+lobe_width)] = True
    
    # Mark the sidelobes.
    for p in peaks:
        if p != main_lobe_idx:  # Skip the main lobe.
            lobe_mask[max(0, p-lobe_width):min(len(y), p+lobe_width)] = True
    
    return lobe_mask

def gaussian_smooth_lobe_mask(y, k=0.03, dt=0.001, sigma=0.005):
    """
    Apply smooth Gaussian decay.
    Args:
        sigma: Gaussian standard deviation controlling the decay rate.
    """
    lobe_mask = identify_lobes(y, k, dt)
    
    # Create a distance field from each sample to the nearest lobe-region boundary.
    distance_field = np.zeros_like(y)
    
    # Find all boundary points.
    boundaries = np.where(np.diff(lobe_mask.astype(int)) != 0)[0]
    
    # Compute the distance from each point to the nearest boundary.
    for i in range(len(y)):
        if lobe_mask[i]:
            distance_field[i] = 0  # Inside the lobe regions.
        else:
            # Find the nearest boundary.
            if len(boundaries) > 0:
                nearest_boundary = boundaries[np.argmin(np.abs(boundaries - i))]
                distance = abs(i - nearest_boundary) * dt
                distance_field[i] = distance
    
    # Apply Gaussian decay.
    smooth_mask = np.exp(-(distance_field**2)/(2*sigma**2))
    
    # Keep values at 1.0 inside lobe regions.
    smooth_mask[lobe_mask] = 1.0
    
    return smooth_mask


def add_phase_perturbation(y, max_phase_shift=0.2):
    """Apply reproducible phase perturbations."""
    n = len(y)
    phase_shift = np.random.uniform(-max_phase_shift, max_phase_shift, n//2 + 1)
    y_fft = np.fft.rfft(y)
    y_fft_perturbed = y_fft * np.exp(1j * phase_shift)
    return np.fft.irfft(y_fft_perturbed, n)

if __name__ == "__main__":
    args = parse_args()
    main(args)
