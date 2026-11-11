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
from shutil import copy

import accelerate
from accelerate.utils import DistributedDataParallelKwargs
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
import numpy as np
import opt

import warnings
warnings.filterwarnings('ignore')

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed


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

class SeisDataset(Dataset):
    def __init__(self, root: str, split: str = "train",):
        """
        Initialize the custom dataset.
        Args:
            file_paths (list): List of data-file paths.
            transform (callable, optional): Data-preprocessing function.
        """
        self.exclude_keywords = list(range(950,1000))

        if split == 'train':
            self.file_paths = [
                os.path.join(root, file) for file in os.listdir(root)
                if os.path.isfile(os.path.join(root, file))  # Include files only, excluding subdirectories.
                and int(file.split('_')[0]) not in self.exclude_keywords # Filter files by the specified identifiers.
            ]
        else:
            self.file_paths = [
                os.path.join(root, file) for file in os.listdir(root)
                if os.path.isfile(os.path.join(root, file))  # Include files only, excluding subdirectories.
                and int(file.split('_')[0]) in self.exclude_keywords  # Filter files by the specified identifiers.
            ]
        print(split,' dataset:', len(self.file_paths))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load the sample at the requested index on demand.
        """
        file_path = self.file_paths[idx]
        # Load data from the file; adapt the loading logic if needed.
        data = torch.load(file_path)
        base_name = os.path.basename(file_path)
        
        imp_min = base_name.split('_')[1]
        imp_max = base_name.split('_')[2]
        seis_min = base_name.split('_')[3]
        seis_max = base_name.split('_')[4]
        dt = base_name.split('_')[5]
        f0 = base_name.split('_')[6]
        pos = base_name.split('_')[7].split('.')[0]
        
        image = data['imp'].unsqueeze(0).to(torch.float32)
        cond = data['seis'].unsqueeze(0).to(torch.float32)
        
        
        return {'image':image, 'mask':cond}  # The value 0 is an example label and can be replaced if needed.

@torch.no_grad()
def inference(min_num,max_num,cond_dataset,unet,scheduler,generator,num_inference_steps,device):
    print('\n******************************Begin inference!***********************************')
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
        
        imp_min, imp_max = torch.tensor(cond_dataset["imp_min"]).view(-1, 1, 1, 1).repeat(image_shape[0],1,1,1).to(device), torch.tensor(cond_dataset["imp_max"]).view(-1, 1, 1, 1).repeat(image_shape[0],1,1,1).to(device)
        seis_min, seis_max = torch.tensor(cond_dataset["seis_min"]).view(-1, 1, 1, 1).repeat(image_shape[0],1,1,1).to(device), torch.tensor(cond_dataset["seis_max"]).view(-1, 1, 1, 1).repeat(image_shape[0],1,1,1).to(device)
        # norm to truely
        cond_images_denorm = cond_images * 0.5 + 0.5
        cond_images_denorm = cond_images_denorm * (seis_max - seis_min) + seis_min
        # cond_images_norm = mean_filter(cond_images_norm, kernel_size=3)
    
    image = randn_tensor(image_shape, generator=generator, device=device, dtype=unet.dtype)
    well_images = torch.zeros_like(clean_images, device=clean_images.device) - 1
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        # 1. predict noise model_output
        model_output = unet(image, t, None).sample

        # 2. compute previous image: x_t -> x_t-1
        image = scheduler.step(model_output, t, image, generator=generator).prev_sample
    image = image.clamp(-1, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    print('*********************************End inference!************************************\n')
    return image, well_images

def quick_save_4images(images, well_image, save_dir, n_step):
    """Simplified helper for quickly saving four images."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 15))
    plt.subplot(3, 2, 1)
    plt.imshow(well_image[0,0,:,:].T.cpu(), cmap='jet', vmin=-1, vmax=1)
    plt.colorbar()
    plt.subplot(3, 2, 2)
    if well_image.max() >  1e-4:
        index = torch.where((well_image[0,0].sum(dim=-1)!=0*well_image.shape[-1]).cpu())[0].sort().values[1].item()
    else:
        index = 100
    plt.plot(images[:,index,:,0].T)
    plt.plot(well_image[0,0,index,:].cpu(),'--')
    for i in range(4):
        plt.subplot(3, 2, 2+i+1)
        plt.imshow(images[i,:,:,0].T, cmap='jet', vmin=-1, vmax=1)
        plt.colorbar()
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
    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
                    
                noise_scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))    

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
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
        # model = UNet2DConditionModel.from_pretrained('/home/hanlong/MyProject/DDPM/output/ddpm3_inp01norm_noattn_predv_before/checkpoint-16956', subfolder="unet", )

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
        # beta_start: float = 0.0001,
        # beta_end: float = 0.02,
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)
    print(noise_scheduler)
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # reset_dataset(accelerator, accelerator.device, True)
    # accelerator.wait_for_everyone()
    dataset = create_dataset(args.dataset_name, args)
    val_dataset = create_dataset(args.dataset_name, args, split='val')
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True
    )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers, drop_last=False
    # )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=(math.ceil(len(train_dataloader) * args.num_epochs  / args.gradient_accumulation_steps)),
    )
    # print('(PID):',accelerator.local_process_index,'train_dataloader',len(train_dataloader))
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

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

    
    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        
        # if epoch>0:
        #     reset_dataset(accelerator, accelerator.device)
        #     accelerator.wait_for_everyone()
        #     dataset = create_dataset(args.dataset_name, args)
        #     # val_dataset = create_dataset(args.dataset_name, args, split='val')
        #     train_dataloader = torch.utils.data.DataLoader(
        #         dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True
        #     )
        #     train_dataloader = accelerator.prepare(train_dataloader)
            
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["image"].to(weight_dtype)
            
            
            
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            

            # plt.figure()  
            # plt.imshow(clean_images[0,0].detach().T.cpu(),cmap='jet',vmin=-1,vmax=1)
            # plt.savefig('clean_images.png')
            # plt.close()
            # plt.figure()  
            # plt.imshow(well_images[0,0].detach().T.cpu(),cmap='jet',vmin=-1,vmax=1)
            # plt.title('time_step:{}'.format(timesteps[0]))
            # plt.savefig('noisy_images.png')
            # plt.close()
            

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_images, timesteps, encoder_hidden_states=None).sample
                

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
                    loss_noise = F.mse_loss(model_output.float(), target.float())
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                """
                Rewarding ControlNet
                """
                # noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
                # pred_original_sample_ = [
                #     noise_scheduler.step(noise_, t_.item(), noisy_latent_).pred_original_sample.to(weight_dtype) \
                #         for (noise_, t_, noisy_latent_) in zip(model_output, timesteps, noisy_images[:,0:1])
                # ]
                # pred_original_sample = torch.stack(pred_original_sample_)
                # loss_well = ((pred_original_sample - well_images) ** 2 * mask).sum() / (mask.sum() + 1e-8)
                # loss_smooth = smooth_loss(pred_original_sample[:,0].mean(2), window=10, threshold=2.0)
                
                # loss = loss_noise + loss_well + 20 * loss_smooth
                loss = loss_noise
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            loss_dic['n_step'] += 1

            # print(sss)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step == 1:
                        save_path = os.path.join(args.output_dir)
                        copy_Files(save_path)
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
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step,'sum_step':loss_dic['n_step']}
            loss_dic['accu_loss'] += loss.detach().item()
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
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
                    generator = torch.Generator(device=clean_images.device).manual_seed(0)
                    scheduler=noise_scheduler
                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())
                    test_id = -1
                    images,well_images_ = inference(
                        min_num=0,
                        max_num=0,
                        cond_dataset=val_dataset[test_id],
                        unet=unet,
                        scheduler=scheduler,
                        generator=generator,
                        num_inference_steps=300,
                        device=clean_images.device
                        )
                    
                    if args.use_ema:
                        ema_model.restore(unet.parameters())
                            
                    quick_save_4images(images, well_images_, save_dir=os.path.join('png_inference', os.path.basename(args.output_dir)), n_step=loss_dic["n_step"])
            
            
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
        logger.info(f"Saved state to {save_path}")
        
    accelerator.end_training()

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
        torch.save(dic, output_folder + f'{os.path.basename(input_path).split(".")[0]}_{slice_imp_min:.2f}_{slice_imp_max:.2f}_{slice_seis_min:.2f}_{slice_seis_max:.2f}_{dt*1000}_{f0}_ix:{ix}.pth')
    
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
        torch.save(dic, output_folder + f'{os.path.basename(input_path).split(".")[0]}_{slice_imp_min:.2f}_{slice_imp_max:.2f}_{slice_seis_min:.2f}_{slice_seis_max:.2f}_{dt*1000}_{f0}_ix:{ix}.pth')

def reset_dataset(accelerator, device, initial=False):
    process_index = accelerator.process_index
    input_folder = "./data/seis/dataset/"  # Replace with the directory containing the .npz files.
    output_folder = "./data/seis/imp_seis_01norm2/"  # Replace with the destination directory for results.
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
        
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed.
        # futures = [executor.submit(process_file, input_path, output_folder) for input_path in npz_files]
        futures = list(tqdm(executor.submit(process_file, input_path, output_folder, device) for input_path in npz_files))
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

if __name__ == "__main__":
    args = parse_args()
    main(args)
