"""Launch the 2D conditional denoising VAE with the original training settings on Windows, Linux, or macOS.

Examples:
python train_vae.py --train-data-dir data/my_training --seismic-source-dir data/raw_npz --validation-data-dir data/my_validation
python train_vae.py --train-data-dir data/my_training --seismic-source-dir data/raw_npz --dry-run

Defaults follow finetune_vae_seis3_inp01norm_well_sigma06_noisecond.sh.
num_epochs=50 is the planned training budget; the released checkpoint is actual step 120500.
Regenerate 2D data from the original NPZ models in seismic-source-dir each epoch and write it to train-data-dir.
If only original NPZ models are available, use --refresh-before-training to initialize the first epoch; existing slices are used by default.
"""
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data-dir", type=Path, required=True,
                        help="Directory of 2D imp/seis training data; refreshed data are written here each epoch.")
    parser.add_argument("--seismic-source-dir", type=Path, required=True,
                        help="Directory of original impedance NPZ models used to generate seismic data each epoch.")
    parser.add_argument("--validation-data-dir", type=Path)
    parser.add_argument("--refresh-before-training", action="store_true",
                        help="Initialize the 2D training directory from original NPZ models before the first epoch; existing first-epoch data are used by default.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/vae2d_retrain"))
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--main-process-port", type=int, default=26789)
    parser.add_argument("--gpu-ids", help="For example, 0,1. By default, use the devices visible in the current environment.")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting training.")
    args = parser.parse_args()
    if min(args.num_processes, args.num_epochs, args.batch_size,
           args.eval_batch_size, args.gradient_accumulation_steps) < 1:
        parser.error("Process count, epoch count, batch sizes, and gradient accumulation steps must be positive.")

    root = Path(__file__).resolve().parent
    # Resolve relative paths from the release root so this script can be called from another working directory.
    train_data = (root / args.train_data_dir).resolve()
    seismic_source = (root / args.seismic_source_dir).resolve()
    validation_data = (root / (args.validation_data_dir or args.train_data_dir)).resolve()
    output = (root / args.output_dir).resolve()
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(args.num_processes),
        "--num_machines", "1", "--machine_rank", "0",
        "--main_process_port", str(args.main_process_port),
        "--mixed_precision", args.mixed_precision,
    ]
    if args.num_processes > 1:
        command.append("--multi_gpu")
    if args.gpu_ids:
        command.extend(["--gpu_ids", args.gpu_ids])
    command.extend([
        str(root / "train" / "train_vae3_inp01norm_well.py"),
        "--dataset_name", "Seis",
        "--train_data_dir", str(train_data),
        "--seismic_source_dir", str(seismic_source),
        "--validation_data_dir", str(validation_data),
        "--resolution", "512",
        "--output_dir", str(output),
        "--train_batch_size", str(args.batch_size),
        "--eval_batch_size", str(args.eval_batch_size),
        "--num_epochs", str(args.num_epochs),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--lr_warmup_steps", "500",
        "--mixed_precision", args.mixed_precision,
        "--checkpointing_steps", "500",
        "--logger", "wandb",
        "--dataloader_num_workers", str(args.workers),
        "--latent_dim", "512",
        "--block_out_channels", "128,128,256,256,512,512,512",
        "--layers_per_block", "3",
        "--recon_loss", "l1",
        "--kl_weight", "1e-4",
        "--kl_warmup_steps", "2000",
        "--noise_std_min", "0.0",
        "--noise_std_max", "1.0",
        "--noise_sampling_strategy", "uniform",
        "--noise_conditioning",
        "--no-well_mask_conditioning",
        "--no-clamp_noisy_input",
        "--validation_noise_std", "1.0",
        "--min_wells", "1",
        "--max_wells", "5",
        "--well_interval", "1",
        "--well_drop_prob", "0.5",
        "--well_loss_weight", "0.0",
        "--refresh_seismic_each_epoch",
    ])
    if args.refresh_before_training:
        command.append("--refresh_seismic_before_training")
    if args.resume_from_checkpoint:
        command.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])
    print(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command))
    if args.dry_run:
        return
    # Validate refresh inputs before launching training to catch invalid paths before the second epoch.
    if not seismic_source.is_dir() or not any(p.is_file() for p in seismic_source.glob("*.npz")):
        parser.error("seismic-source-dir must exist and contain original .npz files.")
    if seismic_source == train_data:
        parser.error("The original NPZ source directory and the 2D training-data output directory must be different.")
    if train_data.exists() and not train_data.is_dir():
        parser.error("train-data-dir must be a directory.")
    if not args.refresh_before_training and not train_data.is_dir():
        parser.error("The initial 2D training directory must exist. If only original NPZ models are available, add --refresh-before-training.")
    if validation_data != train_data and not validation_data.is_dir():
        parser.error("The explicitly specified validation-data directory must exist.")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
