# 2DGeoDiffInv

**A Plug-and-Play Diffusion Regularizer for Seismic Inversion**

This repository provides the code, pretrained models, and example data for the 2D experiments in the paper.

## Abstract

Seismic inversion is the mathematical process of transforming seismic measurements into subsurface physical models, serving as a cornerstone for both probing the Earth's interior and exploring resources. In seismic inversion, current model-driven iterative inversion methods rely on regularization to mitigate ill-posedness but struggle to balance stability and resolution due to inaccurate prior representations. Generative priors, particularly diffusion models, provide a promising way to overcome the limitations of conventional regularization by capturing complex geological distributions, yet existing diffusion-based seismic inversion methods often lack a universal framework and exhibit limited generalization across diverse geological models. Thus, we propose a plug-and-play diffusion regularizer for seismic inversion and an accompanying inversion framework, enabling seamless integration with various inversion algorithms and generalization across different geological models without retraining. The framework decouples each iteration of model-based inversion into a data-fitting step updated by various inversion algorithms and a diffusion regularization step implemented via a noise-addition and denoising process. To achieve a highly generalizable diffusion regularizer, we constructed numerous complex geological models to train an unconditional diffusion model to learn accurate geological priors with high-resolution. Subsequently, by utilizing two independent ControlNets, we incorporated post-stack seismic data and sparse well-log data as conditions to provide global structural and local fine-scale constraints, further narrowing down the solution space. Experimental results on impedance inversion demonstrate that the proposed diffusion regularizer can be flexibly embedded into various inversion algorithms within our framework, achieving high-precision and high-resolution inversion results across the test dataset, the Marmousi model and the field data. Repeated inversions with independent perturbation paths further provide an empirical assessment of inversion uncertainty.

## Repository layout

```text
2DGeoDiffInv/
|-- README.md
|-- CITATION.cff                        # Software and preferred paper citation
|-- requirements.txt                    # Shared inference/training dependencies
|-- inference_*.ipynb                   # 29 experiment notebooks, listed below
|-- inference_utils.py
|-- inversion_utils.py
|-- myddim_schedule.py
|-- paper_plotting.py
|-- utils.py
|-- train_vae.py
|-- models/
|   |-- __init__.py
|   |-- dip.py
|   |-- inr.py
|   `-- vae2d.py
|-- train/
|   |-- train_ddpm3_inp01norm.py
|   |-- train_ddpm3_mutilcontrolnet_well_nofullwell.py
|   |-- train_vae3_inp01norm_well.py
|   |-- opt.py
|   |-- vae2d.py
|   |-- config.yml                      # Multi-process Accelerate configuration
|   |-- config_single.yml               # Single-process Accelerate configuration
|   |-- unet_2d_config2_deep.json
|   |-- unet_2d_config2_deep_noattn.json
|   |-- finetune_*.sh                    # Three training launchers described below
|   `-- mydatasets/
|       |-- __init__.py
|       |-- build.py
|       |-- seis_dataset.py
|       `-- transforms_image.py
|-- data/
|   |-- testset/                        # Main article profile + 10 appendix profiles
|   |-- marmousi/marmousi_2300x730/
|   `-- sensitivity/section_main_vol00983_slice_363/
|-- weights/
|   |-- checkpoint-22608/
|   |   |-- unet/
|   |   |-- controlnet/
|   |   `-- scheduler/
|   `-- vae-checkpoint-120500/vae/
|-- licenses/                           # Third-party license texts
|-- LICENSE
|-- .gitattributes
`-- .gitignore
```

The ordinary profile folders contain impedance, seismic data, well conditions, an initial model, metadata, and a saved wavelet. Each weight directory retains the configuration required to reconstruct its model. New runs create `results/` automatically.

### Python files

| File | Role |
|---|---|
| [inference_utils.py](inference_utils.py) | Provides the article impedance palettes (Rainbow_Plus for test profiles and terrain for Marmousi). |
| [inversion_utils.py](inversion_utils.py) | Shared Ricker-wavelet fitting, interpolation, Gaussian background filtering, and initial-model and seismic-loss helpers used by the notebooks. |
| [myddim_schedule.py](myddim_schedule.py) | Custom DDIM scheduler used for noise addition and reverse diffusion, including the timestep and prediction conventions of the experiments. |
| [paper_plotting.py](paper_plotting.py) | Formats section layouts, axes, colorbars, and well-location overlays. These functions control display, not inversion updates. |
| [utils.py](utils.py) | Impedance-to-reflectivity and seismic forward operators, Gaussian filters, evaluation metrics, and shared data/plot utilities. |
| [models/__init__.py](models/__init__.py) | Package initializer for the model definitions. |
| [models/dip.py](models/dip.py) | Defines the U-Net that parameterizes impedance updates in neural network reparameterization. This is separate from the pretrained diffusion U-Net. |
| [models/inr.py](models/inr.py) | Builds 2D coordinates and the HashGrid encoding plus MLP used for INR inversion. |
| [models/vae2d.py](models/vae2d.py) | Defines the conditional denoising VAE architecture, latent representation, and checkpoint save/load methods. |
| [train_vae.py](train_vae.py) | Portable command-line launcher for VAE training. Converts user-supplied paths and options into an Accelerate command and supports --dry-run. |
| [train/train_ddpm3_inp01norm.py](train/train_ddpm3_inp01norm.py) | Trains the unconditional DDPM U-Net on normalized 2D impedance data. |
| [train/train_ddpm3_mutilcontrolnet_well_nofullwell.py](train/train_ddpm3_mutilcontrolnet_well_nofullwell.py) | Trains the seismic and sparse-well ControlNets while freezing the pretrained diffusion U-Net. |
| [train/train_vae3_inp01norm_well.py](train/train_vae3_inp01norm_well.py) | Trains and validates the conditional denoising VAE, including noise sampling, well conditions, and seismic-data refreshes. |
| [train/opt.py](train/opt.py) | Wavelet, interpolation, reflectivity, and seismic-forward helpers used by the training programs. |
| [train/vae2d.py](train/vae2d.py) | Compatibility import used by the VAE trainer to reuse models/vae2d.py instead of duplicating the model definition. |
| [train/mydatasets/__init__.py](train/mydatasets/__init__.py) | Exports the training dataset factory. |
| [train/mydatasets/build.py](train/mydatasets/build.py) | Constructs the training/validation dataset and its paired impedance/seismic transforms. |
| [train/mydatasets/seis_dataset.py](train/mydatasets/seis_dataset.py) | Loads paired impedance and seismic PTH slices and associated filename metadata, and selects the training or validation split. |
| [train/mydatasets/transforms_image.py](train/mydatasets/transforms_image.py) | Paired crop, normalization, and image-transform utilities that keep impedance and seismic samples aligned. |

### Inference notebooks

The method names follow the paper: traditional inversion, neural network reparameterization (Reparam.), and implicit neural representation (INR).

Each ordinary notebook handles the single profile selected by `data_dir`. The test-set and Marmousi columns below use the same method on their respective inputs. Inversion loops, method-specific settings, figures, and evaluation remain visible in the notebooks.

#### Diffusion regularization

| Method | Test-set notebook | Marmousi notebook |
|---|---|---|
| Traditional inversion with diffusion regularization | [inference_testset_traditional.ipynb](inference_testset_traditional.ipynb) | [inference_marmousi_traditional.ipynb](inference_marmousi_traditional.ipynb) |
| Neural network reparameterization with diffusion regularization | [inference_testset_reparameterization.ipynb](inference_testset_reparameterization.ipynb) | [inference_marmousi_reparameterization.ipynb](inference_marmousi_reparameterization.ipynb) |
| INR HashGrid/MLP parameterization with diffusion regularization | [inference_testset_inr.ipynb](inference_testset_inr.ipynb) | [inference_marmousi_inr.ipynb](inference_marmousi_inr.ipynb) |

#### TV and Tikhonov baselines

These notebooks optimize the physical fitting and regularization objectives without diffusion denoising.

| Method | Test-set notebook | Marmousi notebook |
|---|---|---|
| Traditional inversion + TV | [inference_testset_traditional_tv.ipynb](inference_testset_traditional_tv.ipynb) | [inference_marmousi_traditional_tv.ipynb](inference_marmousi_traditional_tv.ipynb) |
| Traditional inversion + Tikhonov | [inference_testset_traditional_tikhonov.ipynb](inference_testset_traditional_tikhonov.ipynb) | [inference_marmousi_traditional_tikhonov.ipynb](inference_marmousi_traditional_tikhonov.ipynb) |
| Neural network reparameterization + TV | [inference_testset_reparameterization_tv.ipynb](inference_testset_reparameterization_tv.ipynb) | [inference_marmousi_reparameterization_tv.ipynb](inference_marmousi_reparameterization_tv.ipynb) |
| Neural network reparameterization + Tikhonov | [inference_testset_reparameterization_tikhonov.ipynb](inference_testset_reparameterization_tikhonov.ipynb) | [inference_marmousi_reparameterization_tikhonov.ipynb](inference_marmousi_reparameterization_tikhonov.ipynb) |
| INR + TV | [inference_testset_inr_tv.ipynb](inference_testset_inr_tv.ipynb) | [inference_marmousi_inr_tv.ipynb](inference_marmousi_inr_tv.ipynb) |
| INR + Tikhonov | [inference_testset_inr_tikhonov.ipynb](inference_testset_inr_tikhonov.ipynb) | [inference_marmousi_inr_tikhonov.ipynb](inference_marmousi_inr_tikhonov.ipynb) |

#### Framework and VAE comparisons

| Method | Test-set notebook | Marmousi notebook |
|---|---|---|
| DPS diffusion posterior-sampling baseline | [inference_testset_dps.ipynb](inference_testset_dps.ipynb) | [inference_marmousi_dps.ipynb](inference_marmousi_dps.ipynb) |
| DMPlug latent-optimization baseline | [inference_testset_dmplug.ipynb](inference_testset_dmplug.ipynb) | [inference_marmousi_dmplug.ipynb](inference_marmousi_dmplug.ipynb) |
| DiffPIR diffusion-based inversion baseline | [inference_testset_diffpir.ipynb](inference_testset_diffpir.ipynb) | [inference_marmousi_diffpir.ipynb](inference_marmousi_diffpir.ipynb) |
| Proposed method under the framework-comparison configuration | [inference_testset_ours_comparison.ipynb](inference_testset_ours_comparison.ipynb) | [inference_marmousi_ours_comparison.ipynb](inference_marmousi_ours_comparison.ipynb) |
| Conditional denoising VAE inversion baseline | [inference_testset_vae.ipynb](inference_testset_vae.ipynb) | Not included |

#### Sensitivity tests

| Notebook | Role |
|---|---|
| [inference_noise.ipynb](inference_noise.ipynb) | Compares inversions at 30/20/10/5 dB SNR with the dominant frequency fixed at 25 Hz. |
| [inference_frequency.ipynb](inference_frequency.ipynb) | Compares inversions at 15/25/40/60 Hz with SNR fixed at 20 dB. |

## Training

Use Python 3.10 and install a CUDA-enabled PyTorch build suitable for your GPU. From the repository root, install the shared inference and training dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare the full training data separately. The bundled test and Marmousi profiles are inference examples. Set the training-data and output paths in the appropriate launcher before starting a job. All three launchers use `train/config.yml`, which currently specifies two processes and GPU IDs `2,7`; adjust these hardware settings for your machine.

The training entry points are:

| Stage | Training code | Launcher |
|---|---|---|
| Unconditional DDPM | `train/train_ddpm3_inp01norm.py` | `train/finetune_unet_seis3_inp01norm_predv_nohup.sh` |
| Seismic and well ControlNets | `train/train_ddpm3_mutilcontrolnet_well_nofullwell.py` | `train/finetune_unet_seis3_mutilcontrolnet_well_nohup.sh` |
| Conditional denoising VAE | `train/train_vae3_inp01norm_well.py` | `train/finetune_vae_seis3_inp01norm_well_sigma06_noisecond.sh` |

Run the launcher for the stage you need from the repository root:

```bash
# Train the unconditional DDPM.
bash train/finetune_unet_seis3_inp01norm_predv_nohup.sh

# Train the two ControlNets with the U-Net frozen.
bash train/finetune_unet_seis3_mutilcontrolnet_well_nohup.sh

# Train the conditional denoising VAE baseline.
bash train/finetune_vae_seis3_inp01norm_well_sigma06_noisecond.sh
```

Each command starts a background job with `nohup` and writes a log file. The DDPM launchers use `train/unet_2d_config2_deep_noattn.json`. ControlNet training loads the pretrained U-Net from `weights/checkpoint-22608/unet/`; it is byte-identical to the original pretrained U-Net used for this stage. The launchers retain the training hyperparameters, which can be inspected directly in the scripts. Use Bash on Linux or a Bash environment such as Windows WSL for these launchers.

### ControlNet training-data refresh

The original ControlNet trainer refreshes generated training slices during later epochs. Its `reset_dataset()` function uses these relative directories:

| Directory | Contents |
|---|---|
| `data/seis/dataset/` | Original impedance NPZ volumes used to regenerate training slices |
| `data/seis/imp_seis_01norm_nostandwavelet_all_morenoise/` | Generated impedance/seismic PTH slices, matching the supplied ControlNet launcher's training directory |

These paths are set inside `train/train_ddpm3_mutilcontrolnet_well_nofullwell.py`; changing only the launcher argument does not change the refresh paths. The supplied refresh routine partitions source-file IDs between process indices 0 and 1. Keep two processes for this original training workflow, or adapt that partitioning before changing the process count. GPU IDs may be changed while retaining two processes.

Refreshes remove and regenerate numeric-ID training slices in the slice-output directory. Use a dedicated training-slice directory and keep the original NPZ source volumes separate. This workflow does not use or modify the bundled inference profiles under `data/testset/`, `data/marmousi/`, or `data/sensitivity/`.

A portable Python launcher is also provided for VAE training:

```bash
python train_vae.py \
    --train-data-dir data/my_training \
    --validation-data-dir data/my_validation \
    --seismic-source-dir data/my_raw_npz \
    --dry-run
```

`--dry-run` prints the training command without starting a job. The seismic source directory must contain the original impedance NPZ models with `imp` arrays in `[time, x, y]` order. The training directory contains generated 2D `imp/seis` PTH slices and receives subsequent epoch refreshes. Keep these two directories separate. Add `--refresh-before-training` if the first set of slices has not been generated. A separately specified validation directory must already exist. Remove `--dry-run` when the data and configuration are ready.

## Inference

Install [Git LFS](https://git-lfs.com/) before cloning so that model and data files are downloaded as complete files:

```bash
git lfs install
git clone https://github.com/Longhzzz/2DGeoDiffInv.git
cd 2DGeoDiffInv
git lfs pull
```

The same requirements.txt covers inference and training. Install it and start Jupyter from the repository root:

```bash
python -m pip install -r requirements.txt
python -m notebook
```

The INR HashGrid implementation additionally requires an NVIDIA CUDA GPU and the pinned tiny-cuda-nn dependency. Its fixed revision and build commands are recorded in the optional INR section of requirements.txt. Install it after CUDA-enabled PyTorch and the shared dependencies, because the [extension build script](https://github.com/NVlabs/tiny-cuda-nn/blob/147741674580e12be8151c33dbd0950ead16d1f3/bindings/torch/setup.py) imports PyTorch:

```bash
python -m pip install setuptools==78.1.1 wheel==0.45.1 ninja==1.13.0
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn@147741674580e12be8151c33dbd0950ead16d1f3#subdirectory=bindings/torch"
```

The HashGrid backend does not support macOS CPU/MPS. Select the compute device in the notebook's device cell. Traditional inversion and neural network reparameterization notebooks can use PyTorch on CPU, although execution is slower.

Choose the notebook for the desired experiment:

| Experiment | Notebook files |
|---|---|
| Diffusion regularization with traditional inversion, neural network reparameterization, or INR | `inference_{testset,marmousi}_{traditional,reparameterization,inr}.ipynb` |
| TV and Tikhonov baselines | `inference_{testset,marmousi}_{traditional,reparameterization,inr}_{tv,tikhonov}.ipynb` |
| DPS, DMPlug, DiffPIR, and the proposed framework | `inference_{testset,marmousi}_{dps,dmplug,diffpir,ours_comparison}.ipynb` |
| Conditional denoising VAE | `inference_testset_vae.ipynb` |
| Noise sensitivity | `inference_noise.ipynb` |
| Dominant-frequency sensitivity | `inference_frequency.ipynb` |

Braces denote separate files. Each ordinary inference notebook processes one profile selected by `data_dir`. For the main article test profile, use:

```python
data_dir = ROOT / "data" / "testset" / "section_main_vol00983_slice_363"
```

The other ten test profiles are in `data/testset/`. The Marmousi profile is in `data/marmousi/marmousi_2300x730/`. Each profile includes impedance, observed seismic data, well logs and masks, an initial model, metadata, and a saved wavelet. The noise and frequency notebooks use the observation settings supplied in `data/sensitivity/section_main_vol00983_slice_363/`.

Run the notebook cells in order. They load the model and data, display the inputs, prepare the well conditions and initial model, estimate or load the wavelet, run inversion, and display and save the results. The parameterization and regularization notebooks estimate the wavelet from the observations and well logs. Framework comparisons and VAE expose `reestimate_wavelet` to select between saved and re-estimated wavelets. Sensitivity tests use the supplied known wavelets.

The diffusion notebooks load `weights/checkpoint-22608/`, which contains the U-Net, both ControlNets, and scheduler configuration. The VAE notebook loads `weights/vae-checkpoint-120500/vae/`. Run from the repository root so these relative paths resolve correctly.

New results are saved under the notebook's `output_dir` in `results/`, including estimated wavelets, inverted impedance, metrics, figures, and ensemble statistics where applicable. Saved notebook outputs are retained for inspection. Display color limits and well markers do not change the inversion objective. The evaluation and ensemble-selection rules remain visible in each notebook.
## Data and model availability

This release provides author-supplied synthetic test profiles, the Marmousi example, derived seismic observations and well conditions, and the trained diffusion and VAE checkpoints used by the included 2D workflows. The main test profile is volume 00983, section 363; profile identifiers and preparation details are retained in each data folder's `section_metadata.json`. The complete training dataset and field dataset are not bundled.

The code is released under Apache-2.0. The authors declare no additional use or redistribution restrictions for the bundled example data and pretrained weights. Third-party components retain their applicable licenses and attributions in their source files and the `licenses/` directory.

## Citation

> [!IMPORTANT]
> **Please cite our paper when publishing research that uses this project.**
> If you use the code, pretrained models, or example data in this repository to conduct research and publish the resulting findings, please cite **A Plug-and-Play Diffusion Regularizer for Seismic Inversion**. The BibTeX entry is provided below.

The accompanying manuscript does not yet have a public URL or DOI. The citation will be updated when publication information becomes available.

```bibtex
@unpublished{han2026geodiffinv,
  title  = {A Plug-and-Play Diffusion Regularizer for Seismic Inversion},
  author = {Han, Long and Wu, Xinming and Sun, Xiaoming and Li, Jintao and Dou, Yimin},
  year   = {2026},
  note   = {Manuscript. Code and pretrained models: https://github.com/Longhzzz/2DGeoDiffInv}
}
```

Software citation metadata are provided in [CITATION.cff](CITATION.cff). The initial software release is `v1.0.0`.
