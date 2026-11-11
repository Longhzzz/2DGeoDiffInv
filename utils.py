"""Shared input, physical-operator, metric and plotting functions.
The inversion algorithms themselves are defined in the six notebooks.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    ref[tuple(slices)] = diff / (avg[tuple(slices)] + 1e-8)
    return ref

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

def wavelet_forward(wavelet_mat, image, window, seis_min, seis_max, imp_min, imp_max):
    device = image.device
    image_denorm = (image * 0.5 + 0.5) * (imp_max -imp_min) + imp_min
    B, C, H, W = image_denorm.shape
    n_patch = (wavelet_mat.shape[0], wavelet_mat.shape[1])
    image_pred_ori_shaped = image_denorm.reshape(-1, H, W)
    ref_pred_ori = imp2ref_torch(image_pred_ori_shaped, dim=-1)
    # window = (50,50)
    if H % window[0] != 0:
        ref_pred_ori = torch.cat([ref_pred_ori, torch.zeros((B * C, window[0] - ref_pred_ori.shape[1] % window[0], ref_pred_ori.shape[2]), device=device)], dim=1)
    if W % window[1] != 0:
        ref_pred_ori = torch.cat([ref_pred_ori, torch.zeros((B * C, ref_pred_ori.shape[1], window[1] - ref_pred_ori.shape[2] % window[1]), device=device)], dim=2) 
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
    seis_pred_ori_return = torch.zeros((1, n_patch[0]*window[0], n_patch[1]*window[1]+2*(wavelet_mat.shape[-1]//2+1)),device=device)
    for i in range(n_patch[1]):
        seis_pred_ori_return[:,:,i*window[-1]:i*window[-1] + seis_pred_ori.shape[-1]] += seis_pred_ori[0,i,:,:]
    seis_pred_ori_return = seis_pred_ori_return[:,:H,(wavelet_mat.shape[-1]//2+1):(wavelet_mat.shape[-1]//2+1)+W].unsqueeze(0)
    # seis_pred_ori_return = ((seis_pred_ori_return - seis_min) / (seis_max - seis_min) - 0.5) / 0.5
    seis_pred_ori_return = seis_pred_ori_return / max(torch.abs(seis_min), torch.abs(seis_max))
    return seis_pred_ori_return

def gaussian_filter_2d(x, kernel_1d, padding):
    x = F.pad(x, (padding, padding, 0, 0), mode='reflect')
    x = F.conv2d(x, kernel_1d.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, padding, padding), mode='reflect')
    x = F.conv2d(x, kernel_1d.view(1, 1, -1, 1))
    return x

def ssim_2d(pred, target, data_range, window_size=11, sigma=1.5):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    padding = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    with torch.no_grad():
        x = pred[None, None, :, :]
        y = target[None, None, :, :]
        mu_x = gaussian_filter_2d(x, kernel_1d, padding)
        mu_y = gaussian_filter_2d(y, kernel_1d, padding)
        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = gaussian_filter_2d(x * x, kernel_1d, padding) - mu_x2
        sigma_y2 = gaussian_filter_2d(y * y, kernel_1d, padding) - mu_y2
        sigma_xy = gaussian_filter_2d(x * y, kernel_1d, padding) - mu_xy

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return float(ssim_map.mean())

def corrcoef_2d(pred, target):
    x = pred.flatten().double()
    y = target.flatten().double()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if float(denom) == 0:
        return float('nan')
    return float((x * y).sum() / denom)

def calc_metrics(pred, target, data_range):
    diff = pred - target
    mse = torch.mean(diff ** 2)
    rmse = float(torch.sqrt(mse))
    mae = float(torch.mean(torch.abs(diff)))
    psnr = float('inf') if rmse == 0 else 20 * math.log10(data_range / rmse)
    return {
        'RMSE': rmse,
        'PSNR': psnr,
        'SSIM': ssim_2d(pred, target, data_range),
        'MAE': mae,
        'CorrCoef': corrcoef_2d(pred, target),
    }

def load_case(folder, device, preset="framework"):
    folder = Path(folder)
    meta = json.loads((folder / "section_metadata.json").read_text(encoding="utf-8"))
    names = ["clean_section", "seis_slice", "initial_model", "well_image",
             "well_mask", "estimated_wavelet_mat"]
    arrays = {name: np.load(folder / (name + ".npy"), allow_pickle=False) for name in names}
    shape = tuple(meta["target_shape"])
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{folder.name}/{name} contains NaN or infinity.")
        if name != "estimated_wavelet_mat" and array.shape != shape:
            raise ValueError(f"Unexpected {name} shape: {array.shape}; expected {shape}.")
    if (arrays["well_image"] * (1 - arrays["well_mask"])).any():
        raise ValueError("The well condition must be zero outside the saved mask.")
    lo, hi = float(meta["imp_min"]), float(meta["imp_max"])
    if hi <= lo:
        raise ValueError("Impedance normalization bounds must have positive range.")
    tensor = lambda name: torch.from_numpy(arrays[name].copy()).float().to(device)[None, None]
    observed = tensor("seis_slice")
    amplitude = float(np.max(np.abs(arrays["seis_slice"])))
    if amplitude <= 0:
        raise ValueError("The seismic profile has zero amplitude.")
    initial = tensor("initial_model")
    if meta.get("initial_model_scale") == "normalized_0_1":
        initial = (initial - 0.5) / 0.5
    else:
        initial = ((initial - lo) / (hi - lo) - 0.5) / 0.5
    condition = observed / amplitude
    if preset == "framework":
        # Retain the original notebook's 0..1 -> -1..1 round trip.
        condition = (condition * 0.5 + 0.5 - 0.5) / 0.5
        scaled = condition * amplitude
        denoise_condition = scaled / torch.maximum(scaled.min().abs(), scaled.max().abs())
    else:
        denoise_condition = condition
    kernel_size = int(meta["initial_model_kernel"])
    if preset == "framework" and shape == (512, 512):
        # This is the exact kernel construction used by the archived test-set benchmark.
        axis = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2 * (kernel_size / 7) ** 2))
        kernel = kernel / kernel.sum()
    else:
        kernel = torch.from_numpy(gaussian_kernel(kernel_size, kernel_size / 7)).float()
    return dict(id=folder.name, metadata=meta, folder=folder, shape=shape,
                truth=torch.from_numpy(arrays["clean_section"].copy()).float(),
                observed=observed, condition=condition.detach(),
                denoise_condition=denoise_condition.detach(), well=tensor("well_image"),
                well_mask=tensor("well_mask"), initial=initial,
                wavelet=torch.from_numpy(arrays["estimated_wavelet_mat"].copy()).float().to(device),
                window=tuple(meta["forward_window"]), imp_min=lo, imp_max=hi,
                amplitude=amplitude, kernel=kernel.to(device)[None, None])

def fast_length(minimum):
    value = minimum
    while True:
        remaining = value
        for divisor in (2, 3, 5):
            while remaining % divisor == 0:
                remaining //= divisor
        if remaining == 1:
            return value
        value += 1

def prepare_background(data, implementation="conv2d"):
    """Cache the fixed background operator. FFT matches the archived comparison run."""
    kernel = data["kernel"]
    padding = kernel.shape[-1] // 2
    if implementation == "conv2d":
        return lambda image: F.conv2d(F.pad(image, (padding,) * 4, mode="reflect"), kernel)
    if implementation != "fft":
        raise ValueError("Background implementation must be conv2d or fft.")
    height, width = data["shape"]
    k = kernel.shape[-1]
    shape = (fast_length(height + 2 * padding + k - 1),
             fast_length(width + 2 * padding + k - 1))
    spectrum = torch.fft.rfft2(kernel.flip((-2, -1)), s=shape)
    def blur(image):
        padded = F.pad(image, (padding,) * 4, mode="reflect")
        full = torch.fft.irfft2(torch.fft.rfft2(padded, s=shape) * spectrum, s=shape)
        return full[..., k-1:k-1+height, k-1:k-1+width].contiguous()
    return blur

def physical_model(image, data):
    return ((image.detach() * 0.5 + 0.5).clamp(0, 1)
            * (data["imp_max"] - data["imp_min"]) + data["imp_min"]).float().cpu()

def plot_inputs(data):
    images = [data["truth"].numpy(), data["observed"][0, 0].detach().cpu().numpy(),
              physical_model(data["initial"], data)[0, 0].numpy(),
              data["well"][0, 0].detach().cpu().numpy()]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, image, title, cmap in zip(axes, images,
            ["True impedance", "Observed seismic", "Initial impedance", "Well condition"],
            ["jet", "gray", "jet", "jet"]):
        shown = ax.imshow(image.T, aspect="auto", cmap=cmap)
        ax.set_title(title); ax.set_xlabel("Lateral sample"); ax.set_ylabel("Time sample")
        fig.colorbar(shown, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig

def plot_results(data, mean, std):
    truth = data["truth"]
    images = [truth, mean, (mean - truth).abs(), std]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (ax, image, title) in enumerate(zip(axes, images,
            ["True impedance", "Mean inversion", "Absolute error", "Inversion standard deviation"])):
        options = dict(vmin=data["imp_min"], vmax=data["imp_max"]) if i < 2 else {}
        shown = ax.imshow(image.numpy().T, aspect="auto", cmap="jet" if i < 2 else "viridis", **options)
        ax.set_title(title); ax.set_xlabel("Lateral sample"); ax.set_ylabel("Time sample")
        fig.colorbar(shown, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig

def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()
