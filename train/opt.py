"""Only the helpers required by the two 2D training entry points. No import-time data processing."""
import numpy as np
import torch
import torch.nn.functional as F

def imp2ref_torch(x, dim=-1):
    """
    Seismic impedance to reflectivity.
    x: impedance, tensor (can be 1D, 2D, 3D, or more dimensions)
    dim: the dimension along which reflectivity is calculated
    """
    if isinstance(x, torch.Tensor):
        diff = torch.diff(x, dim=dim,)
        avg = (x + torch.roll(x, shifts=-1, dims=dim))
        ref = torch.zeros_like(x)
    else:
        raise ValueError("Unsupported data type")
    
    slices = [slice(None)] * x.ndimension()
    slices[dim] = slice(0, -1)
    ref[tuple(slices)] = diff / avg[tuple(slices)]
    return ref

def ref2seis_torch(x, wavelet):
    """
    Reflectivity to seismic data.

    x: tensor-like (can be 1D/2D/3D tensor)
    wavelet: 1D tensor (wavelet function)
    dim: the dimension of time/depth to apply convolution
    """
    assert wavelet.ndimension() == 1, "wavelet must be 1D"
    
    if isinstance(x, torch.Tensor):
        wavelet = wavelet.to(x.device)  # Ensure wavelet is on the same device as x
        x = x.to(torch.float32)  # Ensure input is float32
        nt, ni, nx = x.shape
        # Unsqueeze wavelet to match the format for conv1d: (out_channels, in_channels, kernel_size)
        wavelet = wavelet.unsqueeze(0).unsqueeze(0)
        
        # We need to permute the dimensions so the convolution happens on the `dim` dimension
        x = x.reshape(nt,1,-1)
        x = x.permute(2,1,0)
        # Perform 1D convolution to simulate seismic data generation
        out = F.conv1d(x, wavelet, padding=wavelet.shape[2] // 2)
        out = out.permute(2,1,0)
        out = out.reshape(nt, ni, nx)
        

        return out
    else:
        raise ValueError("Unsupported data type")

def _get_time_torch(duration: float, dt: float, tc: float, sym: bool = True):
    """
    Helper function to generate time series.
    """
    n = int(duration / dt)
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
    return t * dt + tc

def ricker_torch(f, dt, duration, sym=True):
    """
    Generate Ricker wavelet.
    """
    t = _get_time_torch(duration, dt, 0, sym)
    # Use the torch implementation for Ricker wavelet
    w = (1 - 2 * (torch.pi * f * t) ** 2) * torch.exp(-(torch.pi * f * t) ** 2)
    return w, t

def interp_torch(imp, original_sampling_rate, target_sampling_rate):
    """
    Interpolate 3D data to a new sampling rate (only along the first dimension).
    
    imp: tensor of shape (N, H, W) -> (N: Time/Depth dimension, H and W: other dimensions)
    original_sampling_rate: float
    target_sampling_rate: float
    """
    # Get the original length of the first dimension (N)
    original_length = imp.shape[0]  
    
    # Calculate the original duration
    original_duration = (original_length - 1) * original_sampling_rate
    
    # Calculate target length based on the target sampling rate
    target_length = round(original_duration / target_sampling_rate) + 1
    
    # Perform interpolation
    # Add batch and channel dimensions for interpolation
    imp_unsqueezed = imp.permute(1, 2, 0)  # shape: (1, 1, N, H, W)
    
    # Use F.interpolate, but we only need to interpolate along the first dimension (time/depth)
    interp_result = F.interpolate(
        imp_unsqueezed,              # Input tensor
        size=(target_length),  # Target size: (target_length, H, W)
        mode='linear',               # Linear interpolation
        align_corners=True           # Align the corners for accurate interpolation
    )
    
    # Remove the added batch and channel dimensions
    resampled_imp = interp_result.permute(2,0,1)  # shape: (target_length, H, W)
    
    return resampled_imp
