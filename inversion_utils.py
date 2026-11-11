"""Helpers extracted from the original inference notebook.

Wavelet fitting preserves the original Ricker/Adam arithmetic and returned
wavelet. The executable fitting call and the inversion loop stay in notebooks.
"""
from collections import defaultdict
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import gaussian_kernel, prepare_background, wavelet_forward, imp2ref_torch


def _get_time_torch(duration: float, dt: float, tc: float, sym: bool=True):
    """
    Helper function to generate time series.
    """
    n = int((duration / dt).max())
    odd = n % 2
    if odd and sym:
        t = torch.arange(n)
    elif not odd and sym:
        t = torch.arange(n + 1)
    elif odd and (not sym):
        t = torch.arange(n)
    elif not odd and (not sym):
        t = torch.arange(n) - 1
    t -= t[-1] // 2
    t = t.unsqueeze(0).repeat(dt.shape[0], 1).to(duration.device)
    return t * dt.unsqueeze(1) + tc


def interpolate_2d_by_positions(matrix, target_positions_x, target_positions_z, dim, mode='bilinear', align_corners=False):
    if matrix.ndim == 2:
        x = matrix.unsqueeze(0).unsqueeze(0)
    elif matrix.ndim == 3:
        x = matrix.unsqueeze(1)
    else:
        raise ValueError("Input must be a 2D or 3D tensor.")
    (B, C, H, W) = x.shape
    target_pos_x = target_positions_x.to(x.device)
    target_pos_z = target_positions_z.to(x.device)
    Nx = len(target_positions_x)
    Nz = len(target_positions_z)
    EPS = 1e-06
    (z_min, z_max) = (0.0, W - 1.0)
    divisor = max(z_max - z_min, EPS)
    target_pos_norm_z = 2 * (target_pos_z - z_min) / divisor - 1
    (x_min, x_max) = (0.0, H - 1.0)
    divisor = max(x_max - x_min, EPS)
    target_pos_norm_x = 2 * (target_pos_x - x_min) / divisor - 1
    grid_z = target_pos_norm_z.unsqueeze(0).expand(Nx, Nz)
    grid_x = target_pos_norm_x.unsqueeze(1).expand(Nx, Nz)
    grid = torch.stack([grid_z, grid_x], dim=-1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
    interpolated = F.grid_sample(input=x, grid=grid, mode=mode, align_corners=align_corners, padding_mode='border')
    return interpolated.squeeze(1) if matrix.ndim == 3 else interpolated.squeeze(0).squeeze(0)


def wavelet_fn(wavelet_mat, image, cond_image, window, seis_min, seis_max, imp_min, imp_max, scale, mask=None, f0=None, dt=None):
    device = image.device
    image_denorm = (image * 0.5 + 0.5) * (imp_max - imp_min) + imp_min
    (B, C, H, W) = image_denorm.shape
    n_patch = (wavelet_mat.shape[0], wavelet_mat.shape[1])
    image_pred_ori_shaped = image_denorm.reshape(-1, H, W)
    ref_pred_ori = imp2ref_torch(image_pred_ori_shaped, dim=-1)
    if H % window[0] != 0:
        ref_pred_ori = torch.cat([ref_pred_ori, torch.zeros((B * C, window[0] - ref_pred_ori.shape[1] % window[0], ref_pred_ori.shape[2]), device=device)], dim=1)
    if W % window[1] != 0:
        ref_pred_ori = torch.cat([ref_pred_ori, torch.zeros((B * C, ref_pred_ori.shape[1], window[1] - ref_pred_ori.shape[2] % window[1]), device=device)], dim=2)
    wavelet_mat = wavelet_mat.view(n_patch[0] * n_patch[1], -1).unsqueeze(1)
    unfold = nn.Unfold(kernel_size=window, stride=window)
    ref_pred_ori = unfold(ref_pred_ori.unsqueeze(1))
    ref_pred_ori = ref_pred_ori.view(1, window[0], window[1], n_patch[0], n_patch[1]).permute(0, 3, 4, 1, 2)
    ref_pred_ori = ref_pred_ori.contiguous().view(-1, window[0], window[1])
    ref_pred_ori = ref_pred_ori.permute(1, 0, 2)
    ref_pred_ori = F.pad(ref_pred_ori, (wavelet_mat.shape[-1], wavelet_mat.shape[-1]))
    seis_pred_ori = F.conv1d(ref_pred_ori, wavelet_mat, padding=0, stride=1, groups=n_patch[0] * n_patch[1])
    seis_pred_ori = seis_pred_ori.permute(1, 0, 2)
    seis_pred_ori = seis_pred_ori.view(1, n_patch[0], n_patch[1], window[0], seis_pred_ori.shape[-1]).permute(0, 2, 1, 3, 4).contiguous().view(1, n_patch[1], -1, seis_pred_ori.shape[-1])
    seis_pred_ori_return = torch.zeros((1, n_patch[0] * window[0], n_patch[1] * window[1] + 2 * (wavelet_mat.shape[-1] // 2 + 1)), device=device)
    for i in range(n_patch[1]):
        seis_pred_ori_return[:, :, i * window[-1]:i * window[-1] + seis_pred_ori.shape[-1]] += seis_pred_ori[0, i, :, :]
    seis_pred_ori_return = seis_pred_ori_return[:, :H, wavelet_mat.shape[-1] // 2 + 1:wavelet_mat.shape[-1] // 2 + 1 + W].unsqueeze(0)
    seis_pred_ori_return = seis_pred_ori_return / max(torch.abs(seis_min), torch.abs(seis_max))
    wavelet_mat_shaped = wavelet_mat.reshape(n_patch[0], n_patch[1], wavelet_mat.shape[-1])
    match_loss = torch.linalg.norm(seis_pred_ori_return - cond_image) if mask is None else torch.linalg.norm((seis_pred_ori_return - cond_image) * mask) / mask.sum() * mask.numel()
    diff_nx = torch.diff(wavelet_mat_shaped, dim=0, n=1)
    tv_nx = torch.linalg.norm(diff_nx)
    diff_nz = torch.diff(wavelet_mat_shaped, dim=1, n=1)
    tv_nz = torch.linalg.norm(diff_nz)
    diff_nt = torch.diff(wavelet_mat_shaped, dim=2, n=1)
    tv_nt = torch.linalg.norm(diff_nt)
    wavelet_energy = wavelet_mat_shaped ** 2
    wavelet_length = wavelet_mat_shaped.shape[2]
    max_pos = torch.argmax(wavelet_energy, dim=2, keepdim=True)
    if f0 is not None and f0 > 0:
        main_lobe_points = int(2 / (f0 * dt))
        main_lobe_points = min(main_lobe_points, wavelet_length // 2)
        main_lobe_ratio = main_lobe_points / wavelet_length
    else:
        main_lobe_ratio = 0.2
    main_lobe_half = int(wavelet_length * main_lobe_ratio / 2)
    transition_band_half = int(main_lobe_half * 0.5)
    time_indices = torch.arange(wavelet_length, device=device).unsqueeze(0).unsqueeze(0)
    distance_from_center = torch.abs(time_indices - max_pos)
    steepness = 5.0 / (transition_band_half + 1e-08)
    transition_start = main_lobe_half
    transition_end = main_lobe_half + transition_band_half
    sidelobe_weights = torch.sigmoid(steepness * (distance_from_center - transition_start))
    sidelobe_weights = torch.min(sidelobe_weights, torch.tensor(1.0, device=device))
    sidelobe_loss = torch.linalg.norm(wavelet_energy * sidelobe_weights)
    loss = scale['match'] * match_loss + scale['tv_x'] * tv_nx + scale['tv_z'] * tv_nz + scale['tv_t'] * tv_nt + scale['sidelobe_loss'] * sidelobe_loss
    loss_dic = {'total_loss': loss.item(), 'match': match_loss.item(), 'tv_x': tv_nx.item(), 'tv_z': tv_nz.item(), 'tv_t': tv_nt.item(), 'sidelobe_loss': sidelobe_loss.item()}
    return (loss, loss_dic, seis_pred_ori_return)


def wavelet_fitting(f0, dt, dur, amp, image, cond_image, well_image, num_iterations, seis_min, seis_max, imp_min, imp_max, int_size=(4, 4), mask=None, mode='ricker', method='opt', optimizer='Adam', lr=0.1, momentum=0.9, window=(50, 50), scale={'seismic': {'match': 50, 'tv_x': 1, 'tv_z': 1, 'tv_t': 1}, 'well': {'match': 50, 'tv_x': 1, 'tv_z': 1, 'tv_t': 1}}, *, amplitude_scale=10.0, **kwargs):
    device = image.device
    if mode != 'ricker' or method != 'opt' or optimizer != 'Adam':
        raise ValueError("This release supports only the original Ricker wavelet estimation optimized with Adam.")
    scale = copy.deepcopy(scale)
    n_patch = (image.shape[-2] // window[0] + 1 if image.shape[-2] % window[0] != 0 else image.shape[-2] // window[0], image.shape[-1] // window[1] + 1 if image.shape[-1] % window[1] != 0 else image.shape[-1] // window[1])
    cond_image = cond_image.clone()
    image = image.clone().detach()
    freq_min = 10 / 100
    init_freq = (f0 / 100).repeat(n_patch[0] * n_patch[1]).to(device)
    init_freq_log = torch.log(torch.exp(init_freq - freq_min) - 1.0)
    freq_log = nn.Parameter(init_freq_log)
    init_phase = torch.tensor([0.0]).repeat(n_patch[0] * n_patch[1]).to(device)
    phase_shift = nn.Parameter(init_phase)
    amplitude_min = 0
    init_amplitude = (amp / 1).repeat(n_patch[0] * n_patch[1]).to(device)
    if init_amplitude.max() - amplitude_min > 20:
        init_amp_log = torch.tensor(init_amplitude - amplitude_min, dtype=torch.float32)
    else:
        init_amp_log = torch.log(torch.exp(init_amplitude - amplitude_min) - 1.0)
    amplitude_log = nn.Parameter(init_amp_log)

    def ricker_torch_complex(f, dt, duration, phase_shift, amplitude, sym=True):
        t = _get_time_torch(duration, dt, 0, sym).to(f.device)
        n_samples = t.shape[-1]
        f = f.view(-1, 1)
        phase_shift = phase_shift.view(-1, 1)
        amplitude = amplitude.view(-1, 1)
        device = f.device
        dtype = f.dtype
        freqs = torch.fft.fftfreq(n_samples, d=dt[0]).to(device=device, dtype=dtype).unsqueeze(0)
        pi_ft = torch.pi * f * t
        w_0 = (1 - 2 * pi_ft ** 2) * torch.exp(-pi_ft ** 2)
        W_f = torch.fft.fft(w_0)
        W_f[:, torch.where(freqs > 0)[1]] *= torch.exp(1j * phase_shift)
        W_f[:, torch.where(freqs < 0)[1]] *= torch.exp(-1j * phase_shift)
        W_f[:, torch.where(freqs == 0)[1]] *= torch.cos(phase_shift)
        return (torch.fft.ifft(W_f).real * amplitude, t)
    opti = torch.optim.Adam([freq_log, phase_shift, amplitude_log], lr=lr)
    scale_tvx = scale['well']['tv_x'] + scale['seismic']['tv_x']
    scale_tvz = scale['well']['tv_z'] + scale['seismic']['tv_z']
    scale['well']['tv_x'] = 0
    scale['seismic']['tv_x'] = 0
    scale['well']['tv_z'] = 0
    scale['seismic']['tv_z'] = 0
    norm_dics = defaultdict(list)
    norm_dics2 = defaultdict(list)
    window_new = (int_size[0], int_size[1])
    for i in tqdm(range(num_iterations)):
        freq = (freq_min + F.softplus(freq_log)) * 100
        amp = (amplitude_min + F.softplus(amplitude_log)) * amplitude_scale
        phase = phase_shift * torch.pi
        if cond_image.shape[-2] % int_size[0] != 0:
            tar_axis_x = (torch.arange(int_size[0] // 2, cond_image.shape[-2] + int_size[0] // 2, int_size[0]) - window[0] // 2) / window[0]
        else:
            tar_axis_x = (torch.arange(int_size[0] // 2, cond_image.shape[-2], int_size[0]) - window[0] // 2) / window[0]
        if cond_image.shape[-1] % int_size[1] != 0:
            tar_axis_z = (torch.arange(int_size[1] // 2, cond_image.shape[-1] + int_size[1] // 2, int_size[1]) - window[1] // 2) / window[1]
        else:
            tar_axis_z = (torch.arange(int_size[1] // 2, cond_image.shape[-1], int_size[1]) - window[1] // 2) / window[1]
        amp_input = amp.view(n_patch[0], n_patch[1])
        freq_input = freq.view(n_patch[0], n_patch[1])
        phase_input = phase.view(n_patch[0], n_patch[1])
        freq_interp = interpolate_2d_by_positions(freq_input, tar_axis_x, tar_axis_z, dim=1, mode='bilinear', align_corners=True)
        amp_interp = interpolate_2d_by_positions(amp_input, tar_axis_x, tar_axis_z, dim=1, mode='bicubic', align_corners=True)
        phase_interp = interpolate_2d_by_positions(phase_input, tar_axis_x, tar_axis_z, dim=1, mode='bicubic', align_corners=True)
        (wavelet_mat, _) = ricker_torch_complex(freq_interp.view(-1), torch.tensor([dt]).repeat(freq_interp.shape[0] * freq_interp.shape[1]), torch.tensor([5 / (freq_min * 100)]).repeat(freq_interp.shape[0] * freq_interp.shape[1]), phase_interp.view(-1), amp_interp.view(-1), sym=True)
        wavelet_mat = wavelet_mat.view(tar_axis_x.shape[0], tar_axis_z.shape[0], -1)
        tv_nx1 = torch.linalg.norm(torch.diff(freq_log.reshape(n_patch[0], n_patch[1]), dim=0, n=1))
        tv_nx2 = torch.linalg.norm(torch.diff(amplitude_log.reshape(n_patch[0], n_patch[1]), dim=0, n=1))
        tv_nx3 = torch.linalg.norm(torch.diff(phase.reshape(n_patch[0], n_patch[1]), dim=0, n=1))
        tv_nz1 = torch.linalg.norm(torch.diff(freq_log.reshape(n_patch[0], n_patch[1]), dim=1, n=1))
        tv_nz2 = torch.linalg.norm(torch.diff(amplitude_log.reshape(n_patch[0], n_patch[1]), dim=1, n=1))
        tv_nz3 = torch.linalg.norm(torch.diff(phase.reshape(n_patch[0], n_patch[1]), dim=1, n=1))
        reg = scale_tvx * (tv_nx1 + tv_nx2 + tv_nx3) + scale_tvz * (tv_nz1 + tv_nz2 + tv_nz3)
        opti.zero_grad()
        (norm1, norm_dic1, seis_pred_ori_return) = wavelet_fn(wavelet_mat, image, cond_image, window_new, seis_min, seis_max, imp_min, imp_max, scale=scale['seismic'], mask=None)
        (norm2, norm_dic2, seis_pred_ori_return) = wavelet_fn(wavelet_mat, well_image, cond_image, window_new, seis_min, seis_max, imp_min, imp_max, scale=scale['well'], mask=mask)
        norm_dic = {'seismic': norm_dic1, 'well': norm_dic2}
        norm = norm1 + norm2 + reg
        norm.backward()
        opti.step()
        for (key, value) in norm_dic2.items():
            if key == 'tv_x' and mode == 'ricker':
                norm_dics[key].append((tv_nx1 + tv_nx2 + tv_nx3).item())
            elif key == 'tv_z' and mode == 'ricker':
                norm_dics[key].append((tv_nz1 + tv_nz2 + tv_nz3).item())
            else:
                norm_dics[key].append(value)
        for (key, value) in norm_dic1.items():
            if key == 'tv_x' and mode == 'ricker':
                norm_dics2[key].append((tv_nx1 + tv_nx2 + tv_nx3).item())
            elif key == 'tv_z' and mode == 'ricker':
                norm_dics2[key].append((tv_nz1 + tv_nz2 + tv_nz3).item())
            else:
                norm_dics2[key].append(value)
    return (wavelet_mat.detach(), norm_dics, norm_dics2, seis_pred_ori_return)



def make_background_filter(image, kernel_size, torch_kernel=True, implementation="conv2d"):
    """Preserve the test-set and Marmousi kernel construction used in the verified runs."""
    if torch_kernel:
        axis = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2 * (kernel_size / 7) ** 2))
        kernel = kernel / kernel.sum()
    else:
        kernel = torch.from_numpy(gaussian_kernel(kernel_size, kernel_size / 7)).float()
    data = dict(kernel=kernel.to(image.device)[None, None], shape=tuple(image.shape[-2:]))
    return prepare_background(data, implementation)


def initial_fn(image, initial_image, background_filter):
    return torch.linalg.norm(background_filter(image) - initial_image) ** 2


def forward_wavelet_fn(wavelet_mat, image, cond_image, window, seis_min, seis_max,
                       imp_min, imp_max):
    predicted = wavelet_forward(wavelet_mat, image, window, seis_min, seis_max, imp_min, imp_max)
    return torch.linalg.norm(predicted - cond_image) ** 2
