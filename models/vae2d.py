import json
import os
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _parse_int_list(values: str | Sequence[int]) -> list[int]:
    if isinstance(values, str):
        return [int(item.strip()) for item in values.split(",") if item.strip()]
    return list(values)


def _norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_norm_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_norm_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return self.act(x + residual)


class ConvDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers_per_block: int):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(),
        )
        self.resnets = nn.Sequential(
            *[ConvResidualBlock(out_channels) for _ in range(layers_per_block)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return self.resnets(x)


class ConvUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers_per_block: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(),
        )
        self.resnets = nn.Sequential(
            *[ConvResidualBlock(out_channels) for _ in range(layers_per_block)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        return self.resnets(x)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        block_out_channels: Iterable[int],
        layers_per_block: int,
    ):
        super().__init__()
        channels = list(block_out_channels)
        blocks = []
        prev_channels = in_channels
        for out_channels in channels:
            blocks.append(ConvDownBlock(prev_channels, out_channels, layers_per_block=layers_per_block))
            prev_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SimpleConditionalDenoisingVAE(nn.Module):
    """Plain convolutional conditional denoising VAE baseline.

    The posterior encoder receives corrupted target plus condition.
    The decoder receives a standard Gaussian latent sample and condition features.
    """

    def __init__(
        self,
        sample_size: int = 512,
        in_channels: int = 1,
        condition_channels: int = 3,
        out_channels: int = 1,
        latent_dim: int = 512,
        block_out_channels: str | Sequence[int] = "128,128,256,256,512,512,512",
        layers_per_block: int = 3,
        base_channels: int | None = None,
        channel_multipliers: str | Sequence[int] | None = None,
        noise_condition_scale: float = 1.0,
    ):
        super().__init__()
        if noise_condition_scale <= 0:
            raise ValueError(f"noise_condition_scale must be positive, got {noise_condition_scale}")
        if base_channels is not None and channel_multipliers is not None:
            block_out_channels = [base_channels * mult for mult in _parse_int_list(channel_multipliers)]
        block_out_channels = _parse_int_list(block_out_channels)
        down_factor = 2 ** len(block_out_channels)
        if sample_size % down_factor != 0:
            raise ValueError(
                f"sample_size={sample_size} must be divisible by {down_factor} "
                f"for block_out_channels={block_out_channels}"
            )

        self.config = {
            "sample_size": sample_size,
            "in_channels": in_channels,
            "condition_channels": condition_channels,
            "out_channels": out_channels,
            "latent_dim": latent_dim,
            "block_out_channels": ",".join(str(item) for item in block_out_channels),
            "layers_per_block": layers_per_block,
            "base_channels": base_channels,
            "channel_multipliers": (
                ",".join(str(item) for item in _parse_int_list(channel_multipliers))
                if channel_multipliers is not None
                else None
            ),
            "noise_condition_scale": float(noise_condition_scale),
        }
        self.latent_dim = latent_dim
        self.noise_condition_scale = float(noise_condition_scale)
        self.down_size = sample_size // down_factor

        posterior_in_channels = in_channels + condition_channels
        self.posterior_encoder = ConvEncoder(
            posterior_in_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )
        self.condition_encoder = ConvEncoder(
            condition_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        hidden_channels = self.posterior_encoder.out_channels
        hidden_dim = hidden_channels * self.down_size * self.down_size
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)

        decoder_channels = block_out_channels[::-1]
        blocks = []
        prev_channels = hidden_channels + self.condition_encoder.out_channels
        for out_ch in decoder_channels[1:]:
            blocks.append(ConvUpBlock(prev_channels, out_ch, layers_per_block=layers_per_block))
            prev_channels = out_ch
        self.decoder_blocks = nn.Sequential(*blocks)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, noisy_image: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.posterior_encoder(torch.cat([noisy_image, condition], dim=1))
        features = features.flatten(start_dim=1)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        cond_features = self.condition_encoder(condition)
        z_features = self.fc_decode(z)
        z_features = z_features.view(
            z.shape[0],
            self.posterior_encoder.out_channels,
            self.down_size,
            self.down_size,
        )
        x = torch.cat([z_features, cond_features], dim=1)
        x = self.decoder_blocks(x)
        return self.final(x)

    def forward(self, noisy_image: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(noisy_image, condition)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        return {"sample": recon, "mu": mu, "logvar": logvar}

    def predict_clean(
        self,
        noisy_image: torch.Tensor,
        condition: torch.Tensor,
        use_posterior_mean: bool = True,
    ) -> torch.Tensor:
        """Denoise an observed image; posterior-mean mode is deterministic."""
        mu, logvar = self.encode(noisy_image, condition)
        z = mu if use_posterior_mean else self.reparameterize(mu, logvar)
        return self.decode(z, condition)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        z = torch.randn(
            condition.shape[0],
            self.latent_dim,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        return self.decode(z, condition)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory: str, map_location: str | torch.device = "cpu"):
        with open(os.path.join(load_directory, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(
            os.path.join(load_directory, "pytorch_model.bin"),
            map_location=map_location,
        )
        model.load_state_dict(state_dict)
        return model
