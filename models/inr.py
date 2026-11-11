"""2D IntraSeismic components. See licenses/IntraSeismic-MIT.txt."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    A multilayer perceptron (MLP) neural network.

    Parameters
    ----------
    D : int, optional
        The depth of the network (number of layers), by default 8.
    W : int, optional
        The width of each layer (number of neurons), by default 256.
    output_ch : int, optional
        The number of output channels, by default 1.
    enc : {'exp', 'exp3d', 'hash', 'rff'}, optional
        The type of encoding to use for the input data, by default 'exp'.
    levels : int, optional
        The number of levels used in the encoding, by default 10.
    scale : int, optional
        The scale factor applied in the encoding, by default 10.
    """    

    def __init__(self,  D=8, W=256, output_ch=1, enc='exp',n_features_per_level=2,levels=10, scale=10):
        super(MLP, self).__init__()       
        self.D = D
        self.W = W
        self.levels = levels
        self.scale = scale
        self.enc = enc

        if enc=='exp':
            self.input_ch = self.levels*n_features_per_level

        elif enc=='exp3d':
            self.input_ch = self.levels*n_features_per_level

        elif enc=='hash' or enc=='rff':
            self.input_ch = self.levels*n_features_per_level 

        self.layers_linears = nn.ModuleList([nn.Linear(self.input_ch, W)]  + [nn.Linear(W, W) for i in range(D-1)])
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        y = x    
        for i, l in enumerate(self.layers_linears):
            y = self.layers_linears[i](y)
            y = F.relu(y)

        output = self.output_linear(y)
        return output

def create_coords(shape):
    """Preserve the original coordinate scaling and row-major ordering."""
    if len(shape) != 2:
        raise ValueError("This release supports 2D profiles only.")
    pixel_size = 1 / np.max(shape)
    lateral, time = np.meshgrid(np.arange(shape[0]) * pixel_size,
                               np.arange(shape[1]) * pixel_size, indexing="ij")
    return torch.stack([torch.tensor(lateral, dtype=torch.float32).ravel(),
                        torch.tensor(time, dtype=torch.float32).ravel()], dim=1)

def create_model(config, device):
    """Construct the original FP32 hash encoding followed by a PyTorch MLP."""
    if torch.device(device).type != "cuda":
        raise RuntimeError("The original HashGrid INR requires an NVIDIA CUDA GPU. "
                           "See requirements.txt and the INR installation instructions in README.md.")
    try:
        import tinycudann as tcnn
    except ImportError as error:
        raise ImportError("Install the tiny-cuda-nn HashGrid backend following "
                          "the optional INR instructions in README.md before running this notebook.") from error
    encoding = tcnn.Encoding(2, config["encoding"], dtype=torch.float32)
    mlp = MLP(D=config["mlp"]["Depth"], W=config["mlp"]["Width"],
              enc=config["mlp"]["enc"], levels=config["encoding"]["n_levels"],
              n_features_per_level=config["encoding"]["n_features_per_level"],
              output_ch=1)
    return nn.Sequential(encoding, mlp).to(device)
