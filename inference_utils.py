"""Impedance palettes used by the inference notebooks.

Rainbow_Plus is from CIGVis 0.2.0, MIT License,
Copyright (c) 2023 Jintao Li; see licenses/CIGVis-MIT.txt.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

Rainbow_Plus = {
    'red': [(0.0, 1.0, 1.0), (0.135135, 0.0, 0.0), (0.382239, 0.0, 0.0),
            (0.46332, 0.0, 0.0), (0.53668, 1.0, 1.0), (0.606178, 1.0, 1.0),
            (0.868726, 1.0, 1.0),
            (1.0, 0.36470588235294116, 0.36470588235294116)],
    'green': [(0.0, 0.0, 0.0), (0.135135, 0.0, 0.0), (0.382239, 1.0, 1.0),
              (0.46332, 1.0, 1.0), (0.53668, 1.0, 1.0),
              (0.606178, 0.6666666666666666, 0.6666666666666666),
              (0.868726, 0.0, 0.0), (1.0, 0.0, 0.0)],
    'blue': [(0.0, 1.0, 1.0), (0.135135, 1.0, 1.0), (0.382239, 1.0, 1.0),
             (0.46332, 0.0, 0.0), (0.53668, 0.0, 0.0), (0.606178, 0.0, 0.0),
             (0.868726, 0.0, 0.0), (1.0, 0.0, 0.0)]
}



def get_impedance_cmap(dataset):
    """Article palettes, without importing OpenGL/CIGVis."""
    if dataset == "marmousi":
        return plt.get_cmap("terrain")
    return LinearSegmentedColormap("Rainbow_Plus", Rainbow_Plus, N=256)

