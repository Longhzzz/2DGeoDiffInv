"""Article section-display settings; they do not affect inversion, normalization, or metric calculations."""
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator


_REGULARIZATION = ("testset_regularization", "marmousi_regularization")
_DATASETS = ("testset", "marmousi") + _REGULARIZATION


def section_figure_size(shape, rows=1, cols=2, panel_width=6.0, dataset=None):
    """shape is (lateral, time); preserve pixel proportions and use the article 2.5:1 display box for Marmousi."""
    if len(shape) != 2:
        raise ValueError("shape must be (lateral_samples, time_samples).")
    lateral, time = (int(value) for value in shape)
    rows, cols = int(rows), int(cols)
    if min(lateral, time, rows, cols) < 1 or panel_width <= 0:
        raise ValueError("Section dimensions, row/column counts, and panel width must be positive.")
    ratio = 0.4 if dataset == "marmousi_regularization" else time / lateral
    title_and_axes = 0.9 if dataset in _REGULARIZATION else 0.65
    return (cols * (float(panel_width) + 0.55) + (cols - 1) * 0.25,
            rows * (float(panel_width) * ratio + title_and_axes) + (rows - 1) * 0.35)


def _well_array(value, shape, name):
    """Return a well-display copy in (lateral, time) order, allowing leading singleton batch/channel dimensions."""
    array = np.ma.asarray(value).squeeze()
    if array.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {array.shape}.')
    return array


def paper_section(ax, shown, dataset, wells=(), well_width=3,
                  well_values=None, well_mask=None, impedance=True,
                  show_wells=True, dt=0.0005):
    """Style an existing AxesImage without changing its data or pixel extent."""
    if dataset not in _DATASETS:
        raise ValueError(f'dataset must be one of {_DATASETS}.')
    array = np.ma.asarray(shown.get_array())
    if array.ndim != 2:
        raise ValueError("The section must be a 2D image.")
    time, lateral = array.shape
    original_xlim, original_ylim = ax.get_xlim(), ax.get_ylim()
    shown.set_interpolation("none")
    # Set a common display norm so colored well overlays and the colorbar use the same range.
    if impedance and shown.get_cmap().name.lower() == "terrain":
        shown.set_clim(3000, 11000)
    if dataset == "marmousi_regularization":
        ax.set_aspect("auto")
        ax.set_box_aspect(0.4)
        target_ylim = (time - 0.5, -0.5)
    else:
        ax.set_box_aspect(None)
        ax.set_aspect("equal", adjustable="box")
        target_ylim = (time, 0) if dataset == "testset_regularization" else original_ylim
    if dataset in _REGULARIZATION:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt used for the displayed time axis must be finite and positive.")
        ax.set_axis_on()
        ax.set_xlabel("Trace no.")
        ax.set_ylabel("Time (s)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value * dt:g}"))
    else:
        ax.set_axis_off()
    ax.set_ylim(target_ylim)

    # These attributes control display only and are not written to inversion parameters or result files.
    shown._paper_dataset = dataset
    shown._paper_impedance = bool(impedance)
    for artist in getattr(shown, "_paper_well_artists", ()):
        artist.remove()
    artists = []
    shown._paper_well_artists = artists
    if not impedance or not show_wells:
        return
    columns = [int(value) for value in wells]
    if any(value < 0 or value >= lateral for value in columns):
        raise ValueError("Conditioning well locations exceed the lateral extent of the section.")
    if dataset not in _REGULARIZATION:
        # Framework comparisons use white bars positioned and sized from the actual conditioning-well metadata.
        width = int(well_width)
        if width < 1:
            raise ValueError("well_width must be positive.")
        for column in columns:
            artists.append(ax.axvspan(column - 0.5,
                                      min(column + width, lateral) - 0.5,
                                      color="white", linewidth=0, zorder=4))
    elif columns and well_values is not None and well_mask is not None:
        values = _well_array(well_values, (lateral, time), "well_values")
        mask = _well_array(well_mask, (lateral, time), "well_mask")
        lo, hi = shown.get_clim()
        if hi <= lo:
            raise ValueError("The upper impedance display limit must exceed the lower limit.")
        for column in columns:
            trace = np.ma.asarray(values[column])
            coverage = np.asarray(mask[column].filled(0)) > 0
            coverage &= ~np.ma.getmaskarray(trace)
            coverage &= np.isfinite(np.asarray(trace.filled(np.nan)))
            # Connect neighboring points only when both belong to valid well segments; do not bridge missing intervals.
            indices = np.flatnonzero(coverage[:-1] & coverage[1:])
            if not len(indices):
                continue
            segments = np.empty((len(indices), 2, 2), dtype=float)
            segments[:, :, 0] = column
            segments[:, 0, 1] = indices
            segments[:, 1, 1] = indices + 1
            border_alpha = 0.5 if dataset == "marmousi_regularization" else 1.0
            color_width = 7 if dataset == "marmousi_regularization" else 5
            color_order = 8 if dataset == "marmousi_regularization" else 5
            border = LineCollection(segments, color="white", linewidth=9,
                                    alpha=border_alpha, zorder=4)
            colored = LineCollection(segments, cmap=shown.get_cmap(),
                                     norm=Normalize(0, 1), linewidth=color_width,
                                     zorder=color_order)
            colored.set_array((np.asarray(trace.data)[indices] - lo) / (hi - lo))
            ax.add_collection(border, autolim=False)
            ax.add_collection(colored, autolim=False)
            artists.extend([border, colored])
    # Well markers must not expand the axis limits automatically.
    ax.set_xlim(original_xlim)
    ax.set_ylim(target_ylim)


def paper_colorbar(fig, shown, ax):
    """Match the colorbar height to the section, retaining article ticks without adding physical-unit labels."""
    # Position the colorbar relative to the actual image axes to avoid height differences from box_aspect and axes-divider.
    colorbar_axis = ax.inset_axes([1.025, 0, 0.035, 1], transform=ax.transAxes)
    name = shown.get_cmap().name.lower()
    if name == "terrain":
        shown.set_clim(3000, 11000)
        ticks = ([3000, 6000, 8000, 10000, 11000]
                 if getattr(shown, "_paper_dataset", None) == "marmousi_regularization"
                 else [3000, 5000, 7000, 9000, 11000])
    elif name == "rainbow_plus" and np.allclose(
            shown.get_clim(), (1716.4, 4165.2), rtol=0, atol=0.01):
        # Use the explicit final-article ticks for profile 00983; retain the individual range for other profiles.
        ticks = [2000, 2500, 3000, 3500, 4000]
    else:
        ticks = None
    colorbar = fig.colorbar(shown, cax=colorbar_axis, ticks=ticks)
    if ticks is None:
        colorbar.locator = MaxNLocator(nbins=5)
        colorbar.update_ticks()
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(labelsize=9, length=3, width=0.5)
    return colorbar
