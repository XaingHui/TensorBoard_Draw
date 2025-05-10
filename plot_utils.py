import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_scalars(log_file_path):
    ea = EventAccumulator(log_file_path)
    ea.Reload()
    return ea.Tags().get('scalars', [])

def get_all_scalars(selected_event_paths, global_event_files):
    scalar_options = {}
    for file in selected_event_paths:
        full_file = dict(global_event_files)[file]
        scalars = load_scalars(full_file)
        for scalar in scalars:
            key = f"{scalar} ({file})"
            scalar_options[key] = (full_file, scalar)
    return scalar_options

def smooth(values, weight):
    if weight <= 1:
        return values
    return [np.mean(values[max(0, i - weight + 1):i+1]) for i in range(len(values))]

def plot_selected_scalars(selected_paths, selected_scalars, title_map, xlabel, ylabel, dpi, smoothing, color_settings, show_grid, font_family, font_size, global_event_files):
    scalar_map = get_all_scalars(selected_paths, global_event_files)
    grouped_scalars = {}
    for selected in selected_scalars:
        original_scalar_name = selected.split(' (')[0]
        grouped_scalars.setdefault(original_scalar_name, []).append(selected)

    saved_files = []
    color_palette = sns.color_palette("tab10", n_colors=20)
    color_cycle = iter(color_palette)

    save_dir = tempfile.mkdtemp()
    font_props = FontProperties(family=font_family if font_family else None, size=font_size)

    for scalar_name, selections in grouped_scalars.items():
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

        for sel in selections:
            file_path, scalar_tag = scalar_map[sel]
            ea = EventAccumulator(file_path)
            ea.Reload()
            events = ea.Scalars(scalar_tag)
            steps = [e.step for e in events]
            values = smooth([e.value for e in events], smoothing)

            color = color_settings.get(sel) or next(color_cycle)
            label = title_map.get(sel, sel)
            ax.plot(steps, values, label=label, color=color)

        ax.set_title(scalar_name, fontproperties=font_props)
        ax.set_xlabel(xlabel or "Step", fontproperties=font_props)
        ax.set_ylabel(ylabel or "Value", fontproperties=font_props)
        ax.tick_params(axis='both', labelsize=font_size)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(font_props)

        if show_grid:
            ax.grid(True)
        ax.legend(prop=font_props)

        file_safe_name = scalar_name.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(save_dir, f"{file_safe_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        saved_files.append(save_path)
        plt.close(fig)

    return saved_files
