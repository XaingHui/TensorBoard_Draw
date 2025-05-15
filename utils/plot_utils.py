import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# 读取指定文件的 scalar 列表
def load_scalars(log_file_path):
    ea = EventAccumulator(log_file_path)
    ea.Reload()
    return ea.Tags().get('scalars', [])


# 构造 scalar_map: {"scalar_name (file_name)": (event_file_path, scalar_name)}
def get_all_scalars(selected_event_paths, global_event_files):
    scalar_options = {}
    for file in selected_event_paths:
        full_file = dict(global_event_files)[file]
        scalars = load_scalars(full_file)
        for scalar in scalars:
            key = f"{scalar} ({file})"
            scalar_options[key] = (full_file, scalar)
    return scalar_options


# 平滑函数
def smooth(values, weight):
    if weight <= 1:
        return values
    return [np.mean(values[max(0, i - weight + 1):i + 1]) for i in range(len(values))]


# 绘图函数
def plot_selected_scalars(
    selected_paths,
    selected_scalars,
    title_map,
    xlabel,
    ylabel,
    dpi,
    smoothing,
    color_settings,
    show_grid,
    font_family,
    font_size,
    global_event_files,
):
    # 正确构造 scalar_map
    scalar_map = get_all_scalars(selected_paths, global_event_files)

    # 归一名称 mapping: display_name → ori_key
    reverse_title_map = {v: k for k, v in title_map.items()}

    saved_files = []
    color_palette = sns.color_palette("tab10", n_colors=20)
    color_cycle = iter(color_palette)
    save_dir = tempfile.mkdtemp()

    font_props = FontProperties(family=font_family if font_family else None, size=font_size)

    for display_scalar in selected_scalars:
        ori_key = reverse_title_map.get(display_scalar, display_scalar)

        if ori_key not in scalar_map:
            print(f"[\u8b66\u544a] Scalar '{display_scalar}' (原始: {ori_key}) 不在 scalar_map 中, 跳过")
            continue

        file_path, scalar_tag = scalar_map[ori_key]

        ea = EventAccumulator(file_path)
        ea.Reload()
        events = ea.Scalars(scalar_tag)
        if not events:
            print(f"[\u8b66\u544a] Scalar '{display_scalar}' 没有事件数据, 跳过")
            continue

        steps = [e.step for e in events]
        values = smooth([e.value for e in events], smoothing)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

        label = display_scalar
        color = color_settings.get(display_scalar, next(color_cycle))
        ax.plot(steps, values, label=label, color=color)

        ax.set_title(label, fontproperties=font_props)
        ax.set_xlabel(xlabel or "Step", fontproperties=font_props)
        ax.set_ylabel(ylabel or "Value", fontproperties=font_props)
        ax.tick_params(axis='both', labelsize=font_size)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(font_props)

        if show_grid:
            ax.grid(True)
        ax.legend(prop=font_props)

        def sanitize_filename(name):
            return re.sub(r'[\\/*?:"<>|()\s]', "_", name)

        safe_title = sanitize_filename(ori_key)
        save_path = os.path.join(save_dir, f"{safe_title}.png")

        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        saved_files.append(save_path)

    return saved_files
