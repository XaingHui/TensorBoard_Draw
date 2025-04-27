import os
import zipfile
import tempfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import gradio as gr

# 全局变量
global_tmp_dir = None
global_event_files = []  # [(short_path, full_path)]

# 解压zip到临时目录
def extract_zip(zip_file):
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    return tmp_dir

# 找到所有event文件
def find_event_files(folder):
    event_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if "tfevents" in file:
                event_files.append(os.path.join(root, file))
    return event_files

# 加载一个event文件里的所有scalars
def load_scalars(log_file_path):
    ea = EventAccumulator(log_file_path)
    ea.Reload()
    scalar_tags = ea.Tags().get('scalars', [])
    return scalar_tags

# 收集所有scalar
def get_all_scalars(selected_event_paths):
    scalar_options = {}
    for file in selected_event_paths:
        scalars = load_scalars(file)
        for scalar in scalars:
            key = f"{scalar} ({os.path.basename(file)})"
            scalar_options[key] = (file, scalar)
    return scalar_options

# 简单平滑
def smooth(values, weight):
    if weight <= 1:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - weight + 1)
        smoothed.append(np.mean(values[start:i+1]))
    return smoothed

# 画图
def plot_selected_scalars(selected_paths, selected_scalars, title_map, xlabel, ylabel, dpi, smoothing, color_settings, show_grid):
    scalar_map = get_all_scalars(selected_paths)

    grouped_scalars = {}
    for selected in selected_scalars:
        original_scalar_name = selected.split(' (')[0]
        if original_scalar_name not in grouped_scalars:
            grouped_scalars[original_scalar_name] = []
        grouped_scalars[original_scalar_name].append(selected)

    saved_files = []

    color_palette = sns.color_palette("tab10", n_colors=20)
    color_cycle = iter(color_palette)

    for scalar_name, selections in grouped_scalars.items():
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

        for sel in selections:
            file_path, scalar_tag = scalar_map[sel]
            ea = EventAccumulator(file_path)
            ea.Reload()
            events = ea.Scalars(scalar_tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            values = smooth(values, smoothing)

            if sel in color_settings:
                color = color_settings[sel]
            else:
                color = next(color_cycle)

            label = title_map.get(sel, sel)
            ax.plot(steps, values, label=label, color=color)

        ax.set_title(scalar_name)
        ax.set_xlabel(xlabel if xlabel else "Step")
        ax.set_ylabel(ylabel if ylabel else "Value")
        if show_grid:
            ax.grid(True)
        ax.legend()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(tmp_file.name, bbox_inches='tight')
        saved_files.append(tmp_file.name)
        plt.close(fig)

    return saved_files

# 上传zip
def upload_zip(zip_file):
    global global_tmp_dir, global_event_files
    if global_tmp_dir:
        shutil.rmtree(global_tmp_dir)
    global_tmp_dir = extract_zip(zip_file)
    event_files = find_event_files(global_tmp_dir)
    global_event_files = [(os.path.relpath(f, global_tmp_dir), f) for f in event_files]
    short_paths = [short for short, full in global_event_files]
    return gr.update(choices=short_paths)

# 更新scalar列表
def update_scalar_choices(selected_files):
    selected_paths = []
    for short_path, full_path in global_event_files:
        if short_path in selected_files:
            selected_paths.append(full_path)
    scalar_options = get_all_scalars(selected_paths)
    options = list(scalar_options.keys())
    return gr.update(choices=options)

# 绘图按钮
def start_plot(selected_files, selected_scalars, xlabel, ylabel, dpi, smoothing, custom_titles_json, custom_colors_json, show_grid):
    selected_paths = []
    for short_path, full_path in global_event_files:
        if short_path in selected_files:
            selected_paths.append(full_path)

    try:
        title_map = json.loads(custom_titles_json) if custom_titles_json else {}
    except:
        title_map = {}

    try:
        color_map = {}
        if custom_colors_json:
            color_inputs = custom_colors_json.strip().split(';')
            for entry in color_inputs:
                if ':' in entry:
                    key, value = entry.split(':', 1)
                    color_map[key.strip()] = value.strip()
    except:
        color_map = {}

    return plot_selected_scalars(
        selected_paths, selected_scalars, title_map, xlabel, ylabel, dpi, smoothing, color_map, show_grid
    )

# 打包下载
def pack_images(image_list):
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
        for img in image_list:
            zf.write(img, arcname=os.path.basename(img))
    return tmp_zip.name

# Gradio界面
with gr.Blocks(title="TensorBoard绘图工具") as tensorboard_draw:
    gr.Markdown("# 🎨 TensorBoard 日志绘图工具 v2\n上传 `.zip` 文件，一键绘制多条曲线，适合论文投稿截图！")

    with gr.Row():
        zip_file = gr.File(file_types=[".zip"], label="上传 TensorBoard 日志 (zip压缩包)")
        upload_btn = gr.Button("📦 解压并扫描事件文件")

    event_selector = gr.CheckboxGroup(label="选择 event 文件", choices=[])
    scalar_selector = gr.CheckboxGroup(label="选择要绘制的 Scalars", choices=[])

    update_scalar_btn = gr.Button("📥 更新 Scalar 列表")

    with gr.Row():
        xlabel_input = gr.Textbox(label="横坐标标题（默认 Step）", placeholder="输入横坐标名")
        ylabel_input = gr.Textbox(label="纵坐标标题（默认 Value）", placeholder="输入纵坐标名")

    with gr.Row():
        smoothing_input = gr.Slider(1, 50, value=1, step=1, label="平滑窗口大小")
        dpi_input = gr.Slider(50, 600, value=100, step=10, label="输出图像 DPI")

    show_grid_checkbox = gr.Checkbox(label="显示网格线", value=True)

    custom_titles_input = gr.Textbox(label="自定义曲线名（JSON格式）", placeholder='例如 {"Acc (xxx.events)": "Accuracy", "F1 (xxx.events)": "F1 Score"}')
    custom_colors_input = gr.Textbox(label="自定义曲线颜色（格式 name:color;name2:color2）", placeholder="例如 Acc:red; F1:blue")

    plot_btn = gr.Button("🎨 绘制曲线图")
    pack_btn = gr.Button("🗜️ 打包所有图片成zip")

    output_gallery = gr.Gallery(label="绘制结果", show_label=True, columns=2, allow_preview=True, height="600px", object_fit="contain")
    zip_download = gr.File(label="下载打包zip", visible=False)

    upload_btn.click(upload_zip, inputs=[zip_file], outputs=[event_selector])
    update_scalar_btn.click(update_scalar_choices, inputs=[event_selector], outputs=[scalar_selector])

    plotted_images = plot_btn.click(
        start_plot,
        inputs=[event_selector, scalar_selector, xlabel_input, ylabel_input, dpi_input, smoothing_input, custom_titles_input, custom_colors_input, show_grid_checkbox],
        outputs=[output_gallery]
    )

    pack_btn.click(pack_images, inputs=[output_gallery], outputs=[zip_download])
    pack_btn.click(lambda: gr.update(visible=True), outputs=[zip_download])

tensorboard_draw.launch()
