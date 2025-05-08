import os
import zipfile
import tempfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gradio as gr
from matplotlib.font_manager import FontProperties, fontManager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 全局变量
global_tmp_dir = None
global_event_files = []
uploaded_fonts = {}

# 常用字体列表
default_fonts = [
    "Arial", "Times New Roman", "Courier New", "Georgia",
    "SimHei", "SimSun", "Microsoft YaHei", "Microsoft JhengHei",
    "FangSong", "KaiTi", "DejaVu Sans"
]

def extract_zip(zip_file):
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    return tmp_dir

def upload_files(uploaded_files):
    global global_tmp_dir, global_event_files
    if global_tmp_dir:
        shutil.rmtree(global_tmp_dir)
    global_tmp_dir = tempfile.mkdtemp()

    for file in uploaded_files:
        if file.name.endswith('.zip'):
            with zipfile.ZipFile(file.name, 'r') as zip_ref:
                zip_ref.extractall(global_tmp_dir)
        else:
            shutil.copy(file.name, global_tmp_dir)

    event_files = find_event_files(global_tmp_dir)
    global_event_files = [(short, os.path.join(global_tmp_dir, short)) for short in event_files]
    return gr.update(choices=[f"./{short}" for short, _ in global_event_files])

def find_event_files(folder):
    event_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if "tfevents" in file:
                relative_path = os.path.relpath(os.path.join(root, file), folder)
                event_files.append(relative_path)
    return event_files

def load_scalars(log_file_path):
    ea = EventAccumulator(log_file_path)
    ea.Reload()
    return ea.Tags().get('scalars', [])

def get_all_scalars(selected_event_paths):
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
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - weight + 1)
        smoothed.append(np.mean(values[start:i+1]))
    return smoothed

def plot_selected_scalars(selected_paths, selected_scalars, title_map, xlabel, ylabel, dpi, smoothing, color_settings, show_grid, font_family, font_size):
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
            values = [e.value for e in events]
            values = smooth(values, smoothing)

            color = color_settings.get(sel) or next(color_cycle)
            label = title_map.get(sel, sel)
            ax.plot(steps, values, label=label, color=color)

        ax.set_title(scalar_name, fontproperties=font_props)
        ax.set_xlabel(xlabel if xlabel else "Step", fontproperties=font_props)
        ax.set_ylabel(ylabel if ylabel else "Value", fontproperties=font_props)
        ax.tick_params(axis='both', labelsize=font_size)

        # ✅ 设置坐标轴刻度的字体属性（包括 10/20 等数字）
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

def pack_images(image_list):
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
        for img in image_list:
            zf.write(img, arcname=os.path.basename(img))
    return tmp_zip.name

def upload_font_file(font_file):
    if font_file is None:
        return gr.update(choices=default_fonts)
    font_path = font_file.name
    font_prop = FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    fontManager.addfont(font_path)
    uploaded_fonts[font_name] = font_path
    fontManager._rebuild()
    return gr.update(choices=default_fonts + list(uploaded_fonts.keys()), value=font_name)

with gr.Blocks(title="TensorBoard绘图工具") as tensorboard_draw:
    gr.Markdown("# 🎨 TensorBoard 日志绘图工具 v3\n上传 `.zip` `.event` 文件，支持多文件，支持自定义颜色、字体，适合论文截图！")

    with gr.Row():
        files = gr.File(file_types=[".zip", ".event"], label="上传TensorBoard日志 (zip/event)", file_count="multiple")
        upload_btn = gr.Button("📦 解压或加载文件")

    event_selector = gr.CheckboxGroup(label="选择 event 文件", choices=[])
    scalar_selector = gr.CheckboxGroup(label="选择要绘制的 Scalars", choices=[])
    update_scalar_btn = gr.Button("📥 更新 Scalar 列表")

    custom_titles_input = gr.Textbox(label="曲线名字映射（JSON格式）", lines=6)
    color_picker_group = gr.Textbox(label="曲线颜色映射（JSON格式）", lines=6, placeholder='{"scalar_name (event_path)": "#FF0000"}')

    with gr.Row():
        xlabel_input = gr.Textbox(label="横坐标标题（默认 Step）")
        ylabel_input = gr.Textbox(label="纵坐标标题（默认 Value）")

    with gr.Row():
        font_selector = gr.Dropdown(label="选择字体", choices=default_fonts, value="Arial")
        font_size_selector = gr.Dropdown(label="字体大小", choices=[str(s) for s in [8,9,10,11,12,14,16,18,20,24,28,32,36]], value="12")
        font_upload = gr.File(file_types=[".ttf"], label="上传自定义字体文件（.ttf）")

    font_upload.change(upload_font_file, inputs=[font_upload], outputs=[font_selector])

    with gr.Row():
        smoothing_input = gr.Slider(1, 50, value=1, step=1, label="平滑窗口大小")
        dpi_input = gr.Slider(50, 600, value=100, step=10, label="输出图像 DPI")

    show_grid_checkbox = gr.Checkbox(label="显示网格线", value=True)

    plot_btn = gr.Button("🎨 绘制曲线图")
    pack_btn = gr.Button("🗜️ 打包所有图片成zip")

    output_gallery = gr.Gallery(label="绘制结果", show_label=True, columns=2, allow_preview=True, height="600px", object_fit="contain")
    zip_download = gr.File(label="下载打包zip", visible=False)

    upload_btn.click(upload_files, inputs=[files], outputs=[event_selector])

    def update_scalar_choices(selected_files):
        selected_paths = [f.lstrip("./") for f in selected_files]
        scalar_options = get_all_scalars(selected_paths)
        options = list(scalar_options.keys())
        default_titles = {opt: opt.split(' (')[0] for opt in options}
        return gr.update(choices=options), json.dumps(default_titles, indent=2)

    update_scalar_btn.click(update_scalar_choices, inputs=[event_selector], outputs=[scalar_selector, custom_titles_input])

    def start_plot(selected_files, selected_scalars, xlabel, ylabel, dpi, smoothing,
                   custom_titles_json, custom_colors_json, show_grid, font_family, font_size_str):
        selected_paths = [f.lstrip("./") for f in selected_files]
        try:
            title_map = json.loads(custom_titles_json) if custom_titles_json else {}
        except:
            title_map = {}
        try:
            color_map = json.loads(custom_colors_json) if custom_colors_json else {}
        except:
            color_map = {}
        font_size = int(font_size_str)
        return plot_selected_scalars(
            selected_paths, selected_scalars, title_map, xlabel, ylabel,
            dpi, smoothing, color_map, show_grid, font_family, font_size
        )

    plot_btn.click(start_plot, inputs=[
        event_selector, scalar_selector, xlabel_input, ylabel_input,
        dpi_input, smoothing_input, custom_titles_input, color_picker_group,
        show_grid_checkbox, font_selector, font_size_selector
    ], outputs=[output_gallery])

    pack_btn.click(pack_images, inputs=[output_gallery], outputs=[zip_download])
    pack_btn.click(lambda: gr.update(visible=True), outputs=[zip_download])

if __name__ == "__main__":
    tensorboard_draw.launch()
