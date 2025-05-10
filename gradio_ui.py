import json
import gradio as gr
from file_utils import upload_files, find_event_files, pack_images
from font_utils import upload_font_file, default_fonts
from plot_utils import plot_selected_scalars, get_all_scalars

global_tmp_dir = None
global_event_files = []

def build_ui():
    with gr.Blocks(title="TensorBoard绘图工具") as tensor_board:
        gr.Markdown("# 🎨 TensorBoard 日志绘图工具 v3")

        with gr.Row():
            files = gr.File(file_types=[".zip", ".event"], label="上传日志", file_count="multiple")
            upload_btn = gr.Button("📦 解压或加载文件")

        event_selector = gr.CheckboxGroup(label="选择 event 文件", choices=[])
        scalar_selector = gr.CheckboxGroup(label="选择 Scalars", choices=[])
        update_scalar_btn = gr.Button("📥 更新 Scalar 列表")

        custom_titles_input = gr.Textbox(label="曲线名字映射（JSON格式）", lines=6)
        color_picker_group = gr.Textbox(label="曲线颜色映射（JSON格式）", lines=6)

        with gr.Row():
            xlabel_input = gr.Textbox(label="横坐标标题")
            ylabel_input = gr.Textbox(label="纵坐标标题")

        with gr.Row():
            font_selector = gr.Dropdown(label="选择字体", choices=default_fonts, value="Arial")
            font_size_selector = gr.Dropdown(label="字体大小", choices=[str(s) for s in [8,10,12,14,16,18,20,24]], value="12")
            font_upload = gr.File(file_types=[".ttf"], label="上传字体")

        smoothing_input = gr.Slider(1, 50, value=1, step=1, label="平滑窗口大小")
        dpi_input = gr.Slider(50, 600, value=100, step=10, label="输出图像 DPI")
        show_grid_checkbox = gr.Checkbox(label="显示网格线", value=True)

        plot_btn = gr.Button("🎨 绘制曲线图")
        pack_btn = gr.Button("🗜️ 打包所有图片")

        output_gallery = gr.Gallery(label="绘图结果", columns=2, height="600px")
        zip_download = gr.File(label="下载zip", visible=False)

        def handle_upload(files_):
            global global_tmp_dir, global_event_files
            global_tmp_dir, global_event_files = upload_files(files_, global_tmp_dir)
            return gr.update(choices=[f"./{short}" for short, _ in global_event_files])

        upload_btn.click(handle_upload, inputs=[files], outputs=[event_selector])
        font_upload.change(lambda f: gr.update(choices=upload_font_file(f)[0], value=upload_font_file(f)[1]), inputs=[font_upload], outputs=[font_selector])

        def update_scalar_choices(selected_files):
            selected_paths = [f.lstrip("./") for f in selected_files]
            scalar_options = get_all_scalars(selected_paths, global_event_files)
            options = list(scalar_options.keys())
            default_titles = {opt: opt.split(' (')[0] for opt in options}
            return gr.update(choices=options), json.dumps(default_titles, indent=2)

        update_scalar_btn.click(update_scalar_choices, inputs=[event_selector], outputs=[scalar_selector, custom_titles_input])

        def start_plot(selected_files, selected_scalars, xlabel, ylabel, dpi, smoothing, title_json, color_json, show_grid, font_family, font_size_str):
            selected_paths = [f.lstrip("./") for f in selected_files]
            title_map = json.loads(title_json) if title_json else {}
            color_map = json.loads(color_json) if color_json else {}
            return plot_selected_scalars(selected_paths, selected_scalars, title_map, xlabel, ylabel, dpi, smoothing, color_map, show_grid, font_family, int(font_size_str), global_event_files)

        plot_btn.click(start_plot, inputs=[
            event_selector, scalar_selector, xlabel_input, ylabel_input,
            dpi_input, smoothing_input, custom_titles_input, color_picker_group,
            show_grid_checkbox, font_selector, font_size_selector
        ], outputs=[output_gallery])

        pack_btn.click(pack_images, inputs=[output_gallery], outputs=[zip_download])
        pack_btn.click(lambda: gr.update(visible=True), outputs=[zip_download])

    return tensor_board
