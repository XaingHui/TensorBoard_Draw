# 🎨 TensorBoard Logs 绘图工具 (Web版)

一个轻量、简洁、可扩展的 TensorBoard 日志绘图应用。支持多文件对比、平滑、分辨率调整、颜色自定义等功能！

## 🚀 功能特点
- 上传整个 TensorBoard logs 文件夹 (zip 格式)
- 自动扫描 `.tfevents` 文件
- 选择要绘制的 event 文件
- 多 scalar 支持，每个 scalar 独立绘图
- 支持自定义：
  - 图像标题
  - 横坐标 / 纵坐标 名称
  - 平滑窗口（Moving Average）
  - 分辨率 DPI
  - 曲线颜色 (JSON 格式)
- 一键保存高质量图片！

## 📦 安装依赖

```bash
pip install gradio matplotlib tensorboard
