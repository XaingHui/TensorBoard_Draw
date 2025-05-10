from gradio_ui import build_ui
from wandb import tensorboard

if __name__ == "__main__":
    tensorboard_draw = build_ui()
    tensorboard_draw.launch()
