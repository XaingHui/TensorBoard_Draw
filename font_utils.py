from matplotlib.font_manager import FontProperties, fontManager

default_fonts = [
    "Arial", "Times New Roman", "Courier New", "Georgia",
    "SimHei", "SimSun", "Microsoft YaHei", "Microsoft JhengHei",
    "FangSong", "KaiTi", "DejaVu Sans"
]

uploaded_fonts = {}

def upload_font_file(font_file):
    if font_file is None:
        return default_fonts
    font_path = font_file.name
    font_prop = FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    fontManager.addfont(font_path)
    uploaded_fonts[font_name] = font_path
    fontManager._rebuild()
    return default_fonts + list(uploaded_fonts.keys()), font_name
