import os
import zipfile
import tempfile
import shutil

def extract_zip(zip_file):
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    return tmp_dir

def upload_files(uploaded_files, global_tmp_dir):
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
    event_paths = [(short, os.path.join(global_tmp_dir, short)) for short in event_files]
    return global_tmp_dir, event_paths

def find_event_files(folder):
    event_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if "tfevents" in file:
                relative_path = os.path.relpath(os.path.join(root, file), folder)
                event_files.append(relative_path)
    return event_files

# 压缩打包下载所有图片
def pack_images(image_list):
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
        for img in image_list:
            zf.write(img, arcname=os.path.basename(img))
    return tmp_zip.name