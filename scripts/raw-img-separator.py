import os
import shutil

def separate_raw_images(raw_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    jpeg_dir = os.path.join(output_dir, 'jpeg')
    heic_dir = os.path.join(output_dir, 'heic')
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(heic_dir, exist_ok=True)

    for filename in os.listdir(raw_dir):
        filepath = os.path.join(raw_dir, filename)
        if os.path.isfile(filepath):
            if filename.lower().endswith('.jpeg') or filename.lower().endswith('.jpg'):
                shutil.copy(filepath, os.path.join(jpeg_dir, filename))
            elif filename.lower().endswith('.heic'):
                shutil.copy(filepath, os.path.join(heic_dir, filename))

separate_raw_images("data/raw", "data/output")