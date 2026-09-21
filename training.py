import os
import shutil
import random
from ultralytics import YOLO

SRC = "dataset"
ANIMALS = ["bom", "broutch", "titus", "toundra", "yuki"]
SPLIT_RATIO = 0.8 

def copy_safe(src_path, dest_folder, filename):
    dest_path = f"{dest_folder}/{filename}"
    if os.path.exists(dest_path):
        name, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(f"{dest_folder}/{name}_{i}{ext}"):
            i += 1
        dest_path = f"{dest_folder}/{name}_{i}{ext}"
    shutil.copy(src_path, dest_path)

def split_dataset():
    for animal in ANIMALS:
        src_folder = f"{SRC}/{animal}"
        if not os.path.exists(src_folder):
            continue

        files = os.listdir(src_folder)
        random.shuffle(files)
        split_idx = int(len(files) * SPLIT_RATIO)
        train_files, val_files = files[:split_idx], files[split_idx:]

        for subset, subset_files in [("train", train_files), ("val", val_files)]:
            dest = f"{SRC}/{subset}/{animal}"
            os.makedirs(dest, exist_ok=True)
            for f in subset_files:
                copy_safe(f"{src_folder}/{f}", dest, f)


if __name__ == '__main__':
    split_dataset()
    model = YOLO("yolov8n-cls.pt")
    model.train(data=SRC, epochs=50, imgsz=224, device=0)