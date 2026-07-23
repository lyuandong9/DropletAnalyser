"""Build YOLOv8-seg training dataset from ROI annotations."""
import os, shutil, random, yaml
from pathlib import Path
from tqdm import tqdm

# Paths
SRC_IMAGES = Path(r'C:\LYD\imageidentification\output\cropped\H0\50cst-H0110rpm0.00175kw')
SRC_LABELS = Path(r'C:\LYD\imageidentification\output\yolo_dataset\labels')
DST = Path(r'C:\LYD\imageidentification\output\yolo_dataset')
TRAIN_RATIO = 0.85
random.seed(42)

# Create directories
for split in ['train', 'val']:
    (DST / 'images' / split).mkdir(parents=True, exist_ok=True)
    (DST / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Get all annotated frames
txt_files = sorted(SRC_LABELS.glob('*.txt'))
frame_ids = [t.stem for t in txt_files]
print(f'Annotated frames: {len(frame_ids)}')

# Shuffle and split
random.shuffle(frame_ids)
split_idx = int(len(frame_ids) * TRAIN_RATIO)
train_frames = sorted(frame_ids[:split_idx])
val_frames = sorted(frame_ids[split_idx:])
print(f'Train: {len(train_frames)}, Val: {len(val_frames)}')

# Count total instances
total_train = 0
total_val = 0

# Copy train data
for fid in tqdm(train_frames, desc='Train'):
    # Copy image (convert BMP to JPG to save space)
    bmp = SRC_IMAGES / f'{fid}.bmp'
    jpg = DST / 'images' / 'train' / f'{fid}.jpg'
    if bmp.exists():
        import cv2
        img = cv2.imread(str(bmp))
        if img is not None:
            cv2.imwrite(str(jpg), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        elif not jpg.exists():
            shutil.copy2(str(bmp), str(jpg))
    # Copy label
    label_src = SRC_LABELS / f'{fid}.txt'
    label_dst = DST / 'labels' / 'train' / f'{fid}.txt'
    shutil.copy2(str(label_src), str(label_dst))
    with open(label_src) as f:
        total_train += len(f.readlines())

# Copy val data
for fid in tqdm(val_frames, desc='Val'):
    bmp = SRC_IMAGES / f'{fid}.bmp'
    jpg = DST / 'images' / 'val' / f'{fid}.jpg'
    if bmp.exists():
        import cv2
        img = cv2.imread(str(bmp))
        if img is not None:
            cv2.imwrite(str(jpg), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        elif not jpg.exists():
            shutil.copy2(str(bmp), str(jpg))
    label_src = SRC_LABELS / f'{fid}.txt'
    label_dst = DST / 'labels' / 'val' / f'{fid}.txt'
    shutil.copy2(str(label_src), str(label_dst))
    with open(label_src) as f:
        total_val += len(f.readlines())

print(f'Train instances: {total_train}, Val instances: {total_val}')

# Generate dataset.yaml
yaml_path = DST / 'dataset.yaml'
config = {
    'path': str(DST.resolve()),
    'train': 'images/train',
    'val': 'images/val',
    'names': {0: 'droplet'},
    'nc': 1,
}
with open(yaml_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f'\nDataset ready: {DST}')
print(f'  images/train/ ({len(train_frames)} JPEG)')
print(f'  images/val/   ({len(val_frames)} JPEG)')
print(f'  labels/train/ ({len(train_frames)} TXT)')
print(f'  labels/val/   ({len(val_frames)} TXT)')
print(f'  dataset.yaml')
print(f'\nTraining command:')
print(f'  python train_launch.py --data {yaml_path} --model yolov8l-seg --epochs 200 --imgsz 1280 --batch 4')
