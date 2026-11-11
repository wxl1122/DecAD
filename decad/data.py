import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, InterpolationMode

CLASSES = {
    'mvtec': ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'),
    'visa': ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'),
}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def manifest(dataset, root, phase, classes=None):
    """Pair masks by basename, and load only normal samples in train."""
    root = Path(root).resolve()
    classes = tuple(classes or CLASSES[dataset])
    if not classes or not set(classes) <= set(CLASSES[dataset]):
        raise ValueError(f'Invalid {dataset} classes: {classes}')
    records = []
    csv_path = root / 'split_csv' / '1cls.csv'
    if dataset == 'visa' and csv_path.is_file():
        with csv_path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row['object'] not in classes or row['split'] != phase:
                    continue
                label = int(row['label'] != 'normal')
                if phase == 'train' and label:
                    raise ValueError('Official 1cls train split contains an anomaly')
                records.append(dict(category=row['object'], image=str(root / row['image']),
                                    mask=str(root / row['mask']) if label else None, label=label))
    else:
        if dataset == 'visa':
            for subpath in ('1cls', 'VisA_pytorch/1cls'):
                if (root / subpath / classes[0] / phase).is_dir():
                    root = root / subpath
                    break
        for category in classes:
            category_root = root / category
            split = category_root / phase
            if not split.is_dir():
                raise FileNotFoundError(split)
            for image in sorted(split.glob('*/*')):
                if image.suffix.lower() not in EXTENSIONS or not image.is_file():
                    continue
                label = int(image.parent.name not in ('good', 'normal'))
                if phase == 'train' and label:
                    raise ValueError(f'Anomalous training image: {image}')
                mask = None
                if label:
                    candidates = [category_root / 'ground_truth' / image.parent.name / f'{image.stem}{suffix}.png'
                                  for suffix in ('_mask', '')]
                    mask = next((str(p) for p in candidates if p.is_file()), None)
                    if mask is None:
                        raise FileNotFoundError(f'Missing mask for {image}')
                records.append(dict(category=category, image=str(image), mask=mask, label=label))
    records.sort(key=lambda r: (r['category'], r['image']))
    for category in classes:
        if not any(r['category'] == category for r in records):
            raise ValueError(f'Empty {phase} split for {category}')
    for record in records:
        for key in ('image', 'mask'):
            if record[key] and not Path(record[key]).is_file():
                raise FileNotFoundError(record[key])
    return records


def fingerprint(records):
    entries = [(r, Path(r['image']).stat().st_size, Path(r['image']).stat().st_mtime_ns)
               for r in records]
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


class Images(Dataset):
    def __init__(self, records, size=448, mask_size=256):
        self.records, self.size, self.mask_size = records, size, mask_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record['image']) as source:
            image = TF.resize(source.convert('RGB'), [self.size, self.size],
                              interpolation=InterpolationMode.BILINEAR)
            image = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if record['mask']:
            with Image.open(record['mask']) as source:
                mask = TF.resize(source.convert('L'), [self.mask_size, self.mask_size],
                                 interpolation=InterpolationMode.NEAREST)
                mask = torch.from_numpy(np.array(mask) > 0).unsqueeze(0)
        else:
            mask = torch.zeros(1, self.mask_size, self.mask_size, dtype=torch.bool)
        return image, mask, record['label'], index


class FeatureCache(Dataset):
    def __init__(self, path, records, expected, preload=False, shared=False):
        if preload and shared:
            raise ValueError('Choose private preloading or shared cache warming')
        self.path = Path(path)
        self.records = records
        with (self.path / 'metadata.json').open() as handle:
            self.metadata = json.load(handle)
        if self.metadata['identity'] != expected:
            raise ValueError(f'Incompatible feature cache: {path}')
        self.array = np.load(self.path / 'features.npy', mmap_mode='r')
        if list(self.array.shape) != self.metadata['shape'] or len(self.array) != len(records):
            raise ValueError('Incomplete feature cache')
        if preload:
            memory_info = Path('/proc/meminfo')
            if memory_info.exists():
                available = next(int(line.split()[1]) * 1024 for line in memory_info.read_text().splitlines()
                                 if line.startswith('MemAvailable:'))
                if available < self.array.nbytes + 8 * 1024**3:
                    raise MemoryError('Cache preloading requires its size plus 8 GiB available RAM')
            print(f'Preloading {self.array.nbytes / 1024**3:.1f} GiB of feature cache into RAM', flush=True)
            self.array = np.array(self.array)
            print('Feature cache preloaded', flush=True)
        elif shared:
            # Stream large reads into a fixed small buffer. Sequential page
            # faults on a cold mmap can otherwise produce tiny HDD requests.
            # Only the kernel file cache is shared; no private full-size copy.
            print(f'Warming {self.array.nbytes / 1024**3:.1f} GiB of shared feature cache', flush=True)
            buffer = bytearray(16 * 1024**2)
            completed = reported = 0
            with (self.path / 'features.npy').open('rb', buffering=0) as stream:
                if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_SEQUENTIAL'):
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
                while count := stream.readinto(buffer):
                    completed += count
                    if completed - reported >= 4 * 1024**3:
                        print(f'Shared feature cache read {completed / 1024**3:.1f} GiB', flush=True)
                        reported = completed
            print('Shared feature cache warmed; file remains read-only', flush=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.array[index]))
