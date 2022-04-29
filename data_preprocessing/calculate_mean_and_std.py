# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import json

dir_cropped = Path('../../03_dane/02_SRS-UJ_Jurkat+THP_2chans_npys/03_cropped/images')
file_out = Path('../../03_dane/02_SRS-UJ_Jurkat+THP_2chans_npys/03_cropped/normalization.json')


N = 0
mean = 0

for fn in dir_cropped.glob('*.npy'):
    data = np.load(fn)
    cell_data = data[(data != 0).any(axis=2)]
    w = len(cell_data)
    cell_mean = np.mean(cell_data, axis=0)

    mean = N/(N+w)*mean + w/(N+w)*cell_mean
    N += w


mean_square_dev = 0
N = 0
for fn in dir_cropped.glob('*.npy'):
    data = np.load(fn)
    cell_data = data[(data != 0).any(axis=2)]
    w = len(cell_data)
    cell_mean_square_dev = np.mean((cell_data - mean)**2, axis=0)

    mean_square_dev = N/(N+w)*mean_square_dev + w/(N+w)*cell_mean_square_dev
    N += w

std = np.sqrt(mean_square_dev)

file_out.parent.mkdir(exist_ok=True, parents=True)
with open(file_out, 'w') as f:
    json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)
