# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

#%% define paths

dir_full_images = Path('../../03_dane/02_SRS-UJ_Jurkat+THP_2chans_npys/01_npy_cube')
dir_masks = Path('../../03_dane/02_SRS-UJ_Jurkat+THP_2chans_npys/02_segmentation')
dir_cropped = Path('../../03_dane/02_SRS-UJ_Jurkat+THP_2chans_npys/03_cropped')
dir_cropped_visualization = Path('../../03_dane/02_SRS-UJ_Jurkat+THP_2chans_npys/04_cropped_visualized')

#%% define parameters

min_cell_size = 10
visualize_cells = True #visualization is slow (~0.75 cell/s)

#%% prepare output dirs

dir_cropped.mkdir(exist_ok=True, parents=True)
dir_cropped_images = dir_cropped / 'images'
dir_cropped_images.mkdir(exist_ok=True)

if visualize_cells:
    dir_cropped_visualization.mkdir(exist_ok=True, parents=True)

#%% define functions

def normalize_image_data(img_data):
    """process image data to 1 channel float in 0-1 range"""
    img_data = np.abs(img_data).mean(axis=2)
    img_data /= 10 #img_data.max() #np.quantile(img_data, 0.95)
    return np.clip(img_data, 0, 1)

def visualize(full, img_out, x, y):
    """left panel: full image with bbox, right panel: cropped cell"""

    img_norm = normalize_image_data(full)
    img_out_norm = normalize_image_data(img_out)

    bottom, top = min(y), max(y)
    left, right = min(x), max(x)
    full_h, full_w = img_norm.shape
    crop_h, crop_w = top - bottom, right-left

    fig, axs=plt.subplots(ncols=2, figsize=(12,6))

    yy, xx = np.meshgrid(np.arange(full.shape[1]), np.arange(full.shape[0]))
    axs[0].pcolormesh(yy, xx, img_norm, vmin=0, vmax=1)
    rect = patches.Rectangle((max(0, bottom-1), max(0, left-1)), min(full_h-bottom+1, crop_h+1), min(full_w-left+1, crop_w+1), linewidth=1, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)

    yy2, xx2 = np.meshgrid(np.arange(bottom, top+1), np.arange(left, right+1))
    im2 = axs[1].pcolormesh(yy2, xx2, img_out_norm, vmin=0, vmax=1)
    fig.colorbar(im2, ax=axs[1])

    return fig

def get_cell_class_from_fname(fname):
    return fname.split('_')[0]


#%% crop, save and visualize cells
info = []
for full_img in dir_full_images.glob('*.npy'):
    img_name = full_img.stem
    cell_type = get_cell_class_from_fname(img_name)
    full = np.load(full_img)

    masks = np.load(dir_masks / f'MASK_{img_name}.npy')

    for cell_number in np.unique(masks):
        if cell_number == 0: #skip background
            continue
        x, y = np.where(masks==cell_number)

        img_out = np.zeros((max(x)-min(x)+1, max(y)-min(y)+1, full.shape[2]), dtype=full.dtype)
        img_out[x-min(x), y-min(y)] = full[x,y]

        img_number = len(info)
        out_file_name = f'{img_name}_{img_number}.npy'
        np.save(dir_cropped_images / out_file_name, img_out)

        info_entry = {'file_name': out_file_name, 'parent_image': img_name, 'label': cell_type, 'img_w': img_out.shape[1], 'img_h': img_out.shape[0]}
        info.append(info_entry)

        if visualize_cells:
            fig = visualize(full, img_out, x, y)
            fig.savefig((dir_cropped_visualization / out_file_name).with_suffix('.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

#%% create metadata files
info_df = pd.DataFrame(info)
info_df.to_csv(dir_cropped / 'info_raw.csv', index=False)

info_df = info_df.loc[(info_df.img_h >= min_cell_size) & (info_df.img_w >= min_cell_size)]
info_df.to_csv(dir_cropped / 'info.csv', index=False)


