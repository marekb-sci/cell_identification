# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


from data import NumpyCropsDataset, AUGMENTATIONS, get_augmentation_tv

n_images = 10
data_dir = r'D:\UW\projects\008_komorki_bialaczka\03_dane\01_Renishaw\04_cropped'
metadata_file =  r'D:\UW\projects\008_komorki_bialaczka\03_dane\01_Renishaw\04_cropped\info.csv'

ds_raw = NumpyCropsDataset(data_dir, metadata_file, background_generator=None)

aug = get_augmentation_tv(AUGMENTATIONS['flips']+AUGMENTATIONS['affine']+AUGMENTATIONS['crop32'])
ds_aug = NumpyCropsDataset(data_dir, metadata_file, img_shape=(1024, 32, 32), transform=aug)

for i_img in range(n_images):
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20,14))
    img_raw = ds_raw[i_img][0]
    c = axs[0,0].imshow(img_raw.mean(axis=0), vmin=800, vmax=1200)
    plt.colorbar(c, ax=axs[0,0])
    for i_row in range(3):
        for i_col in range(4):
            if i_row==0 and i_col ==0:
                continue
            img_raw = ds_aug[i_img][0]
            axs[i_row,i_col].imshow(img_raw.mean(axis=0), vmin=800, vmax=1200)
    fig.savefig(f'dataset_visualization/img_transformations_{i_img:02d}.png', dpi=300)
