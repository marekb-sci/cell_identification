# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from serialize import save_json

metadata_path = '../../03_dane/03_UW_MLset1/04_cropped/info.csv'
test_size = 32
output_path = '../../03_dane/03_UW_MLset1/04_cropped/data_split_1.json'

metadata = pd.read_csv(metadata_path).reindex()
largets_class = metadata['label'].value_counts().max()

train_indies = []
test_indices = []
for label, metadata_class in metadata.groupby('label'):
    train_i_label, test_i_label = train_test_split(metadata_class.index, test_size=test_size)

    train_indies += train_i_label.tolist()
    test_indices += test_i_label.tolist()

    #oversample
    train_indies += np.random.choice(train_i_label, largets_class-len(metadata_class)).tolist()


save_json({'train_indices': train_indies, 'test_indices': test_indices}, output_path)
