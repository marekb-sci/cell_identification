# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
# from copy import deepcopy
import pytorch_lightning as pl

from serialize import load_json, save_json
from data import NumpyCropsDM #, AUGMENTATIONS
from training import ParticlesClassifier #default_config, prepare_config

CONFIGS_DIR = Path(sys.argv[1])

for config_file in CONFIGS_DIR.glob('*.json'):
    config = load_json(config_file)

    data_module = NumpyCropsDM(config['data'])
    model = ParticlesClassifier(config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=3, save_last=True, filename='{epoch}-{val_accuracy:.2f}-{val_loss:.4f}')
#    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20)
    logger_dir, logger_name = os.path.split(config['output_dir'])

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback], #early_stop_callback],
        # default_root_dir = Config.log_dir,
        logger = pl.loggers.TensorBoardLogger(logger_dir, name=logger_name, default_hp_metric=False),
        max_epochs = config['training']['num_epochs'],
        log_every_n_steps = 1,
        **config['training']['trainer_kwargs']
        )
    # trainer = Trainer(default_root_dir='/your/path/to/save/checkpoints')
    save_json(config, Path(trainer.log_dir) / 'config.json')

    trainer.fit(model, datamodule=data_module)
