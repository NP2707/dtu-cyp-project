from pathlib import Path
import numpy as np, pandas as pd, torch
from hydra import compose, initialize
from models import MODELS
from prepare_data import REG, CLS

with initialize(config_path='conf', version_base='1.3'):
    cfg = compose(config_name='config',
                  overrides=['model=atom_jepa', 'model.freeze_encoder=true'])

data = Path(cfg.data)
out = Path('cache'); out.mkdir(exist_ok=True)
model = MODELS[cfg.model.type](cfg.model, cfg.readout, 1, data, cfg.device)
model.eval()

for task, targets in [('regression', REG), ('classification', CLS)]:
    for split in ['train', 'val', 'test']:
        frame = pd.read_csv(data / task / f'{split}.csv')
        rows = []
        with torch.no_grad():
            for batch in model.loader(frame, targets, 16):
                rows.append(model.encoder(batch).float().cpu().numpy())
        np.save(out / f'{task}_{split}.npy', np.concatenate(rows))
        frame[['key'] + targets].to_csv(out / f'{task}_{split}_labels.csv', index=False)
