import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.atom_jepa import AtomJEPAModel


class MultiConformers(Dataset):
    def __init__(self, frame, folder, fallback, targets, owner, limit=0):
        self.frame = frame.reset_index(drop=True)
        self.folder, self.fallback, self.owner, self.limit = folder, fallback, owner, limit
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)
        self.draws = np.zeros(len(self.frame), dtype=np.int64)  # audit: see draw_summary()

    def __len__(self):
        return len(self.frame)

    def _geometries(self, key):
        """[k, n_atoms, 3] for one molecule, falling back to the single-conformer folder."""
        path = self.folder / f'{key}.npz'
        if path.exists():
            with np.load(path) as handle:
                return handle['z'], handle['pos']
        # A molecule whose multi-conformer generation failed keeps its single geometry. Dropping it instead would change the splits, which must never happen.
        with np.load(self.fallback / f'{key}.npz') as handle:
            z, pos = handle['z'], handle['pos']
        return z, (pos[None] if pos.ndim == 2 else pos)

    def __getitem__(self, i):
        key = self.frame.iloc[i].key
        z, pos = self._geometries(key)
        if pos.ndim != 3 or pos.shape[2] != 3:
            raise ValueError(f'{key}: expected pos [k, n_atoms, 3], found {pos.shape}')
        available = len(pos) if not self.limit else min(len(pos), self.limit)
        if self.owner.training and self.owner.augment and available > 1:
            j = int(torch.randint(available, (1,)).item())
        else:
            j = min(self.owner.conformer_index, available - 1)
        self.draws[i] += 1
        return {'atomic_numbers': torch.tensor(np.asarray(z), dtype=torch.long),
                'node_coordinates': torch.tensor(np.asarray(pos[j]), dtype=torch.float32),
                'y': torch.tensor(self.y[i])}


class AtomJEPAMultiConfModel(AtomJEPAModel):
    """AtomJEPAModel reading conformers_multi, with a random geometry per training access."""

    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__(cfg, readout, n_targets, data_dir, device)
        self.folder = data_dir / cfg.conformer_folder
        self.fallback = data_dir / 'conformers'
        if not self.folder.is_dir():
            raise SystemExit(f'{self.folder} does not exist; run prepare_conformers_multi.py')
        if not self.fallback.is_dir():
            raise SystemExit(f'{self.fallback} does not exist; it is the fallback geometry source')
        self.augment = bool(cfg.augment)
        self.conformer_index = int(cfg.conformer_index)
        self.max_conformers = int(cfg.max_conformers)  
        print(f'[multiconf] folder={self.folder.name} augment={self.augment} '
              f'eval_conformer_index={self.conformer_index} '
              f'max_conformers={self.max_conformers or "all"} '
              f'freeze_encoder={cfg.freeze_encoder}')

    def loader(self, frame, targets, batch_size, shuffle=False):
        dataset = MultiConformers(frame, self.folder, self.fallback, targets, self,
                                  limit=self.max_conformers)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self.encoder.collate, num_workers=0)