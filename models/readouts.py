"""Readout heads: map molecule embeddings [batch, in_dim] to predictions [batch, n_targets].

To add a head, write a class taking (in_dim, n_targets, **options), register it in
READOUTS and add conf/readout/<name>.yaml. Use it with e.g. readout=<name>."""
from torch import nn
import torch


class LinearReadout(nn.Module):
    def __init__(self, in_dim, n_targets):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_targets)

    def forward(self, x):
        return self.linear(x)


class MLPReadout(nn.Module):
    """num_layers Linear layers (including the output layer) with ReLU and dropout."""

    def __init__(self, in_dim, n_targets, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers, nn.Linear(dims[-1], n_targets))

    def forward(self, x):
        return self.mlp(x)

class PerTargetReadout(nn.Module):
    """ 
    Shared trunk but independent output head for specification
    """
    def __init__(self, in_dim, n_targets, hidden_dim=512, trunk_layers=2,dropout=0.1):
        super().__init__()
        if trunk_layers < 1:
            raise ValueError("No trunks ma guy")
        dims = [in_dim] + [hidden_dim] * trunk_layers
        layers = []
        for a,b in zip(dims[:-1],dims[1:]):
            layers += [nn.Linear(a,b),nn.GELU(),nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_targets)])

    def forward(self, x):
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=-1)

class _ResidualBlock(nn.Module):
    """ Seutup for ResidualMLPReadout """
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        h = nn.functional.gelu(self.fc1(self.norm(x)))
        return x+ self.fc2(self.dropout(h))


class ResidualMLPReadout(nn.Module):
    """Projection, then num_blocks pre-norm residual blocks, then the output layer."""
 
    def __init__(self, in_dim, n_targets, hidden_dim=512, num_blocks=2, dropout=0.1):
        super().__init__()
        if num_blocks < 1:
            raise ValueError('num_blocks must be at least 1')
        self.project = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, n_targets)
 
    def forward(self, x):
        return self.out(self.norm(self.blocks(self.project(x))))


class BottleneckReadout(nn.Module):
    """Make small and then big and then profit"""
 
    def __init__(self, in_dim, n_targets, bottleneck_dim=64, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets))
 
    def forward(self, x):
        return self.mlp(x)


READOUTS = {'linear': LinearReadout, 'mlp': MLPReadout, 'per_target': PerTargetReadout,
            'resmlp': ResidualMLPReadout, 'bottleneck': BottleneckReadout}
 

def build_readout(cfg, in_dim, n_targets):
    options = {k: v for k, v in cfg.items() if k != 'type'}
    return READOUTS[cfg.type](in_dim, n_targets, **options)
