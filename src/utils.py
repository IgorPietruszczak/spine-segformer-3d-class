"""
utils.py — shared helpers
"""
from __future__ import annotations

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark    = True
    torch.backends.cudnn.deterministic = False  # benchmark=True & deterministic=True conflict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    return f"{n/1_000:.0f}K"


def save_checkpoint(path: str, model: torch.nn.Module, cfg, **kwargs) -> None:
    torch.save({
        "model": model.state_dict(),
        "cfg":   cfg.__dict__,
        **kwargs,
    }, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    device: str = "cpu", strict: bool = True) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=strict)
    return ckpt
