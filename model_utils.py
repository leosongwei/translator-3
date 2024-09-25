import torch

def set_requires_grad(model: torch.nn.Module, should_enable):
    for p in model.parameters():
        p.requires_grad_(should_enable)