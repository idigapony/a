import torch

def get_optimizer(model, lr=5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr)