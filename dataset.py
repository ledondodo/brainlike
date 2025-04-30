import torch
from torch.utils.data import Dataset

class SpikesDataset(Dataset):
    def __init__(self, stimuli, spikes, device):
        self.stimuli = torch.tensor(stimuli, dtype=torch.float32, device=device)
        self.spikes = torch.tensor(spikes, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, idx):
        stimulus = self.stimuli[idx]
        spike = self.spikes[idx]
        return stimulus, spike
    
class ActivationsDataset(Dataset):
    def __init__(self, activations, spikes, device):
        self.activations = activations.to(device)
        self.spikes = torch.tensor(spikes, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        activation = self.activations[idx]
        spike = self.spikes[idx]
        return activation, spike