from torch import Tensor
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]

        # Convert the sample and target to PyTorch tensors if needed
        sample = Tensor(sample)
        target = Tensor(target)

        return sample, target

class CustomTextDataset(Dataset):
    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]

        # Convert the sample and target to PyTorch tensors if needed
        # text type sample = Tensor(sample)
        target = Tensor(target)

        return sample, target
