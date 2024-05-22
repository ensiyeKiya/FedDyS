import numpy as np
from numpy import log2

from fedavg import FedAvgClient
import itertools
from collections import Counter
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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


def local_sampling(candidate_samples, K, hatN, ):
    sample_choice = np.random.binomial(size=(len(candidate_samples),), n=1, p=K / hatN)
    candidate_samples = candidate_samples[sample_choice == 1]
    return candidate_samples


class FedSamplingClient(FedAvgClient):
    def __init__(self, model, args, logger, device, wandb_loger):
        super().__init__(model, args, logger, device, wandb_loger)
        self.class_num = len(self.dataset.classes)
        # self.alpha = alpha
        # self.train_users = train_users

    def set_train_params(self, hat_n, K, r):
        self.hat_n = hat_n
        self.K = K
        self.r = r

    def fit(self):
        """
        The function for specifying operations in local training phase.
        If you wanna implement your method and your method has different local training operations to FedAvg, this method has to be overrided.
        """
        self.model.train()
        targets = []
        inputs = []
        for batch_idx, (input, target) in enumerate(self.trainloader.dataset):
            inputs.append(input)
            targets.append(target)
        class_counts = [0] * self.class_num
        for (k, v) in Counter(targets).items():
            class_counts[k] = v

        self.num_selected_samples = []
        for e in range(self.local_epoch):
            sample_ids = local_sampling(torch.arange(0, len(inputs), dtype=torch.int32), self.K, int(self.hat_n * self.r))
            sampled_inputs = [inputs[i.item()] for i in sample_ids]
            sampled_targets = [targets[i.item()] for i in sample_ids]

            length = len(sampled_inputs)
            self.num_selected_samples.append(length)
            dataset = CustomDataset(sampled_inputs, sampled_targets)
            trainloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

            for i, (x, y) in enumerate(trainloader.dataset):
                if i > 0 and x.shape != prev_shape:
                    print(f"Inconsistent shape at index {i}. Previous: {prev_shape}, Current: {x.shape}")
                if i > 0 and y.shape != y_prev_shape:
                    print(f"Inconsistent shape at index {i}. Previous: {y_prev_shape}, Current: {y.shape}")
                prev_shape = x.shape
                y_prev_shape = y.shape
                y_prev = y
            for x, y in trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.args.wandb:
            self.wandb_logger.experiment.log(
                {f"client_{self.client_id}/num_selected_samples(avg)": np.mean(self.num_selected_samples)},
                step=self.current_epoch)
