from typing import List, Dict, Tuple, Union, Optional
import numpy as np
from numpy import log2
from fedavg import FedAvgClient
import itertools
from collections import Counter, OrderedDict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.config.utils import trainable_params
from src.utls.dataset import CustomDataset


def eb(cntr, num_classes):
    n = sum(cntr)
    k = num_classes
    H = 0
    for val in cntr:
        if val != 0:
            H += (val / n) * log2((val / n))

    H = -H  # Shannon Diversity Index
    return H / log2(k)  # "Shannon equitability Index"


def entropy_balance(lst, num_classes):
    # return entropy(seq)
    n = len(lst)
    counter_seq = [0] * num_classes
    for (k, v) in Counter(lst).items():
        counter_seq[k] = v
    return eb(counter_seq, num_classes)


def get_labels_from_dataloader(dataloader: DataLoader) -> Optional[List[int]]:
    dataset = dataloader.dataset

    # If the dataset is a Subset
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices

        # Handle different types of datasets
        if hasattr(original_dataset, 'targets'):  # Direct attribute access
            return [original_dataset.targets[i] for i in indices]
        elif hasattr(original_dataset, 'imgs'):  # For datasets like ImageFolder
            return [original_dataset.imgs[i][1] for i in indices]
        elif hasattr(original_dataset, 'samples'):  # Similar to ImageFolder but uses 'samples'
            return [original_dataset.samples[i][1] for i in indices]
    else:
        # Direct dataset
        if hasattr(dataset, 'targets'):
            return dataset.targets
        elif hasattr(dataset, 'imgs'):
            return [img[1] for img in dataset.imgs]
        elif hasattr(dataset, 'samples'):
            return [sample[1] for sample in dataset.samples]

    # If none of the conditions are met, return None or raise an error
    return None


def dynamic_sample_selection(model, trainloader, class_counts, client_idx, logger, batch_size,
                             rnd=-1, epoch=None, n_classes=10, device=torch.device("cpu"),
                             th_mode='mean', th=None):
    model.eval()
    class_counts = torch.Tensor(class_counts).to(device)

    # Gather data from trainloader
    data_items, labels = zip(*trainloader)
    data_items = list(itertools.chain.from_iterable(data_items))
    labels = list(itertools.chain.from_iterable(labels))
    dataset = CustomDataset(data_items, labels)
    new_trainloader = DataLoader(dataset, batch_size=len(data_items), shuffle=False)

    # find minority classes
    thd = torch.mean(class_counts)
    minority_classes = torch.where(class_counts < thd)[0]
    is_minority_class = torch.tensor([t.item() in minority_classes for t in labels])

    selected_inputs = []
    selected_targets = []

    for data, target in new_trainloader:
        data, target = data.to(device), target.to(device)
        logits = F.softmax(model(data), dim=1)
        probabilities, y_pred = logits.data.max(1, keepdim=True)
        if torch.isnan(probabilities).any():
            logger.log(f"!!! probabilities is NAN")
            return trainloader

        # Calculate confidence and next_max values
        non_max_logits = logits.clone()
        rows = torch.arange(y_pred.size(0))
        non_max_logits[rows, y_pred.squeeze()] = -1
        next_max, _ = non_max_logits.data.max(1, keepdim=True)
        confidences = probabilities.squeeze() - next_max.squeeze()
        if is_minority_class.dim() > 0:
            confidences[is_minority_class] /= 2  # penalizing minority classes to increase the chance of selection
        selection_metric = torch.exp(confidences)  # Inverse metric: higher value for higher confidence

        threshold = None
        if th_mode == 'mean':
            threshold = torch.mean(selection_metric)
        elif th_mode == 'percentage' and th is not None:
            threshold = torch.quantile(selection_metric, th)
        else:
            raise Exception("threshold is not set")

        logger.log(f"threshold: {threshold}, min(selection_metric): {torch.min(selection_metric)} max(selection_metric): {torch.max(selection_metric)}")
        keep_indices = selection_metric.le(threshold).nonzero().flatten()
        if len(keep_indices) == 0:
            break
        selected_indices = torch.LongTensor(list(keep_indices))
        selected_inputs.extend(data[selected_indices])
        selected_targets.extend(target[selected_indices])

        logger.log(f'round {rnd} client {client_idx} epoch {epoch}'
                   f' total->selected:{len(dataset)}->{len(selected_targets)}, '
                   f' removed:{len(dataset) - len(selected_targets)}, '
                   f' entropy:{entropy_balance(labels, n_classes):.3f}->{entropy_balance(selected_targets, n_classes):.3f}'
                   )

    if len(selected_inputs) == 0:
        return None

    logger.log(f'info_gain: r{rnd}, c{client_idx} - {len(selected_inputs)} sample(s)')
    print(f"selected_targets:{len(selected_targets)}")
    dataset = CustomDataset(selected_inputs, selected_targets)
    dynloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dynloader


class FedDySClient(FedAvgClient):
    def __init__(self, model, args, logger, device, wandb_loger):
        super().__init__(model, args, logger, device, wandb_loger)
        self.class_num = len(self.dataset.classes)

    def train(
            self,
            client_id: int,
            local_epoch: int,
            new_parameters: OrderedDict[str, torch.Tensor],
            current_epoch: int,
            return_diff=True,
            verbose=False,
    ) -> Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
        """
        The funtion for including all operations in client local training phase.
        If you wanna implement your method, consider to override this funciton.

        Args:
            client_id (int): The ID of client.

            local_epoch (int): The number of epochs for performing local training.

            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.

            return_diff (bool, optional):
            Set as `True` to send the difference between FL model parameters that before and after training;
            Set as `False` to send FL model parameters without any change.  Defaults to True.

            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
            [The difference / all trainable parameters, the weight of this client, the evaluation metric stats].
        """
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)
        self.current_epoch = current_epoch

        if self.args.wandb and self.args.algo == "FedAvg":
            self.wandb_logger.experiment.log(
                {f"client_{self.client_id}/total_num_samples": len(self.trainset)}
                , step=self.current_epoch)

        eval_stats = self.train_and_log(verbose=verbose)

        entropy = entropy_balance(get_labels_from_dataloader(self.trainloader), num_classes=self.class_num)
        # print(f"entropy is {entropy}")
        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                    new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return delta, entropy, eval_stats
        else:
            return (
                trainable_params(self.model, detach=True),
                entropy,
                eval_stats,
            )

    def fit(self):
        """
        The function for specifying operations in local training phase.
        If you wanna implement your method and your method has different local training operations to FedAvg, this method has to be overrided.
        """
        self.model.train()
        targets = []
        for batch_idx, (_, target) in enumerate(self.trainloader):
            targets.extend(target.tolist())
        class_counts = [0] * self.class_num
        for (k, v) in Counter(targets).items():
            class_counts[k] = v

        e = -1
        self.num_selected_samples = []
        for e in range(self.local_epoch):

            dyn_loader = dynamic_sample_selection(self.model, self.trainloader, class_counts, self.client_id,
                                                   self.logger, batch_size=self.args.batch_size,
                                                   rnd=self.current_epoch,
                                                   epoch=e, n_classes=self.class_num,
                                                   device=self.device,
                                                   th_mode=self.args.th_mode,
                                                   th=self.args.th)

            num_samples = len(dyn_loader.dataset) if dyn_loader and hasattr(dyn_loader, 'dataset') else 0
            self.num_selected_samples.append(num_samples)

            if dyn_loader is None:
                self.logger.log(f'No samples are selected for epoch {e}')
                break
            self.trainloader = dyn_loader

            self.model.train()
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.args.wandb:
            self.wandb_logger.experiment.log(
                {f"client_{self.client_id}/num_selected_samples(avg)": np.mean(self.num_selected_samples)},
                step=self.current_epoch)
