import os
import random
from copy import deepcopy
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple, Union
from pathlib import Path

import torch
import pynvml
import numpy as np
from torch.utils.data import DataLoader

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union
from rich.console import Console
from torcheval.metrics.functional import multiclass_f1_score

from data.utils.datasets import BaseDataset
# circular import from src.config.models import DecoupledModel
from src.utls.language import process_x

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = PROJECT_DIR / "out"
TEMP_DIR = PROJECT_DIR / "temp"


def fix_random_seed(seed: int) -> None:
    """Fix the random seed of FL training.

    Args:
        seed (int): Any number you like as the random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_best_device(use_cuda: bool) -> torch.device:
    """Dynamically select the vacant CUDA device for running FL experiment.

    Args:
        use_cuda (bool): `True` for using CUDA; `False` for using CPU only.

    Returns:
        torch.device: The selected CUDA device.
    """
    # This function is modified by the `get_best_gpu()` in https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/functional.py
    # Shout out to FedLab, which is an incredible FL framework!
    if not torch.cuda.is_available() or not use_cuda:
        return torch.device("cpu")
    pynvml.nvmlInit()
    gpu_memory = []
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        assert max(gpu_ids) < torch.cuda.device_count()
    else:
        gpu_ids = range(torch.cuda.device_count())

    for i in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append(memory_info.free)
    gpu_memory = np.array(gpu_memory)
    best_gpu_id = np.argmax(gpu_memory)
    return torch.device(f"cuda:{best_gpu_id}")


def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
    detach=False,
    requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.

    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): The source that contains parameters.
        requires_name (bool, optional): If set to `True`, The names of parameters would also return in another list. Defaults to False.
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`. Defaults to False.

    Returns:
        Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]: List of parameters, [List of names of parameters].
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters


def vectorize(
    src: Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], detach=True
) -> torch.Tensor:
    """Vectorize and concatenate all tensors in `src`.

    Args:
        src (Union[OrderedDict[str, torch.Tensor]List[torch.Tensor]]): The source of tensors.
        detach (bool, optional): Set to `True`, return the `.detach().clone()`. Defaults to True.

    Returns:
        torch.Tensor: The vectorized tensor.
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    if isinstance(src, list):
        return torch.cat([func(param).flatten() for param in src])
    elif isinstance(src, OrderedDict):
        return torch.cat([func(param).flatten() for param in src.values()])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
    class_num=None,
) -> Tuple[float, float, int, float]:
    """For evaluating the `model` over `dataloader` and return the result calculated by `criterion`.

    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Target dataloader.
        criterion (optional): The metric criterion. Defaults to torch.nn.CrossEntropyLoss(reduction="sum").
        device (torch.device, optional): The device that holds the computation. Defaults to torch.device("cpu").

    Returns:
        Tuple[float, float, int, float]: [metric, correct, sample num, macro_f1]
    """
    model.eval()
    correct = 0
    loss = 0
    sample_num = 0
    all_y = []
    all_pred = []
    if model.is_language_model:
        hidden_state = model.init_hidden(dataloader.batch_size)
    for x, y in dataloader:
        if model.is_language_model:
            x = process_x(x)
            y = y.float()
            if x.size(0) < dataloader.batch_size:
                continue  # skip last batch
            x, y = x.to(device), y.to(device)
            logits, hidden_state = model(x, hidden_state)
        else:
            x, y = x.to(device), y.to(device)
            logits = model(x)

        loss += criterion(logits, y).item()
        if model.is_binary_classification:
            pred = torch.round(logits)
        else:
            pred = torch.argmax(logits, -1)
        correct += (pred == y).sum().item()
        sample_num += len(y)
        all_y.extend(y)
        all_pred.extend(pred)

    # macro_f1 = multiclass_f1_score(torch.IntTensor(all_y),torch.IntTensor(all_pred), num_classes=class_num)
    macro_f1 = torch.tensor(0.0)
    if all_pred:
        macro_f1 = multiclass_f1_score(torch.stack(all_y).int(),torch.stack(all_pred).int(), num_classes=class_num)
    return loss, correct, sample_num, macro_f1


def count_labels(
    dataset: BaseDataset, indices: List[int] = None, min_value=0
) -> List[int]:
    """For counting number of labels in `dataset.targets`.

    Args:
        dataset (BaseDataset): Target dataset.
        indices (List[int]): the subset indices. Defaults to all indices of `dataset` if not specified.
        min_value (int, optional): The minimum value for each label. Defaults to 0.

    Returns:
        List[int]: The number of each label.
    """
    if indices is None:
        indices = list(range(len(dataset.targets)))
    counter = Counter(dataset.targets[indices].tolist())
    return [counter.get(i, min_value) for i in range(len(dataset.classes))]


class Logger:
    def __init__(
        self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
        """This class is for solving the incompatibility between the progress bar and log function in library `rich`.

        Args:
            stdout (Console): The `rich.console.Console` for printing info onto stdout.
            enable_log (bool): Flag indicates whether log function is actived.
            logfile_path (Union[Path, str]): The path of log file.
        """
        self.stdout = stdout
        self.logfile_stream = None
        self.enable_log = enable_log
        if self.enable_log:
            self.logfile_stream = open(logfile_path, "w")
            self.logger = Console(
                file=self.logfile_stream, record=True, log_path=False, log_time=False
            )

    def log(self, *args, **kwargs):
        self.stdout.log(*args, **kwargs)
        if self.enable_log:
            self.logger.log(*args, **kwargs)

    def close(self):
        if self.logfile_stream:
            self.logfile_stream.close()

#
# class Logger:
#     def __init__(
#             self,
#             stdout: Console,
#             enable_log: bool,
#             logfile_path: Union[Path, str],
#             max_log_size: int = 10*1024*1024,  # 10MB by default
#             backup_count: int = 5  # Keep 5 backup logs by default
#     ):
#         """
#         Args:
#             stdout (Console): The `rich.console.Console` for printing info onto stdout.
#             enable_log (bool): Flag indicates whether log function is activated.
#             logfile_path (Union[Path, str]): The path of log file.
#             max_log_size (int): Maximum size in bytes before rotating the log file.
#             backup_count (int): Number of backup log files to keep.
#         """
#         self.stdout = stdout
#         self.enable_log = enable_log
#
#         if self.enable_log:
#             # Ensure the logfile_path is a Path object
#             logfile_path = Path(logfile_path)
#
#             # Set up rotating file handler
#             self.logfile_handler = RotatingFileHandler(
#                 logfile_path,
#                 maxBytes=max_log_size,
#                 backupCount=backup_count
#             )
#
#             # Set up a logger
#             self.logger = logging.getLogger("rich_logger")
#             self.logger.setLevel(logging.INFO)
#             self.logger.addHandler(self.logfile_handler)
#
#     def log(self, *args, **kwargs):
#         self.stdout.log(*args, **kwargs)
#         if self.enable_log:
#             message = " ".join(str(arg) for arg in args)
#             self.logger.info(message)
#
#     def close(self):
#         if self.enable_log:
#             handlers = self.logger.handlers[:]
#             for handler in handlers:
#                 handler.close()
#                 self.logger.removeHandler(handler)
