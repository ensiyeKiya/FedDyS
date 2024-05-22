import pickle
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.utls.language import process_x

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

from src.config.utils import trainable_params, evaluate, Logger
from src.config.models import DecoupledModel
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS


class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        args: Namespace,
        logger: Logger,
        device: torch.device,
        wandb_loger
    ):
        self.args = args
        self.device = device
        self.model = model.to(self.device)
        self.client_id: int = None
        self.wandb_logger = wandb_loger

        # load dataset and clients' data indices
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        self.data_indices: List[List[int]] = partition["data_indices"]

        if self.model.is_language_model:
            general_data_transform = None
            general_target_transform = None
            train_data_transform = None
            train_target_transform = None
        else:
            # --------- you can define your own data transformation strategy here ------------
            general_data_transform = transforms.Compose(
                [transforms.Normalize(MEAN[self.args.dataset], STD[self.args.dataset])]
            )
            general_target_transform = transforms.Compose([])
            train_data_transform = transforms.Compose([])
            train_target_transform = transforms.Compose([])
            # --------------------------------------------------------------------------------

        self.dataset = DATASETS[self.args.dataset](
            root=PROJECT_DIR / "data" / args.dataset,
            args=args.dataset_args,
            general_data_transform=general_data_transform,
            general_target_transform=general_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )

        self.trainloader: DataLoader = None
        self.testloader: DataLoader = None
        self.trainset: Subset = Subset(self.dataset, indices=[])
        self.testset: Subset = Subset(self.dataset, indices=[])
        self.global_testset: Subset = None
        if self.args.global_testset:
            all_testdata_indices = []
            for indices in self.data_indices:
                all_testdata_indices.extend(indices["test"])
            self.global_testset = Subset(self.dataset, all_testdata_indices)

        self.local_epoch = self.args.local_epoch

        if self.model.is_binary_classification:
            self.criterion = torch.nn.BCELoss().to(self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.logger = logger
        self.personal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.personal_params_name: List[str] = []
        self.init_personal_params_dict: Dict[str, torch.Tensor] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if not param.requires_grad
        }
        self.opt_state_dict = {}

        if self.model.is_language_model:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.local_lr)
        else:
            self.optimizer = torch.optim.SGD(
                params=trainable_params(self.model),
                lr=self.args.local_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())
        self.class_num = len(self.dataset.classes)

    def load_dataset(self):
        """This function is for loading data indices for No.`self.client_id` client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.testset.indices = self.data_indices[self.client_id]["test"]
        self.trainloader = DataLoader(self.trainset, self.args.batch_size)
        if self.args.global_testset:
            self.testloader = DataLoader(self.global_testset, self.args.batch_size)
        else:
            self.testloader = DataLoader(self.testset, self.args.batch_size)

    def train_and_log(self, verbose=False) -> Dict[str, Dict[str, float]]:
        """This function includes the local training and logging process.

        Args:
            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: The logging info, which contains metric stats.
        """
        before = {
            "train_loss": 0,
            "test_loss": 0,
            "train_correct": 0,
            "test_correct": 0,
            "train_size": 1,
            "test_size": 1,
        }
        after = deepcopy(before)
        before = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            self.save_state()
            after = self.evaluate()

        if self.args.wandb:
            if len(self.trainset) > 0 and self.args.eval_train:
                self.wandb_logger.experiment.log({
                    f"client_{self.client_id}/train_size_before": before["train_size"],
                    f"client_{self.client_id}/train_size_after": after["train_size"],
                    f"client_{self.client_id}/train_loss_before": before["train_loss"] / before["train_size"],
                    f"client_{self.client_id}/train_loss_after": after["train_loss"] / after["train_size"],
                    f"client_{self.client_id}/train_correct_before": before["train_correct"] / before["train_size"] * 100.0,
                    f"client_{self.client_id}/train_correct_after": after["train_correct"] / after["train_size"] * 100.0,
                    f"client_{self.client_id}/macro_f1_before": before["macro_f1"],
                    f"client_{self.client_id}/macro_f1_after": after["macro_f1"]
                }, step=self.current_epoch)
            if len(self.testset) > 0 and self.args.eval_test:
                self.wandb_logger.experiment.log({
                    f"client_{self.client_id}/test_size_before": before["test_size"],
                    f"client_{self.client_id}/test_size_after": after["test_size"],
                    f"client_{self.client_id}/test_loss_before": before["test_loss"] / before["test_size"],
                    f"client_{self.client_id}/test_loss_after": after["test_loss"] / after["test_size"],
                    f"client_{self.client_id}/test_correct_before": before["test_correct"] / before["test_size"] * 100.0,
                    f"client_{self.client_id}/test_correct_after": after["test_correct"] / after["test_size"] * 100.0,
                    f"client_{self.client_id}/macro_f1_before": before["macro_f1"],
                    f"client_{self.client_id}/macro_f1_after": after["macro_f1"]
                }, step=self.current_epoch)
        if verbose:
            if len(self.trainset) > 0 and self.args.eval_train:
                self.logger.log(
                    "client [{}] (train)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}% [bold red]m_f1: {:.2f} -> {:.2f}".format(
                        self.client_id,
                        before["train_loss"] / before["train_size"],
                        after["train_loss"] / after["train_size"],
                        before["train_correct"] / before["train_size"] * 100.0,
                        after["train_correct"] / after["train_size"] * 100.0,
                        before["macro_f1"],
                        after["macro_f1"]
                    )
                )
            if len(self.testset) > 0 and self.args.eval_test:
                self.logger.log(
                    "client [{}] (test)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}% [bold red]m_f1: {:.2f} -> {:.2f}".format(
                        self.client_id,
                        before["test_loss"] / before["test_size"],
                        after["test_loss"] / after["test_size"],
                        before["test_correct"] / before["test_size"] * 100.0,
                        after["test_correct"] / after["test_size"] * 100.0,
                        before["macro_f1"] * 100.0,
                        after["macro_f1"] * 100.0
                    )
                )

        eval_stats = {"before": before, "after": after}
        return eval_stats

    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        """Load model parameters received from the server.

        Args:
            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.
        """
        personal_parameters = self.personal_params_dict.get(
            self.client_id, self.init_personal_params_dict
        )
        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )
        self.model.load_state_dict(new_parameters, strict=False)
        # personal params would overlap the dummy params from new_parameters from the same layers
        self.model.load_state_dict(personal_parameters, strict=False)

    def save_state(self):
        """Save client model personal parameters and the state of optimizer at the end of local training."""
        self.personal_params_dict[self.client_id] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (key in self.personal_params_name)
        }
        self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())

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

        if self.args.wandb and hasattr(self.args,"algo") and self.args.algo == "FedAVG":
            self.wandb_logger.experiment.log(
                {f"client_{self.client_id}/total_num_samples": len(self.trainset)}
                , step=self.current_epoch)

        eval_stats = self.train_and_log(verbose=verbose)

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return delta, len(self.trainset), eval_stats
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.trainset),
                eval_stats,
            )

    def fit(self):
        """
        The function for specifying operations in local training phase.
        If you wanna implement your method and your method has different local training operations to FedAvg, this method has to be overrided.
        """
        self.model.train()
        for _ in range(self.local_epoch):
            if self.model.is_language_model:
                hidden_state = self.model.init_hidden(self.trainloader.batch_size)
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                if self.model.is_language_model:
                    x = process_x(x)
                    y = y.float()
                    y.requires_grad = False
                    if x.size(0) < self.args.batch_size:
                        break
                    x, y = x.to(self.device), y.to(self.device)
                    hidden_state = tuple([each.data for each in hidden_state])
                    logits, hidden_state = self.model(x, hidden_state)
                else:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)

                if self.model.is_binary_classification:
                    loss = self.criterion(logits.squeeze(), y.float())
                else:
                    loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                if self.model.is_language_model:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None) -> Dict[str, float]:
        """The evaluation function. Would be activated before and after local training if `eval_test = True` or `eval_train = True`.

        Args:
            model (torch.nn.Module, optional): The target model needed evaluation (set to `None` for using `self.model`). Defaults to None.

        Returns:
            Dict[str, float]: The evaluation metric stats.
        """
        # disable train data transform while evaluating
        self.dataset.enable_train_transform = False

        eval_model = self.model if model is None else model
        eval_model.eval()
        train_loss, test_loss = 0, 0
        train_correct, test_correct = 0, 0
        train_sample_num, test_sample_num = 0, 0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        macro_f1 = 0.0
        if len(self.testset) > 0 and self.args.eval_test:
            test_loss, test_correct, test_sample_num, macro_f1 = evaluate(
                model=eval_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
                class_num=self.class_num,
            )

        if len(self.trainset) > 0 and self.args.eval_train:
            train_loss, train_correct, train_sample_num, macro_f1 = evaluate(
                model=eval_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
                class_num=self.class_num
            )

        self.dataset.enable_train_transform = True

        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_correct": train_correct,
            "test_correct": test_correct,
            "train_size": float(max(1, train_sample_num)),
            "test_size": float(max(1, test_sample_num)),
            "macro_f1": macro_f1
        }

    def test(
        self, client_id: int, new_parameters: OrderedDict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Test function. Only be activated while in FL test round.

        Args:
            client_id (int): The ID of client.
            new_parameters (OrderedDict[str, torch.Tensor]): The FL model parameters.

        Returns:
            Dict[str, Dict[str, float]]: the evalutaion metrics stats.
        """
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)

        before = {
            "train_loss": 0,
            "train_correct": 0,
            "train_size": 1.0,
            "test_loss": 0,
            "test_correct": 0,
            "test_size": 1.0,
            "macro_f1": 0
        }
        after = deepcopy(before)

        before = self.evaluate()
        if self.args.finetune_epoch > 0:
            self.finetune()
            after = self.evaluate()
        return {"before": before, "after": after}

    def finetune(self):
        """
        The fine-tune function. If your method has different fine-tuning opeation, consider to override this.
        This function will only be activated while in FL test round.
        """
        self.model.train()
        for _ in range(self.args.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
