from argparse import Namespace
from copy import deepcopy
from typing import List
import torch
from fedavg import FedAvgServer
from src.client.fedDyS import FedDySClient


class FedDySServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedDyS",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedDySClient(deepcopy(self.model), self.args, self.logger, self.device, self.wandb_logger)

    @torch.no_grad()
    def aggregate(
            self,
            delta_cache: List[List[torch.Tensor]],
            weight_cache: List[int],
            return_diff=True,
    ):
        """
        This function is for aggregating recevied model parameters from selected clients.
        The method of aggregation is weighted averaging by default.

        Args:
            delta_cache (List[List[torch.Tensor]]): `delta` means the difference between client model parameters that before and after local training.

            weight_cache (List[int]): Weight for each `delta` (client dataset size by default).

            return_diff (bool): Differnt value brings different operations. Default to True.
        """
        print(f"weight_cache: {weight_cache}")
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        if return_diff:
            delta_list = [list(delta.values()) for delta in delta_cache]
            aggregated_delta = [
                torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
                for diff in zip(*delta_list)
            ]

            for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
                param.data -= diff
        else:
            for old_param, zipped_new_param in zip(
                    self.global_params_dict.values(), zip(*delta_cache)
            ):
                old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                    dim=-1
                )
        self.model.load_state_dict(self.global_params_dict, strict=False)


if __name__ == "__main__":
    server = FedDySServer()
    server.run()
