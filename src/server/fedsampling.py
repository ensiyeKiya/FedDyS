from argparse import Namespace
from copy import deepcopy
import numpy as np
from fedavg import FedAvgServer
from src.client.fedsampling import FedSamplingClient


class Estimator:
    def __init__(self, train_users, alpha, M):
        self.M = M
        self.alpha = alpha
        self.train_users = train_users

    def query(self, userid):
        fake_response = np.random.randint(1,self.M)
        real_response = len(self.train_users[userid])
        choice = np.random.binomial(n=1, p=self.alpha)
        response = choice*real_response + (1-choice)*fake_response
        return response

    def estimate(self,):
        R = 0
        for uid in range(len(self.train_users)):
            R += self.query(uid)
        hat_n = (R-len(self.train_users)*(1-self.alpha)*self.M/2)/self.alpha
        hat_n = max(hat_n, len(self.train_users))
        return hat_n

class FedSamplingServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedSampling",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedSamplingClient(deepcopy(self.model), self.args, self.logger, self.device, self.wandb_logger)
        self.estimator = Estimator([cl["train"] for cl in self.trainer.data_indices], alpha=0.5, M=700)  # todo check alpha, M

    def train_one_round(self):
        self.hat_n = self.estimator.estimate()
        delta_cache = []
        weight_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            self.trainer.set_train_params(self.hat_n, K=1200, r=1.0)
            (
                delta,
                weight,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                current_epoch=self.current_epoch,
                new_parameters=client_local_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )

            delta_cache.append(delta)
            weight_cache.append(weight)

        self.aggregate(delta_cache, weight_cache)


if __name__ == "__main__":
    server = FedSamplingServer()
    server.run()
