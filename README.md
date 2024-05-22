# FedDyS: Enhancing Federated Learning Efficiency with Dynamic Sample Selection

This is the code accompanying IEEE ISCC - DistInSys 2024 paper "FedDyS: Enhancing Federated Learning Efficiency with Dynamic Sample Selection "
Paper link: []

## Overview

FedDyS is a dynamic sample selection technique designed to enhance federated learning (FL) by reducing computational demands and mitigating data heterogeneity. It achieves this by eliminating non-essential training samples, which not only shrinks the training set size on local devices but also enhances data diversity. FedDyS preserves data privacy and prevents the catastrophic forgetting effect, common in FL, leading to improved accuracy and faster convergence. This makes FedDyS particularly suitable for low-resource settings, achieving optimal results with less than 15% of the usual number of training samples.


### used baseline FL Methods

- ***FedAvg*** -- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (AISTATS'17)

- ***FedProx*** -- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) (MLSys'20)

- ***SCAFFOLD*** -- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) (ICML'20)

- ***MOON*** -- [Model-Contrastive Federated Learning](http://arxiv.org/abs/2103.16257) (CVPR'21)
- ***FedSampling*** -- [FedSampling: A Better Sampling Strategy for Federated Learning](https://www.ijcai.org/proceedings/2023/0462.pdf) (JCAI-23)

## Environment Preparation

Note that this code needs `3.10 <= python < 3.12`.
To install necessary packages: 
```
pip install -r requirements.txt
```

## Data Preparation

```shell
# partition the CIFAR-10 according to Dir(0.1) for 100 clients
cd data
python generate_data.py -d cinic10 -a 0.1 -cn 100
cd ../
```

## Run Examples to start
```shell
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_mean" --th_mode mean --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_1%" --th_mode percentage --th 0.01 --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_5%" --th_mode percentage --th 0.05 --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_15%" --th_mode percentage --th 0.15 --algo FedDyS
python3 -u ./src/server/fedDyS2.python3 -u ./src/server/fedavg.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "FedAVG_CINIC10_transfer" --algo FedAVG
python3 -u ./src/server/fedprox.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "FedProx_CINIC10_transfer" --algo FedProx
python3 -u ./src/server/moon.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "moon_CINIC10_transfer" --algo MOON
python3 -u ./src/server/fedsampling.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedsampling_CINIC10_transfer" --algo FedSampling
```

## Generic Arguments 🔧

| Argument                       | Description                                                                                                                                                                                                                                                                                                                               |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dataset`                    | The name of dataset that experiment run on.                                                                                                                                                                                                                                                                                               |
| `--model`                      | The model backbone experiment used.                                                                                                                                                                                                                                                                                                       |
| `--seed`                       | Random seed for running experiment.                                                                                                                                                                                                                                                                                                       |
| `--join_ratio`                 | Ratio for (client each round) / (client num in total).                                                                                                                                                                                                                                                                                    |
| `--global_epoch`               | Global epoch, also called communication round.                                                                                                                                                                                                                                                                                            |
| `--local_epoch`                | Local epoch for client local training.                                                                                                                                                                                                                                                                                                    |
| `--finetune_epoch`             | Epoch for clients fine-tunning their models before test.                                                                                                                                                                                                                                                                                  |
| `--test_gap`                   | Interval round of performing test on clients.                                                                                                                                                                                                                                                                                             |
| `--eval_test`                  | Non-zero value for performing evaluation on joined clients' testset before and after local training.                                                                                                                                                                                                                                      |
| `--eval_train`                 | Non-zero value for performing evaluation on joined clients' trainset before and after local training.                                                                                                                                                                                                                                     |
| `--local_lr`                   | Learning rate for client local training.                                                                                                                                                                                                                                                                                                  |
| `--momentum`                   | Momentum for client local opitimizer.                                                                                                                                                                                                                                                                                                     |
| `--weight_decay`               | Weight decay for client local optimizer.                                                                                                                                                                                                                                                                                                  |
| `--verbose_gap`                | Interval round of displaying clients training performance on terminal.                                                                                                                                                                                                                                                                    |
| `--batch_size`                 | Data batch size for client local training.                                                                                                                                                                                                                                                                                                |
| `--use_cuda`                   | Non-zero value indicates that tensors are in gpu.                                                                                                                                                                                                                                                                                         |
| `--visible`                    | Non-zero value for using Visdom to monitor algorithm performance on `localhost:8097`.                                                                                                                                                                                                                                                     |
| `--global_testset`             | Non-zero value for evaluating client models over the global testset before and after local training, instead of evaluating over clients own testset. The global testset is the union set of all client's testset. *NOTE: Activating this setting will considerably slow down the entire training process, especially the dataset is big.* |
| `--save_log`                   | Non-zero value for saving algorithm running log in `FL-bench/out/${algo}`.                                                                                                                                                                                                                                                                |
| `--straggler_ratio`            | The ratio of stragglers (set in `[0, 1]`). Stragglers would not perform full-epoch local training as normal clients. Their local epoch would be randomly selected from range `[--straggler_min_local_epoch, --local_epoch)`.                                                                                                              |
| `--straggler_min_local_epoch`  | The minimum value of local epoch for stragglers.                                                                                                                                                                                                                                                                                          |
| `--external_model_params_file` | The relative file path of external (pretrained) model parameters (`*.pt`). e.g., `../../out/FedAvg/mnist_100_lenet5.pt`. Please confirm whether the shape of parameters compatible with the model by yourself. ⚠ This feature is enabled only when `unique_model=False`, which is pre-defined by each FL method.                          |
| `--save_model`                 | Non-zero value for saving output model(s) parameters in `FL-bench/out/${algo}`.  The default file name pattern is `${dataset}_${global_epoch}_${model}.pt`.                                                                                                                                                                               |
| `--save_fig`                   | Non-zero value for saving the accuracy curves showed on Visdom into a `.jpeg` file at `FL-bench/out/${algo}`.                                                                                                                                                                                                                             |
| `--save_metrics`               | Non-zero value for saving metrics stats into a `.csv` file at `FL-bench/out/${algo}`.                                                                                                                                                                                                                                                     |
| `--wandb`                      | whether to use Weights & Biases (W&B)                                                                                                                                                                                                                                                                                                     |
| `--run_name`                   | the name of the experiment run for Weights & Biases (W&B)                                                                                                                                                                                                                                                                                 |
| `--algo`                       | algorithm name                                                                                                                                                                                                                                                                                                                            |
| `--th_mode`                    | threshold mode : mean or percentage                                                                                                                                                                                                                                                                                                       |
| `--th`                         | threshold percentage                                                                                                                                                                                                                                                                                                                      |

## ⭐
This repository is derived from the FL-bench project: (for more details and to access more methods check this repository)
https://github.com/KarhouTam/FL-bench/