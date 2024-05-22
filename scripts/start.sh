export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH=/usr/local/lib/python3.9/site-packages/nvidia/cuda_runtime/lib/:$LD_LIBRARY_PATH
export WANDB_API_KEY=80a40f269eff3e2153b32ba2f076f4daa7273d1e;
export STF_WANDB_ACCT=onemoretest;
export STF_WANDB_PROJ=FedDyS_test;
export WANDB_LOG_DIR=./data/STF_LOG_DIR

#data splitting
#python3 -u ./data/generate_data.py -d cifar10 -a 0.1 -cn 10
#python3 -u ./data/generate_data.py -d cifar10 -a 1.0 -cn 10
#python3 -u ./data/generate_data.py -d imdb -a 1.0 -cn 50
#python3 -u ./data/generate_data.py -d cinic10 -a 1.0 -cn 100
#python3 -u ./data/generate_data.py -d cinic10 -a 0.5 -cn 100

#femnist
###FL-bench/data/femnist]# ./preprocess.sh -s niid --sf 1.0 -k 0 -t sample
#python3 -u ./data/generate_data.py -d femnist -a 1.0 -cn 100
#python3 -u ./data/generate_data.py -d femnist -a 0.1 -cn 100

#benchmarking
#python3 -u ./src/server/fedavgmDyS.py -d cifar10 -m mobile -jr 1.0 -ge 200 -le 1 --verbose_gap 1


####python3 -u ./src/server/fedavg.py -d imdb -m rnn -jr 1.0 -ge 10 --momentum 0.9 --weight_decay 0.0001 -le 1 --verbose_gap 1

# text dataset
#python3 -u ./src/server/fedDyS.py -d imdb -m rnn -jr 0.2 -ge 50 --momentum 0.9 --weight_decay 0.0001 -le 5 --verbose_gap 1 -lr 0.005
#python3 -u ./src/server/fedavg.py -d imdb -m rnn -jr 0.2 -ge 50 --momentum 0.9 --weight_decay 0.0001 -le 5 --verbose_gap 1 -lr 0.005


#python3 -u ./src/server/fedDyS.py -d cifar10 -m mobile -jr 1.0 -ge 200 -le 1 --verbose_gap 1 --wandb True --run_name fedDyS_tanh_1
#python3 -u ./src/server/fedavg.py -d cifar10 -m mobile -jr 1.0 -ge 200 -le 1 --verbose_gap 1 --wandb True --run_name fedAVG_tanh_1

#python3 -u ./src/server/fedavgm.py -d cifar10 -m mobile -jr 1.0 -ge 200 -le 1 --verbose_gap 1
#python3 -u ./src/server/fedprox.py -d cifar10 -m mobile -jr 1.0 -ge 200 -le 1 --verbose_gap 1
#python3 -u ./src/server/fedavgmDyS.py -d cifar10 -m mobile -jr 1.0 -ge 200 -le 1 --verbose_gap 1 #-ge 150
#python3 -u ./src/server/moon.py -d cifar10 -m mobile -jr 1.0 -ge 200 -le 1 --verbose_gap 1
#python3 -u ./src/server/scaffold.py -d cifar10 -m mobile -jr 1.0 -ge 200 -le 1 --verbose_gap 1

#python3 -u ./src/server/fedavgmDyS.py -d cifar10 -m mobile -jr 1.0 -ge 30 --verbose_gap 1 #-ge 150
#python3 -u ./src/server/fedsampling.py -d cifar10 -m mobile -jr 1.0 -ge 2000 -le 1 --verbose_gap 1 #-ge 150


#python3 -u ./src/server/fedavg.py -d femnist -m lenet5 -jr 0.1 -ge 10 -le 5 --verbose_gap 1
#python3 -u ./src/server/fedDyS.py -d femnist -m lenet5 -jr 0.1 -ge 10 -le 5 --verbose_gap 1
#python3 -u ./src/server/fedavg.py -d femnist -m lenet5 -jr 0.1 -ge 50 -le 1 --verbose_gap 1
#python3 -u ./src/server/fedDyS.py -d femnist -m lenet5 -jr 0.1 -ge 50 -le 1 --verbose_gap 1

#python3 -u ./src/server/fedavg.py -d femnist -m lenet5 -jr 0.1 -ge 500 -le 1 --verbose_gap 1
#python3 -u ./src/server/fedDyS.py -d femnist -m lenet5 -jr 0.1 -ge 2 -le 1 --verbose_gap 1

#python3 -u ./src/server/fedavg.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1
#python3 -u ./src/server/scaffold.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1
#python3 -u ./src/server/fedprox.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1
#python3 -u ./src/server/moon.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1

#python3 -u ./src/server/fedavg.py -d femnist -m res18 -jr 0.01 -ge 200 -le 5 --verbose_gap 1
#python3 -u ./src/server/fedprox.py -d femnist -m res18 -jr 0.01 -ge 200 -le 5 --verbose_gap 1
#python3 -u ./src/server/moon.py -d femnist -m res18 -jr 0.01 -ge 200 -le 5 --verbose_gap 1
#rm -f ./tmp/moon*
#python3 -u ./src/server/scaffold.py -d femnist -m res18 -jr 0.01 -ge 200 -le 5 --verbose_gap 1
#rm -f ./tmp/scaffold*

#ResNet18
#python3 -u ./src/server/fedavg.py -d cinic10 -m res18 -jr 0.1 -ge 500 -le 5 --verbose_gap 1

#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb True --run_name fedDyS_tanh_2
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb True --run_name "fedDyS_tanh_2_emp_x"
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb True --run_name "fedDyS_tanh_2_emp_x_0.1"


#data distribution
#python3 -u ./data/generate_data.py -d cinic10 -a 1.0 -cn 100

#benchmark
#python3 -u ./src/server/fedavgmDyS.py -d cinic10 -m mobile -jr 0.1 -ge 20 -le 5 --verbose_gap 1 --wandb True --run_name "fedAvgMDyS_frozen"
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "FedDyS_new"
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "FedDyS_frozen_lr0.005" --local_lr 0.005
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "FedDyS" --local_lr 0.001
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "FedDyS_lr0.001" --local_lr 0.001
#python3 -u ./src/server/fedDyS.py -d cinic10 -m lenet5 -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "FedDyS_lenet5"
#python3 -u ./src/server/fedavg.py -d cinic10 -m lenet5 -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "FedAVG_lenet5"

#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 20 -le 5 --verbose_gap 1 --wandb True --run_name "FedDyS_orig noshuffle"
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 20 -le 5 --verbose_gap 1 --wandb True --run_name "FedDyS_orig noshuffle lr 0.001" --local_lr 0.001
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "FedDyS_entropy_score_0% =FedAVG noshuffle lr 0.001 lmomentum 0.45" --local_lr 0.001 --momentum 0.45

# we should repeat these tests for feddys and alpha = 0.1, 0.5, 1.0
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_mean" --th_mode mean --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_1%" --th_mode percentage --th 0.01 --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_5%" --th_mode percentage --th 0.05 --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_15%" --th_mode percentage --th 0.15 --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_25%" --th_mode percentage --th 0.25 --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_50%" --th_mode percentage --th 0.50 --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_70%" --th_mode percentage --th 0.70 --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_90%" --th_mode percentage --th 0.90 --algo FedDyS

# we should repeat these tests for other methods (fedavg, fedprox, moon, fedsampling) and alpha = 0.1, 0.5, 1.0
python3 -u ./src/server/fedavg.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "FedAVG_CINIC10_transfer" --algo "FedAVG"
python3 -u ./src/server/fedprox.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "FedProx_CINIC10_transfer" --algo "FedProx"
python3 -u ./src/server/moon.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "moon_CINIC10_transfer" --algo "MOON"
python3 -u ./src/server/fedsampling.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedsampling_CINIC10_transfer" --algo "FedSampling"

# for femnist dataset for mobilenet-v2
python3 -u ./src/server/fedDyS2.py -d femnist -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_mean" --th_mode mean --algo FedDyS
python3 -u ./src/server/fedDyS2.py -d femnist -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_CINIC10_transfer_th_1%" --th_mode percentage --th 0.01 --algo FedDyS
python3 -u ./src/server/fedavg.py -d femnist -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "FedAVG_FEMNIST_MOBILE_transfer" --algo "FedAVG"
python3 -u ./src/server/fedprox.py -d femnist -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "FedProx_FEMNIST_MOBILE_transfer" --algo "FedProx"
python3 -u ./src/server/moon.py -d femnist -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "moon_FEMNIST_MOBILE_transfer" --algo "MOON"
python3 -u ./src/server/fedsampling.py -d femnist -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedsampling_FEMNIST_MOBILE_transfer" --algo "FedSampling"
## for femnist dataset for res18
#python3 -u ./src/server/fedDyS2.py -d femnist -m res18 -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_FEMNIST_res18_th_mean" --th_mode mean --algo FedDyS
#python3 -u ./src/server/fedDyS2.py -d femnist -m res18 -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_FEMNIST_res18_th_5%" --th_mode percentage --th 0.05 --algo FedDyS
#python3 -u ./src/server/fedDyS2.py -d femnist -m res18 -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedDyS2_FEMNIST_res18_th_20%" --th_mode percentage --th 0.2 --algo FedDyS
#python3 -u ./src/server/fedavg.py -d femnist -m res18-jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "FedAVG_FEMNIST_res18" --algo "FedAVG"
#python3 -u ./src/server/fedprox.py -d femnist -m res18 -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "FedProx_FEMNIST_res18" --algo "FedProx"
#python3 -u ./src/server/moon.py -d femnist -m res18 -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "moon_FEMNIST_res18" --algo "MOON"
#python3 -u ./src/server/fedsampling.py -d femnist -m res18 -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb 1 --run_name "fedsampling_FEMNIST_res18" --algo "FedSampling"




#python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 200 -le 5 --verbose_gap 1 --wandb True --run_name "fedDyS2 shuffle no pretrain th mean" --th_mode mean
#python3 -u ./src/server/fedavg.py -d cinic10 -m mobile -jr 0.1 -ge 200 -le 5 --verbose_gap 1 --wandb True --run_name "FedAVG no pretrain " --algo "FedAVG"
#python3 -u ./src/server/fedDyS2.py -d cinic10 -m mobile -jr 0.1 -ge 200 -le 5 --verbose_gap 1 --wandb True --run_name "fedDyS2 shuffle no pretrain th 40%" --th_mode percentage --th 0.4
#python3 -u ./src/server/fedprox.py -d cinic10 -m mobile -jr 0.1 -ge 200 -le 5 --verbose_gap 1 --wandb True --run_name "FedProx"
#python3 -u ./src/server/scaffold.py -d cinic10 -m mobile -jr 0.1 -ge 200 -le 5 --verbose_gap 1 --wandb True --run_name "scaffold local_lr 0.1" --local_lr 0.1
#python3 -u ./src/server/moon.py -d cinic10 -m mobile -jr 0.1 -ge 200 -le 5 --verbose_gap 1 --wandb True --run_name "moon"
#python3 -u ./src/server/fedsampling.py -d cinic10 -m mobile -jr 0.1 -ge 200 -le 5 --verbose_gap 1 --wandb True --run_name "fedsampling"

#python3 -u ./src/server/fedavgmDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedavgmDyS_entropy_score_2% noshuffle lr 0.001 lmomentum 0.9 frozen" --local_lr 0.001 --momentum 0.9

#python3 -u ./src/server/fedavgmDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedavgmDyS_entropy_score_2% noshuffle lr 0.001 lmomentum 0.9" --local_lr 0.001
#python3 -u ./src/server/fedavgmDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedavgmDyS_entropy_score_0% noshuffle lr 0.001 lmomentum 0.9" --local_lr 0.001 --momentum 0.9
#python3 -u ./src/server/fedaDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedDyS_entropy_score_2% noshuffle lr 0.001 lmomentum 0.9" --local_lr 0.001 --momentum 0.9
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedDyS_entropy_score_0% noshuffle lr 0.001 lmomentum 0.9" --local_lr 0.001 --momentum 0.9
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedDyS_entropy_score_2% noshuffle lr 0.001" --local_lr 0.001
#python3 -u ./src/server/fedavgm.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedAVGM noshuffle lr 0.001 lmomentum 0.9" --local_lr 0.001 --momentum 0.9
#python3 -u ./src/server/fedavgm.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedAVGM noshuffle lr 0.001 " --local_lr 0.001
#python3 -u ./src/server/fedavg.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedAVG noshuffle lr 0.001 lmomentum 0.9" --local_lr 0.001 --momentum 0.9

#python3 -u ./src/server/fedavgmDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedavgmDyS_entropy_score 1%REV noshuffle lr 0.001 lmomentum 0.9" --local_lr 0.001 --momentum 0.9
#python3 -u ./src/server/fedavgmDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedavgmDyS_entropy_score_1% noshuffle lr 0.001 lmomentum 0.9" --local_lr 0.001 --momentum 0.9
#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "FedDyS_entropy_score_0% =FedAVG noshuffle lr 0.001 lmomentum 0.45" --local_lr 0.001 --momentum 0.45
#python3 -u ./src/server/fedavgmDyS.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedavgmDyS_entropy_score_0% =FedAVG noshuffle lr 0.001 lmomentum 0.45" --local_lr 0.001 --momentum 0.45
#python3 -u ./src/server/fedavg.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedavg noshuffle lr 0.001" --local_lr 0.001
#python3 -u ./src/server/fedavg.py -d cinic10 -m mobile -jr 0.1 -ge 50 -le 5 --verbose_gap 1 --wandb True --run_name "fedavg noshuffle lr 0.001 lmomentum 0.9" --local_lr 0.001 --momentum 0.9
#python3 -u ./src/server/fedavg.py -d cinic10 -m mobile -jr 0.1 -ge 20 -le 5 --verbose_gap 1 --wandb True --run_name "FedAVG_retest"
#python3 -u ./src/server/fedavg.py -d cinic10 -m mobile -jr 0.1 -ge 25 -le 5 --verbose_gap 1 --algo FedAVG --wandb True --run_name "fedAVG"

#python3 -u ./src/server/fedDyS.py -d cinic10 -m lenet5 -jr 0.1 -ge 50 -le 1 --verbose_gap 1 --wandb True --run_name "FedDyS_lenet5_1e"
#python3 -u ./src/server/fedDyS.py -d cinic10 -m lenet5 -jr 0.1 -ge 50 -le 1 --verbose_gap 1 --wandb True --run_name "FedDyS_lenet5_1e"
#python3 -u ./src/server/fedavg.py -d cinic10 -m lenet5 -jr 0.1 -ge 50 -le 1 --verbose_gap 1 --wandb True --run_name "FedAVG_lenet5_1e"


#python3 -u ./src/server/fedDyS.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb True --run_name "fedDyS_AISTATS"
#python3 -u ./src/server/fedavgmDyS.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --wandb True --run_name "fedAvgMDyS_tanh_2.15_10perc_drop_global_testset" --global_testset 1

#python3 -u ./src/server/fedavg.py -d cinic10 -m mobile -jr 0.1 -ge 500 -le 5 --verbose_gap 1 --algo FedAVG --wandb True --run_name fedAVG

#python3 -u ./src/server/scaffold.py -d cinic10 -m mobile -jr 0.1 -ge 200 -le 5 --verbose_gap 1
#python3 -u ./src/server/fedavgm.py -d cinic10 -m mobile -jr 0.1 -ge 100 -le 1 --verbose_gap 1
#python3 -u ./src/server/fedavgmDyS.py -d cinic10 -m mobile -jr 0.1 -ge 100 -le 1 --verbose_gap 1
