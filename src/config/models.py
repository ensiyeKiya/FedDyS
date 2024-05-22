import json
from collections import OrderedDict
from typing import Dict, List, Optional, Type
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .utils import PROJECT_DIR, get_best_device


class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: List[nn.Module] = []
        self.is_language_model = False
        self.is_binary_classification = False

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.clone().detach())

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def check_avaliability(self):
        if not self.is_language_model:
            if self.base is None or self.classifier is None:
                raise RuntimeError(
                    "You need to re-write the base and classifier in your custom model class."
                )
            self.dropout = [
                module
                for module in list(self.base.modules()) + list(self.classifier.modules())
                if isinstance(module, nn.Dropout)
            ]

    def forward(self, x: torch.Tensor):
        return self.classifier(F.relu(self.base(x)))

    def get_final_features(self, x: torch.Tensor, detach=True) -> torch.Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: torch.Tensor) -> Optional[List[torch.Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


# CNN used in FedAvg
class FedAvgCNN(DecoupledModel):
    def __init__(self, dataset: str):
        super(FedAvgCNN, self).__init__()
        config = {
            "mnist": (1, 1024, 10),
            "medmnistS": (1, 1024, 11),
            "medmnistC": (1, 1024, 11),
            "medmnistA": (1, 1024, 11),
            "covid19": (3, 196736, 4),
            "fmnist": (1, 1024, 10),
            "emnist": (1, 1024, 62),
            "femnist": (1, 1, 62),
            "cifar10": (3, 1600, 10),
            "cinic10": (3, 1600, 10),
            "cifar100": (3, 1600, 100),
            "tiny_imagenet": (3, 3200, 200),
            "celeba": (3, 133824, 2),
            "svhn": (3, 1600, 10),
            "usps": (1, 800, 10),
            "domain": infer(dataset, "avgcnn"),
            "imdb": (1, 80, 1),
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(config[dataset][1], 512),
            )
        )
        self.classifier = nn.Linear(512, config[dataset][2])


class LeNet5(DecoupledModel):
    def __init__(self, dataset: str) -> None:
        super(LeNet5, self).__init__()
        config = {
            "mnist": (1, 256, 10),
            "medmnistS": (1, 256, 11),
            "medmnistC": (1, 256, 11),
            "medmnistA": (1, 256, 11),
            "covid19": (3, 49184, 4),
            "fmnist": (1, 256, 10),
            "emnist": (1, 256, 62),
            "femnist": (1, 256, 62),
            "cifar10": (3, 400, 10),
            "cinic10": (3, 400, 10),
            "svhn": (3, 400, 10),
            "cifar100": (3, 400, 100),
            "celeba": (3, 33456, 2),
            "usps": (1, 200, 10),
            "tiny_imagenet": (3, 2704, 200),
            "domain": infer(dataset, "lenet5"),
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(config[dataset][1], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
            )
        )

        self.classifier = nn.Linear(84, config[dataset][2])


class TwoNN(DecoupledModel):
    def __init__(self, dataset):
        super(TwoNN, self).__init__()
        config = {
            "mnist": (784, 10),
            "medmnistS": (784, 11),
            "medmnistC": (784, 11),
            "medmnistA": (784, 11),
            "fmnist": (784, 10),
            "emnist": (784, 62),
            "femnist": (784, 62),
            "cifar10": (3072, 10),
            "cinic10": (3072, 10),
            "svhn": (3072, 10),
            "cifar100": (3072, 100),
            "usps": (1536, 10),
            "synthetic": (60, 10),  # default dimension and classes
        }

        self.base = nn.Linear(config[dataset][0], 200)
        self.classifier = nn.Linear(200, config[dataset][1])
        self.activation = nn.ReLU()

    def need_all_features(self):
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.base(x))
        x = self.classifier(x)
        return x

    def get_final_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        x = torch.flatten(x, start_dim=1)
        x = self.base(x)
        return func(x)

    def get_all_features(self, x):
        raise RuntimeError("2NN has 0 Conv layer, so is unable to get all features.")


class MobileNetV2(DecoupledModel):
    def __init__(self, dataset):
        super(MobileNetV2, self).__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cinic10": 10,
            "cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "mobile"),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.classifier = nn.Linear(
            self.base.classifier[1].in_features, config[dataset]
        )

        #freeze the base model
        for param in self.base.parameters():
            param.requires_grad = False

        self.base.classifier[1] = nn.Identity()


class ResNet18(DecoupledModel):
    def __init__(self, dataset):
        super(ResNet18, self).__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cinic10": 10,
            "cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "res18"),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        #freeze the base model
        for param in self.base.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.base.fc.in_features, config[dataset])
        self.base.fc = nn.Identity()

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().forward(x)

    def get_all_features(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)


class AlexNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        config = {
            "covid19": 4,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "alex"),
        }
        if dataset not in config.keys():
            raise NotImplementedError(f"AlexNet does not support dataset {dataset}")

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(
            self.base.classifier[-1].in_features, config[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


class SqueezeNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cinic10": 10,
            "cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "sqz"),
        }

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        sqz = models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        )
        self.base = sqz.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(sqz.classifier[1].in_channels, config[dataset], kernel_size=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return self.classifier(self.base(x))

    def get_all_features(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)


# Some dirty codes for adapting DomainNet
def infer(dataset, model_type):
    if dataset == "domain":
        with open(PROJECT_DIR / "data" / "domain" / "metadata.json", "r") as f:
            metadata = json.load(f)
        class_num = metadata["class_num"]
        img_size = metadata["image_size"]
        coef = {"avgcnn": 50, "lenet5": 42.25}
        if model_type in ["alex", "res18", "sqz", "mobile"]:
            return class_num
        return (3, int(coef[model_type] * img_size), class_num)




#####################
### Language Models
#####################





import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SentimentRNN(DecoupledModel):
    def __init__(self, dataset):  # no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.3):
        super(SentimentRNN, self).__init__()

        self.is_language_model = True
        self.is_binary_classification = True

        no_layers = 2
        vocab_size = 80  # len(vocab) + 1 #extra 1 for padding
        embedding_dim = 8
        output_dim = 1
        hidden_dim = 32 #256
        drop_prob = 0.2

        self.device = get_best_device(True)

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)


        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()



    def forward(self, x, hidden):
        batch_size = x.size(0)
        # if batch_size < hidden[0].size(1):
        #     hidden = (hidden[0][:, :batch_size, :], hidden[1][:, :batch_size, :])
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden



    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(self.device)
        hidden = (h0, c0)
        return hidden


class RNNModelContainer(DecoupledModel):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    Modified by: Hongyi Wang from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self, rnn_type, ntoken, ninp_l1, nhid_l1, ninp_l2, nhid_l2, dropout=0.5, tie_weights=False):
        super(RNNModelContainer, self).__init__()
        #self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp_l1)

        self.rnn1 = nn.LSTM(ninp_l1, nhid_l1, 1)

        assert ninp_l2 == nhid_l1  # these two shapes are designed to be equal
        self.rnn2 = nn.LSTM(ninp_l2, nhid_l2, 1)

        self.decoder = nn.Linear(nhid_l2, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid_l2 != ninp_l2:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid_l1 = nhid_l1
        self.nhid_l2 = nhid_l2

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        #emb = self.drop(self.encoder(input))
        emb = self.encoder(input)
        output1, hidden1 = self.rnn1(emb, hidden[0])

        output2, hidden2 = self.rnn2(output1, hidden[1])
        #output = self.drop(output)

        #decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = self.decoder(output2[-1,:,:])

        return decoded.t(), (hidden1, hidden2)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return ((weight.new_zeros(1, bsz, self.nhid_l1),
                 weight.new_zeros(1, bsz, self.nhid_l1)),
                (weight.new_zeros(1, bsz, self.nhid_l2),
                 weight.new_zeros(1, bsz, self.nhid_l2)))




MODEL_DICT: Dict[str, Type[DecoupledModel]] = {
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "2nn": TwoNN,
    "mobile": MobileNetV2,
    "res18": ResNet18,
    "alex": AlexNet,
    "sqz": SqueezeNet,
    "rnn": SentimentRNN
}

