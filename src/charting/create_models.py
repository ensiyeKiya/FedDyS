# Import necessary packages
import itertools
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from numpy import log2
from sklearn.manifold import TSNE
from torchsummary import summary
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn.functional as F

# Check for GPU availability
# from src.client.fedDyS import dynamic_sample_selection
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "mobilenetv2_cifar10.pth"

# Load CIFAR10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV2 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load the CIFAR10 training dataset
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split the dataset to get 1,000 samples
_, subset_train = random_split(cifar10_train, [len(cifar10_train) - 10000, 10000])

train_loader = DataLoader(subset_train, batch_size=256, shuffle=True)

def train_model(trail_loader):
    # Define the MobileNetV2 model
    model = models.mobilenet_v2(pretrained=False, num_classes=10)  # CIFAR10 has 10 classes
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model (simple training loop for demonstration)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

    # Evaluate the model (on the same 1,000 samples for simplicity)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    print(f"Accuracy: {100. * correct / total}%")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model


#train_model(train_loader)

# Create the model and then load the state dictionary
model = models.mobilenet_v2(pretrained=False, num_classes=10)
model.is_language_model = False
model.load_state_dict(torch.load(model_save_path))
model.to(device)

def get_indices_for_class(target_class, num_samples, labels):
    return [i for i, label in enumerate(labels) if label == target_class][:num_samples]

# Get the indices for the desired classes and samples
combined_indices = get_indices_for_class(0, 900, cifar10_train.targets)
# combined_indices.extend(get_indices_for_class(1, 100, cifar10_train.targets))
combined_indices.extend(get_indices_for_class(2, 85, cifar10_train.targets))
combined_indices.extend(get_indices_for_class(3, 10, cifar10_train.targets))
combined_indices.extend(get_indices_for_class(4, 5, cifar10_train.targets))

# Create the DataLoader using the combined indices
custom_dataset = Subset(cifar10_train, combined_indices)
custom_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Print the total number of samples in the custom DataLoader
print(f"Total samples in DataLoader: {len(custom_loader.dataset)}")
summary(model, (3, 224, 224))

tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=0)

def plot_t_sne(loader,model):
    def extract_model_outputs(loader, model):
        model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        return np.vstack(all_outputs), np.hstack(all_labels)

    outputs, labels = extract_model_outputs(loader, model)

    # Perform T-SNE on the outputs
    tsne_results = tsne.fit_transform(outputs)

    # Visualize the T-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='jet', alpha=0.5)
    plt.colorbar(scatter)
    # plt.title('T-SNE Visualization of Model Outputs on Custom Dataset')
    plt.title('')
    plt.show()



plot_t_sne(custom_loader, model)


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

def dynamic_sample_selection(model, trainloader, class_counts, client_idx, logger, batch_size,
                             rnd=-1, epoch=None, n_classes=10, device=torch.device("cpu")):
    model.eval()

    # Adjust class counts and ratios
    class_counts = torch.Tensor(class_counts).to(device)
    min_count = torch.max(0.1 * torch.sum(class_counts), torch.min(class_counts[class_counts > 0]))
    class_counts[class_counts < min_count] = min_count
    class_ratios = class_counts / min_count
    class_ratios /= torch.sum(class_ratios)
    class_ratios *= 5
    class_ratios += 1
    min_ratio = torch.min(class_ratios)

    # Gather data from trainloader
    data_items, labels = zip(*trainloader)
    data_items = list(itertools.chain.from_iterable(data_items))
    labels = list(itertools.chain.from_iterable(labels))
    dataset = CustomDataset(data_items, labels)
    # new_trainloader = DataLoader(dataset, batch_size=len(data_items), shuffle=False)
    new_trainloader = DataLoader(dataset, batch_size=100, shuffle=False)

    selected_inputs = []
    selected_targets = []

    if model.is_language_model:
        hidden_state = model.init_hidden(new_trainloader.batch_size)

    for data, target in new_trainloader:
        data, target = data.cuda(), target.cuda()

        # Compute logits and confidence
        if model.is_language_model:
            x = process_x(data)
            target = target.float()
            if x.size(0) < new_trainloader.batch_size:
                continue
            logits, hidden_state = model(x, hidden_state)
        else:
            logits = F.softmax(model(data), dim=1)

        probabilities, y_pred = logits.data.max(1, keepdim=True)
        correct_mask = torch.transpose(y_pred, 0, 1) == target
        correct_mask = correct_mask.squeeze()

        non_max_logits = logits.clone()
        rows = torch.arange(y_pred.size(0))
        non_max_logits[rows, y_pred.squeeze()] = -1
        next_max, _ = non_max_logits.data.max(1, keepdim=True)

        # Compute deltas
        confidences = probabilities.squeeze() - next_max.squeeze()
        ratios = class_ratios[target].clone() / min_ratio
        deltas = confidences * ratios
        deltas_fl1 = torch.clamp(deltas, min=0)
        deltas_fl2 = deltas_fl1 * correct_mask.float()
        # deltas2 = torch.pow(1.5, -1 * deltas_fl2)
        deltas2 = torch.exp(-1 * deltas_fl2)

        # Selection logic
        non_errors = deltas2[deltas2 < 0.9999999]
        if non_errors.numel() < 200 or non_errors.std() < 0.01:
            mu_mean = 0.5
            mu_std = 0.5
        else:
            mu_mean = non_errors.mean() - 1.5 * non_errors.std()
            mu_std = max(non_errors.std() / 2, 0.001)
        gaussian_random_mu = mu_mean + mu_std * torch.randn(deltas2.size(0)).cuda()

        diff = deltas2 - gaussian_random_mu
        selected_indices = diff.gt(0).nonzero().flatten()

        # Log and gather selected samples
        selected_inputs.extend(data[selected_indices])
        selected_targets.extend(target[selected_indices])
        selected_targets_num = [t.item() for t in selected_targets]
        labels_num = [t.item() for t in labels]
        print(f"igec: items:{Counter(labels_num).items()}->{Counter(selected_targets_num).items()}")
        print(f'ige: r{rnd} c{client_idx} e{epoch}'
                   f' l:{logits.mean().item():.2f}\'{logits.var().item():.2f}'
                   f' c:{confidences.mean().item():.2f}\'{confidences.var().item():.2f}'
                   f' d:{deltas.mean().item():.2f}\'{deltas.var().item():.2f}'
                   f' dfl1:{deltas_fl1.mean().item():.2f}\'{deltas_fl1.var().item():.2f}'
                   f' dfl2:{deltas_fl2.mean().item():.2f}\'{deltas_fl2.var().item():.2f}'
                   f' d2:{deltas2.mean().item():.2f}\'{deltas2.var().item():.2f}'
                   f' mu:{gaussian_random_mu.mean().item():.2f}\'{gaussian_random_mu.var().item():.2f}'
                   f' df:{diff.mean().item():.2f}\'{diff.var().item():.2f}'
                   f' acc:{100*torch.sum(correct_mask).item()/len(correct_mask):.0f}%'
                   f' tot:{len(dataset)}->{len(selected_targets)}, '
                   f' rem:{len(dataset) - len(selected_targets)}, '
                   f' ent:{entropy_balance(labels_num, n_classes):.3f}->{entropy_balance(selected_targets_num, n_classes):.3f}'
                   )
        del logits, confidences, deltas, deltas_fl1, deltas_fl2, deltas2, gaussian_random_mu, diff, non_errors, \
            selected_targets_num, labels_num,

    print(f'info_gain: r{rnd}, c{client_idx} - {len(selected_inputs)} sample(s)')
    print(f"selected_targets:{len(selected_targets)}")
    dataset = CustomDataset(selected_inputs, selected_targets)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


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


targets = []
for batch_idx, (_, target) in enumerate(custom_loader):
    targets.extend(target.tolist())
class_counts = [0] * 10
for (k, v) in Counter(targets).items():
    class_counts[k] = v

dyn_loader = dynamic_sample_selection(model, custom_loader, class_counts, -1,
                                      None, batch_size=50, rnd=-1,
                                      epoch=-1, n_classes=10, device=device)

plot_t_sne(dyn_loader, model)

targets = []
for batch_idx, (_, target) in enumerate(dyn_loader):
    targets.extend(target.tolist())
class_counts2 = [0] * 10
for (k, v) in Counter(targets).items():
    class_counts2[k] = v
print(class_counts)
print(class_counts2)
print(eb([900, 85, 10, 5, 0, 0,0,0,0,0], 10))
print(eb([485, 59, 9, 3, 0, 0,0,0,0,0], 10))
