import torch
from torch import nn
import torch.nn.functional as F
import os
import torch.optim as optim
from data_loading import get_data

from ray import tune
import ray

import pickle
from ray import train
import tempfile
from ray.train import Checkpoint
import pickle
from pathlib import Path

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

class CNNModel(nn.Module):
    def __init__(self, l1=120, l2=60):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxPool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*9*9, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 15)

    def forward(self, x):
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def set_training_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def save_model(cnn, path):
    torch.save(cnn.state_dict(), path)

def save_entire_model(model, path):
    model_scripted = torch.jit.script(model)
    model_scripted.save(path)

def train_model(config):
    dataset_path = "./vegetables/vegetable_images"
    dataset_type = "/train"
    train_loader, _ = get_data(dataset_path, dataset_type)
    cnn = CNNModel(l1=config["hidden_layer_size_1"], l2=config["hidden_layer_size_2"])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=config["lr"])
    nb_epochs = 5
    running_loss = 0.0
    for epoch in range(nb_epochs):
        for i, data in enumerate(train_loader):
            image_batch, labels = data
            image_batch, labels = data[0], data[1]
            optimizer.zero_grad()
            res = cnn.forward(image_batch)
            l = loss_fn(res, labels)
            l.backward()
            optimizer.step()
            running_loss += l.item()
            if i % 5 == 0:
                print("Epoch {}, Iter {} : loss {}".format(epoch, i, running_loss))
                running_loss = 0.0


    checkpoint_data = {
        "epoch": epoch,
        "net_state_dict": cnn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(checkpoint_data, fp)
        
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report({"loss": running_loss}, checkpoint=checkpoint)
    
    return {"loss" :running_loss}

def dummy_test_fc(config):
    print("test")
    return  {"lr" :config["lr"]}
    


if __name__ == "__main__":
    ray.init(num_cpus=2)
    config = {
        "l1": tune.choice([120, 240]),
        "l2": tune.choice([30,60]),
        "batch_size": tune.choice([8,16,32]),
        "lr": tune.loguniform(1e-4, 1e-1)
    }
    
    train_model = tune.with_resources(train_model, {"cpu": 2})
    analysis = tune.run(train_model, config=config, num_samples=10)
    best_trial = analysis.get_best_trial("loss", "min", "last")
    print("Configuration for best trial: {}".format(best_trial.config))
    print("Loss for best trial: {}".format(best_trial.last_result["loss"]))

