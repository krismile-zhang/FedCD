import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LIGHT(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_LIGHT, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2,  padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(128, 256, 5, stride=1, padding=2)
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, self.num_flat_features(x))
        features = F.relu(self.fc1(x))
        x = F.relu(self.fc2(features))
        y = F.relu(self.fc3(x))
        return features, y

    def get_logit(self, features):
        x = F.relu(self.fc2(features))
        y = F.relu(self.fc3(x))
        return y
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2,  padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2,  padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2,  padding=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        features = F.relu(self.fc1(x))
        y = F.relu(self.fc2(features))
        return features, y

    def get_logit(self, features):
        y = F.relu(self.fc2(features))
        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


