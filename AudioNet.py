import torch
import torch.nn
import torch.nn.functional
import torch.optim

class NeuralNet(torch.nn.Module):
   def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv1 = torch.nn.Conv2d(1, 12, 3, stride = 1, padding = 0)
        self.bn1 = torch.nn.BatchNorm2d(12)
        self.conv2 = torch.nn.Conv2d(12, 24, 3, stride = 1, padding = 0)
        self.bn2 = torch.nn.BatchNorm2d(24)
        self.conv3 = torch.nn.Conv2d(24, 48, 3, stride = 1, padding = 0)
        self.bn3 = torch.nn.BatchNorm2d(48)
        self.fc1 = torch.nn.Linear(48 * 6 * 6, 128)
        self.fc2 = torch.nn.Linear(128, 8)
        self.adapt = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.drop = torch.nn.Dropout(0.5)
        
   def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.nn.functional.relu(self.bn3(self.conv3(x))))
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
