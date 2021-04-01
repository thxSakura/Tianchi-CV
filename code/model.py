import torch.nn as nn

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        self.liner1 = nn.Linear(3360, 1024)
        self.relu = nn.ReLU()
        self.liner21 = nn.Linear(1024, 11)
        self.liner22 = nn.Linear(1024, 11)
        self.liner23 = nn.Linear(1024, 11)
        self.liner24 = nn.Linear(1024, 11)
        self.liner25 = nn.Linear(1024, 11)
        self.liner26 = nn.Linear(1024, 11)
    def forward(self, data):
        x = self.cnn(data)
        x = x.view(x.shape[0], -1)
        x = self.liner1(x)
        x = self.relu(x)
        c1 = self.liner21(x)
        c2 = self.liner22(x)
        c3 = self.liner23(x)
        c4 = self.liner24(x)
        c5 = self.liner25(x)
        c6 = self.liner26(x)
        return c1, c2, c3, c4, c5, c6