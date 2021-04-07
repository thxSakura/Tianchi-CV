import torch.nn as nn
from torchvision.models import resnet18, resnet50

OUT_FEATURES = 1000

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.resnet = resnet18(pretrained=True)
        
#         for p in self.parameters():		# 固定权重
#             p.requires_grad = False
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
#             nn.ReLU(),
#             nn.MaxPool2d(3, stride=2),
#         )
#         self.liner1 = nn.Linear(23328, OUT_FEATURES)
#         self.relu = nn.ReLU()
        self.liner21 = nn.Linear(OUT_FEATURES, 11)
        self.liner22 = nn.Linear(OUT_FEATURES, 11)
        self.liner23 = nn.Linear(OUT_FEATURES, 11)
        self.liner24 = nn.Linear(OUT_FEATURES, 11)
        self.liner25 = nn.Linear(OUT_FEATURES, 11)

    def forward(self, data):
        x = self.resnet(data)
        x = x.view(x.shape[0], -1)

#         x = self.liner1(x)
#         x = self.relu(x)
        c1 = self.liner21(x)
        c2 = self.liner22(x)
        c3 = self.liner23(x)
        c4 = self.liner24(x)
        c5 = self.liner25(x)
        return c1, c2, c3, c4, c5
