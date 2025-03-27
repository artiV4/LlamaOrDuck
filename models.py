import numpy as np
import torch 
import torchvision as tv

def get_densenet121():
    densenet121 = tv.models.densenet121(pretrained=False)
    num_ftrs = densenet121.classifier.in_features
    densenet121.classifier = torch.nn.Linear(num_ftrs, 2)
    return "densenet121_noGaussBlur", densenet121



class RGB_CNN1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 256, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.dropout = torch.nn.Dropout(0.5)

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((5,5)),
            torch.nn.Flatten(),
            torch.nn.Linear(256*5*5, 6144),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(6144, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.dropout(x) # no dropout for test data
        x = self.classifier(x)
        return x
    

def get_basicCNN():
    return "BasicCNN", RGB_CNN1()


class AlexNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 2),
            # torch.nn.Linear(4096, 512), # AlexNet2 differs here
            # torch.nn.Linear(512, 64), # and here
            # torch.nn.Linear(64, 2), # and here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def get_AlexNet():
    return "AlexNet_224_GaussBlur", AlexNet()