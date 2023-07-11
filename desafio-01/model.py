import torch.nn as nn
import torchvision.models as models

#Definindo um modelo Resnet customizado
class ResNet_pt(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet_pt, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Pre-trained ResNet-50
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x