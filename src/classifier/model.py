import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import timm

class ResNet50_TL(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet50_TL, self).__init__()
        if pretrained:
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) # Pretrained model
        else:
            self.model = models.resnet50(weights=None)  # Non-pretrained model (random weights)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class DenseNet121_TL(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DenseNet121_TL, self).__init__()
        self.model = models.densenet121(pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class EfficientNetB0_TL(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB0_TL, self).__init__()
        self.model = models.efficientnet_b0(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class RegNet_TL(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(RegNet_TL, self).__init__()
        self.model = models.regnet_y_400mf(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class Xception_TL(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Xception_TL, self).__init__()
        self.model = timm.create_model("xception", pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class ConvNeXT_TL(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ConvNeXT_TL, self).__init__()
        self.model = models.convnext_base(pretrained=pretrained)
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.3):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)  # Dropout après la première activation

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)  # Dropout après la deuxième activation

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)  # Appliquer Dropout après ReLU

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)  # Appliquer Dropout après ReLU

        out += self.shortcut(x)
        return F.relu(out)  # Activation finale

class CustomResNet(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.3):
        super(CustomResNet, self).__init__()
        self.layer1 = BasicBlock(3, 64, dropout_prob=dropout_prob)
        self.layer2 = BasicBlock(64, 128, stride=2, dropout_prob=dropout_prob)
        self.layer3 = BasicBlock(128, 256, stride=2, dropout_prob=dropout_prob)
        self.layer4 = BasicBlock(256, 512, stride=2, dropout_prob=dropout_prob)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout_fc = nn.Dropout(p=dropout_prob)  # Dropout sur la couche fully connected
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)  # Appliquer Dropout avant la couche fully connected
        return self.fc(x)

def main() -> None:
    num_classes = 5
    
    # Pretrained model
    model_pretrained = ResNet50_TL(num_classes, pretrained=True)
    sample_input = torch.randn(1, 3, 256, 256)
    output_pretrained = model_pretrained(sample_input)
    print(f"Output of pretrained model: {output_pretrained.shape}")

    # Non-pretrained model
    model_random = ResNet50_TL(num_classes, pretrained=False)
    output_random = model_random(sample_input)
    print(f"Output of non-pretrained model: {output_random.shape}")
    

if __name__ == "__main__":
    print("To train please got to ~/pfe/ and run \'python main.py model\'")
