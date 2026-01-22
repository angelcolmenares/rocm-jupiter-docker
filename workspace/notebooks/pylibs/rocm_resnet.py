# workspace/notebooks/pylibs/rocm_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ROCmBasicBlock(nn.Module):
    """Basic block with GroupNorm instead of BatchNorm for ROCm compatibility"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, 1, stride, bias=False),
                nn.GroupNorm(groups, self.expansion * out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out

class ROCmResNet(nn.Module):
    """ResNet variant with GroupNorm for ROCm compatibility"""
    def __init__(self, block, num_blocks, num_classes=10, groups=8):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(groups, 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, groups=groups)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride, groups):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, groups))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# FastAI compatible functions
def rocm_resnet18(pretrained=False, num_classes=1000, **kwargs):
    """
    Constructs a ROCm-compatible ResNet-18 model.
    
    Args:
        pretrained: If True, returns a model with pretrained weights (not implemented yet)
        num_classes: Number of output classes
        **kwargs: Additional arguments (ignored)
    """
    model = ROCmResNet(ROCmBasicBlock, [2, 2, 2, 2], num_classes)
    
    # Note: Currently no pretrained weights available for ROCm models
    # You could add weight loading logic here if you have pretrained weights
    if pretrained:
        print("Warning: Pretrained weights not available for ROCm ResNet18")
        print("Training from scratch...")
    
    return model

def rocm_resnet50(pretrained=False, num_classes=1000, **kwargs):
    """
    Constructs a ROCm-compatible ResNet-50 model.
    
    Args:
        pretrained: If True, returns a model with pretrained weights (not implemented yet)
        num_classes: Number of output classes
        **kwargs: Additional arguments (ignored)
    """
    model = ROCmResNet(ROCmBasicBlock, [3, 4, 6, 3], num_classes)
    
    if pretrained:
        print("Warning: Pretrained weights not available for ROCm ResNet50")
        print("Training from scratch...")
    
    return model

# Alternative: Functions without pretrained parameter (for backward compatibility)
def rocm_resnet18_basic(num_classes=10):
    """Basic version without pretrained parameter"""
    return ROCmResNet(ROCmBasicBlock, [2, 2, 2, 2], num_classes)

def rocm_resnet50_basic(num_classes=10):
    """Basic version without pretrained parameter"""
    return ROCmResNet(ROCmBasicBlock, [3, 4, 6, 3], num_classes)