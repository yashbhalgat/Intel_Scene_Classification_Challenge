import torch
import torch.nn as nn
from torchvision import models
import pdb

NUM_CLASSES = 6

class Quadrant(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Quadrant, self).__init__()
        model1 = models.densenet161(pretrained=True, num_classes=1000)
        self.model1 = nn.Sequential(*list(model1.children())[:-1])

        model2 = models.densenet161(pretrained=True, num_classes=1000)
        self.model2 = nn.Sequential(*list(model2.children())[:-1])

        model3 = models.densenet161(pretrained=True, num_classes=1000)
        self.model3 = nn.Sequential(*list(model3.children())[:-1])

        model4 = models.densenet161(pretrained=True, num_classes=1000)
        self.model4 = nn.Sequential(*list(model4.children())[:-1])

        num_ftrs = model1.classifier.in_features + \
                   model2.classifier.in_features + \
                   model3.classifier.in_features + \
                   model4.classifier.in_features
        
        del model1
        del model2
        del model3
        del model4

        self.avgpool = nn.AvgPool2d(3)

        self.classifier = nn.Linear(num_ftrs, NUM_CLASSES)

    def forward(self, p1, p2, p3, p4):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        f1 = self.model1(p1)
        f2 = self.model1(p2)
        f3 = self.model1(p3)
        f4 = self.model1(p4)
        feat = torch.cat((f1, f2, f3, f4),1)
        
        pooled = self.avgpool(feat).squeeze()

        out = self.classifier(pooled)

        return out
