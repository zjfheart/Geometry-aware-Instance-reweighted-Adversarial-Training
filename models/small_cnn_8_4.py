from collections import OrderedDict
import torch.nn as nn


class SmallCNN_8_4(nn.Module):
    def __init__(self, drop=0):
        super(SmallCNN_8_4, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 2)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 2)),
            ('relu2', activ),
            ('conv3', nn.Conv2d(32, 32, 2)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(32, 32, 2)),
            ('relu4', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv5', nn.Conv2d(32, 64, 2)),
            ('relu5', activ),
            ('conv6', nn.Conv2d(64, 64, 2)),
            ('relu6', activ),
            ('conv7', nn.Conv2d(64, 64, 2)),
            ('relu7', activ),
            ('conv8', nn.Conv2d(64, 64, 2)),
            ('relu8', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, 100)),
            ('relu3', activ),
            ('fc4', nn.Linear(100, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc4.weight, 0)
        nn.init.constant_(self.classifier.fc4.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits