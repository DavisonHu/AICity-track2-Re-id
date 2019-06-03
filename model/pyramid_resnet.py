from opt import opt
import torch
import torch.nn as nn
from model.resnet import resnet152


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        resnet = resnet152(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )

        self.conv11 = resnet.layer_1
        self.conv22 = resnet.layer_2
        self.conv33 = resnet.layer_3

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(512, 512, 32),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True, inplace=0.2)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(512, 512, 16),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True, inplace=0.2)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(512, 512, 8),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True, inplace=0.2)
        )

        self.bottleneck_11 = nn.BatchNorm1d(512)
        self.bottleneck_11.bias.requires_grad_(False)
        self.classifier_11 = nn.Linear(in_features=512, out_features=opt.class_num, bias=False)
        self.color_11 = nn.Linear(in_features=512, out_features=10, bias=False)

        self.bottleneck_22 = nn.BatchNorm1d(512)
        self.bottleneck_22.bias.requires_grad_(False)
        self.classifier_22 = nn.Linear(in_features=512, out_features=opt.class_num, bias=False)
        self.color_22 = nn.Linear(in_features=512, out_features=10, bias=False)

        self.bottleneck_33 = nn.BatchNorm1d(512)
        self.bottleneck_33.bias.requires_grad_(False)
        self.classifier_33 = nn.Linear(in_features=512, out_features=opt.class_num, bias=False)
        self.color_33 = nn.Linear(in_features=512, out_features=10, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv11(x)
        feature_1 = self.conv1_1(x).view(x.shape[0], -1)
        x = self.conv22(x)
        feature_2 = self.conv2_2(x).view(x.shape[0], -1)
        x = self.conv33(x)
        feature_3 = self.conv3_3(x).view(x.shape[0], -1)

        class_1 = self.classifier_11(self.bottleneck_11(feature_1))
        color_1 = self.color_11(self.bottleneck_11(feature_1))
        class_2 = self.classifier_22(self.bottleneck_22(feature_2))
        color_2 = self.color_22(self.bottleneck_22(feature_2))
        class_3 = self.classifier_33(self.bottleneck_33(feature_3))
        color_3 = self.color_33(self.bottleneck_33(feature_3))

        feature = torch.cat([feature_1, feature_2, feature_3], dim=1)

        return feature, feature_1, feature_2, feature_3, class_1, class_2, class_3, color_1, color_2, color_3
