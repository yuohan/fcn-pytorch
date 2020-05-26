import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):

    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        if pretrained:
            features = [self.conv1_1, self.relu1_1, self.conv1_2, self.relu1_2, self.pool1,
                        self.conv2_1, self.relu2_1, self.conv2_2, self.relu2_2, self.pool2,
                        self.conv3_1, self.relu3_1, self.conv3_2, self.relu3_2, self.conv3_3, self.relu3_3, self.pool3,
                        self.conv4_1, self.relu4_1, self.conv4_2, self.relu4_2, self.conv4_3, self.relu4_3, self.pool4,
                        self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2, self.conv5_3, self.relu5_3, self.pool5]
            vgg16 = models.vgg16(pretrained=True)

            for layer1, layer2 in zip(features, vgg16.features):
                if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                    layer1.weight.data.copy_(layer2.weight.data)
                    layer1.bias.data.copy_(layer2.bias.data)

            features = [self.fc6, self.relu6, self.drop6,
                        self.fc7, self.relu7, self.drop7]
            for layer1, layer2 in zip(features, vgg16.classifier[:-1]):
                if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Linear):
                    layer1.weight.data.copy_(layer2.weight.data.view(layer1.weight.size()))
                    layer1.bias.data.copy_(layer2.bias.data.view(layer1.bias.size()))

    def forward(self, x):

        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)
        pool3 = x

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.pool4(x)
        pool4 = x

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.pool5(x)
        
        x = self.relu6(self.fc6(x))
        x = self.drop6(x)

        x = self.relu7(self.fc7(x))
        x = self.drop7(x)

        return pool3, pool4, x


# from https://github.com/shelhamer/fcn.berkeleyvision.org
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN8(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super(FCN8, self).__init__()

        self.num_classes = num_classes

        # base network
        self.vgg16 = VGG16(pretrained=pretrained)
        # conv2d score layers
        self.score_fc7 = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        # conv_transpose2d upsampling layers
        self.upscore2x_1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore2x_2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # initialize conv2d layer
        for m in [self.score_fc7, self.score_pool4, self.score_pool3]:
            m.weight.data.zero_()
            if m.bias is not None:
                m.bias.data.zero_()
        # initialize conv_transpose2d layer
        for m in [self.upscore2x_1, self.upscore2x_2, self.upscore8x]:
            weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
            m.weight.data.copy_(weight)

    def crop(self, target, mask):
        return target[:, :, :mask.size()[2], :mask.size()[3]]

    def forward(self, x):

        pool3, pool4, fc7 = self.vgg16(x)

        h = self.score_fc7(fc7)
        upscore_fc7 = self.upscore2x_1(h)

        score_pool4 = self.crop(self.score_pool4(pool4*0.01), upscore_fc7)
        h = upscore_fc7 + score_pool4
        upscore_pool4 = self.upscore2x_2(h)

        score_pool3 = self.crop(self.score_pool3(pool3*0.001), upscore_pool4)
        h = upscore_pool4 + score_pool3
        upscore_pool3 = self.upscore8x(h)

        return self.crop(upscore_pool3, x)

    def save(self, save_path):

        state = {
            'state_dict': self.state_dict(),
            'num_classes': self.num_classes
        }
        torch.save(state, save_path)

    @classmethod
    def load(cls, model_path, device):

        state = torch.load(model_path, map_location=device)
        model = cls(state['num_classes'], pretrained=False).to(device)
        model.load_state_dict(state['state_dict'])
        return model