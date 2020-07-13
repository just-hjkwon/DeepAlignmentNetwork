import torch
import torch.nn as nn


class VGGBasedModel(nn.Module):
    def __init__(self, in_channels):
        super(VGGBasedModel, self).__init__()

        self.conv1a = ConvolutionBatchNormReLU(in_channels=in_channels, out_channels=64, kernel_size=3)
        self.conv1b = ConvolutionBatchNormReLU(in_channels=64, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = ConvolutionBatchNormReLU(in_channels=64, out_channels=128, kernel_size=3)
        self.conv2b = ConvolutionBatchNormReLU(in_channels=128, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = ConvolutionBatchNormReLU(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3b = ConvolutionBatchNormReLU(in_channels=256, out_channels=256, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4a = ConvolutionBatchNormReLU(in_channels=256, out_channels=512, kernel_size=3)
        self.conv4b = ConvolutionBatchNormReLU(in_channels=512, out_channels=512, kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, padding=0)
        self.fc2 = nn.Linear(in_features=256, out_features=136)

    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        x = self.dropout(x)

        x = self.fc1(x)
        x = x.squeeze()
        x = self.fc2(x)

        return x


class ConvolutionBatchNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvolutionBatchNormReLU, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConnectionLayers(nn.Module):
    def __init__(self):
        super(ConnectionLayers, self).__init__()

        self.fc = nn.Linear(in_feature=512, out_features=3136)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(size=(112, 112))

    def forward(self, image, previous_landmark, current_landmark, middle_feature_vector):
        transform_matrix = self.estimate_transform_matrix(previous_landmark, current_landmark)
        transformed_landmark = self.transform_landmark(current_landmark, transform_matrix)
        transformed_image = self.transform_image(image, transform_matrix)
        heatmap = self.generate_heatmap(image.shape, transformed_landmark)

        x = self.fc(middle_feature_vector)
        x = self.relu(x)
        x = torch.reshape(x, (-1, 56, 56, 1))
        x = self.upsample(x)

        return transformed_image, heatmap, x

    @staticmethod
    def estimate_transform_matrix(previous_landmark, current_landmark):
        matrix = None
        return matrix

    @staticmethod
    def transform_image(image, transform_matrix):
        transformed_image = None
        return image

    @staticmethod
    def transform_landmark(current_landmark, transform_matrix):
        transformed_landmark = None
        return transformed_landmark

    @staticmethod
    def generate_heatmap(image_size, landmark):
        heatmap_image = None
        return heatmap_image