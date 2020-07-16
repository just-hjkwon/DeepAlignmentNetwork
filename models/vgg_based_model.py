import numpy as np

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

        self.fc1_feature = None

        self.previous_landmarks = None
        self.transform_matrices = None
        self.untransform_matrices = None

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
        self.fc1_feature = x

        x = x.squeeze()
        x = self.fc2(x)

        return x

    def loss(self, prediction, target):
        landmark = target[0].view(-1, 68, 2)
        predicted_landmark_deltas = prediction.view(-1, 68, 2)

        predicted_landmarks = []

        for (prev_landmark, transform_matrix, untransform_matrix, predicted_landmark_delta) in zip(
                self.previous_landmarks,
                self.transform_matrices,
                self.untransform_matrices,
                predicted_landmark_deltas):
            transform_matrix = torch.tensor(transform_matrix, device=prediction.device)
            untransform_matrix = torch.tensor(untransform_matrix, device=prediction.device)

            prev_landmark = torch.tensor(prev_landmark, device=prediction.device)
            prev_landmark = torch.cat(
                [prev_landmark, torch.ones((68, 1), dtype=torch.float32, device=prediction.device)],
                axis=1)

            predicted_landmark = torch.matmul(transform_matrix, prev_landmark.T).T + predicted_landmark_delta
            predicted_landmark = torch.cat(
                [predicted_landmark, torch.ones((68, 1), dtype=torch.float32, device=prediction.device)], axis=1)

            predicted_landmark = torch.matmul(untransform_matrix, predicted_landmark.T).T

            predicted_landmarks.append(predicted_landmark)

        pupil_distance = target[1]

        predicted_landmarks = torch.stack(predicted_landmarks, axis=0)

        loss = nn.MSELoss()(predicted_landmarks / pupil_distance.view(-1, 1, 1),
                            landmark / pupil_distance.view(-1, 1, 1))

        return loss

    def untransform_landmark(self, landmark_delta, index):
        landmark_delta = landmark_delta.reshape(68, 2)

        previous_landmark = self.previous_landmarks[index]
        transform_matrix = self.transform_matrices[index]
        untransform_matrix = self.untransform_matrices[index]

        previous_landmark = np.hstack([previous_landmark, np.ones((previous_landmark.shape[0], 1))]).T
        landmark = np.matmul(transform_matrix, previous_landmark).T + landmark_delta

        landmark = np.hstack([landmark, np.ones((landmark.shape[0], 1))]).T
        landmark = np.matmul(untransform_matrix, landmark).T

        return landmark

    def set_previous_prediction_prior(self, previous_landmarks, transform_matrices, untransform_matrices):
        self.previous_landmarks = previous_landmarks
        self.transform_matrices = transform_matrices
        self.untransform_matrices = untransform_matrices


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
