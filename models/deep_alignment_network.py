import torch
import torch.nn as nn
from torchvision import transforms

import cv2
import numpy as np

from skimage.transform import SimilarityTransform

from .vgg_based_model import VGGBasedModel


class DeepAlignmentNetwork(nn.Module):
    def __init__(self, average_landmark, stage_count, end_to_end=False):
        super(DeepAlignmentNetwork, self).__init__()

        self.average_landmark = average_landmark.astype(dtype=np.float32)
        self.stage_count = stage_count
        self.models = torch.nn.Sequential()

        for i in range(stage_count):
            if i == 0:
                model = VGGBasedModel(in_channels=1)
                self.models.add_module("VGGBasedModel %d" % i, model)
            else:
                model = VGGBasedModel(in_channels=3)
                self.models.add_module("VGGBasedModel %d" % i, model)

        self.connection_layer = ConnectionLayers(self.average_landmark)

        if end_to_end is False:
            for i in range(stage_count - 1):
                for param in self.models[i].parameters():
                    param.requires_grad = False

        self.transform_matrices = []
        self.untransform_matrices = []
        self.middle_stage_outputs = []

    def forward(self, x):
        self.transform_matrices = []
        self.untransform_matrices = []
        self.middle_stage_outputs = []

        images = x.cpu().numpy()
        images = np.transpose(np.clip(images * 255.0, 0, 255).astype(np.uint8), (0, 2, 3, 1))

        for i in range(self.stage_count):
            if i == 0:
                identical_transform_matrix = np.eye(2, 3, dtype=np.float32)
                identical_transform_matrix[:, 2] = 1.0

                batch_size = len(x)

                identical_transforms = [identical_transform_matrix] * batch_size
                previous_landmarks = [self.average_landmark] * batch_size

                self.models[i].set_previous_prediction_prior(previous_landmarks, identical_transforms, identical_transforms)

                x = self.models[i](x)

                self.middle_stage_outputs.append(x)

                landmark_delta = x.view(-1, 68, 2).detach().cpu().numpy()
                previous_landmark = self.average_landmark + landmark_delta
            else:
                x = self.connection_layer(images, previous_landmark, self.models[i - 1].fc1_feature)

                self.transform_matrices.append(self.connection_layer.transform_matrices)
                self.untransform_matrices.append(self.connection_layer.untransform_matrices)

                self.models[i].set_previous_prediction_prior(previous_landmark, self.connection_layer.transform_matrices, self.connection_layer.untransform_matrices)

                x = self.models[i](x)

                self.middle_stage_outputs.append(x)

        image = np.copy(images[0])
        landmark_delta = x[0].detach().cpu().numpy()
        landmark = self.untransform_landmark(landmark_delta, 0)

        for l in landmark:
            cv2.circle(image, (int(l[0]), int(l[1])), 2, (255, 255, 255), -1)

        cv2.imshow("image", image)
        cv2.waitKey(10)

        return x

    def loss(self, prediction, target):
        loss = None

        for i in range(self.stage_count):
            if i != self.stage_count - 1:
                _loss = self.models[i].loss(self.middle_stage_outputs[i], target)
            else:
                _loss = self.models[i].loss(prediction, target)

            if loss is None:
                loss = _loss
            else:
                loss += _loss

        return loss

    def untransform_landmark(self, landmark_delta, index):
        return self.models[-1].untransform_landmark(landmark_delta, index)


class CanonicalFaceShape():
    def __init__(self, average_landmark):
        self.average_landmark = average_landmark

    def compute_transform_matrix(self, landmark):
        pass


class ConnectionLayers(nn.Module):
    def __init__(self, canonical_face_shape):
        super(ConnectionLayers, self).__init__()

        self.canonical_face_landmark = canonical_face_shape
        self.similarity_transform = SimilarityTransform()

        self.fc = nn.Linear(in_features=256, out_features=3136)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(size=(112, 112))

        self.transform_matrices = []
        self.untransform_matrices = []

    def forward(self, images, landmarks, previous_fc1_features):
        batch_size = len(images)

        input_images = []
        input_heatmaps = []

        self.transform_matrices = []
        self.untransform_matrices = []

        for (image, landmark) in zip(images, landmarks):
            transform_matrix, untransform_matrix = self.estimate_trans_untransform_matrices(landmark)

            input_image = self.transform_image(image, transform_matrix)

            canonical_landmark = self.transform_landmark(landmark, transform_matrix)
            input_heatmap = self.generate_heatmap(canonical_landmark)

            input_image = transforms.ToTensor()(input_image)
            input_heatmap = transforms.ToTensor()(input_heatmap)

            input_images.append(input_image)
            input_heatmaps.append(input_heatmap)

            self.transform_matrices.append(transform_matrix)
            self.untransform_matrices.append(untransform_matrix)

        input_images = torch.cat(input_images)
        input_heatmaps = torch.cat(input_heatmaps)

        previous_fc1_features = previous_fc1_features.squeeze()
        x = self.fc(previous_fc1_features)
        x = self.relu(x)
        x = torch.reshape(x, (-1, 1, 56, 56))
        input_features = self.upsample(x)
        input_features = input_features.squeeze(axis=1)

        input_images = input_images.to(input_features.device)
        input_heatmaps = input_heatmaps.to(input_features.device)

        input_tensor = torch.stack([input_images, input_heatmaps, input_features], axis=1)

        return input_tensor

    def estimate_trans_untransform_matrices(self, landmark):
        self.similarity_transform.estimate(landmark, self.canonical_face_landmark)
        transform_matrix = self.similarity_transform.params[0:2, :]
        transform_matrix = np.vstack([transform_matrix, np.array([0, 0, 1]).T])

        untransform_matrix = np.linalg.inv(transform_matrix)

        transform_matrix = transform_matrix[0:2, :].astype(dtype=np.float32)
        untransform_matrix = untransform_matrix[0:2, :].astype(dtype=np.float32)

        return transform_matrix, untransform_matrix

    @staticmethod
    def transform_image(image, transform_matrix):
        transformed_image = cv2.warpAffine(image, transform_matrix, (112, 112))

        return transformed_image

    @staticmethod
    def transform_landmark(landmark, transform_matrix):
        homogenious_landmark = np.hstack([landmark, np.ones((landmark.shape[0], 1))])

        transformed_landmark = np.matmul(transform_matrix, homogenious_landmark.T).T

        return transformed_landmark

    @staticmethod
    def generate_heatmap(landmark):
        heatmap_image = np.zeros((112 + 16 * 2, 112 + 16 * 2, 1), dtype=np.float32)

        for l in landmark:
            xrange = np.linspace(round(l[0]) - 16.0, round(l[0]) + 16.0, 33) - l[0]
            yrange = np.linspace(round(l[1]) - 16.0, round(l[1]) + 16.0, 33) - l[1]

            x_distances, y_distances = np.meshgrid(xrange, yrange)
            distances = np.sqrt(x_distances * x_distances + y_distances * y_distances)
            heatmap_patch = 1.0 / (1.0 + distances)
            heatmap_patch[distances > 16.0] = 0.0

            x = int(round(l[0])) + 16
            y = int(round(l[1])) + 16

            heatmap_image[y - 16:y + 16 + 1, x - 16:x + 16 + 1, 0] = np.maximum(
                heatmap_image[y - 16:y + 16 + 1, x - 16:x + 16 + 1, 0], heatmap_patch)

        heatmap_image = heatmap_image[16:16+112, 16:16+112, :]

        return heatmap_image
