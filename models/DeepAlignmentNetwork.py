import torch
import torch.nn as nn

import cv2
import numpy as np

from skimage.transform import SimilarityTransform

from .vgg_based_model import VGGBasedModel


class DeepAlignmentNetwork(nn.Module):
    def __init__(self, average_landmark, stage_count):
        super(DeepAlignmentNetwork, self).__init__()

        self.average_landmark = average_landmark
        self.stage_count = stage_count

        self.models = []
        for i in range(stage_count):
            if stage_count == 0:
                model = VGGBasedModel(in_channels=1)
            else:
                model = VGGBasedModel(in_channels=3)

            self.models.append(model)

        self.canonical_face_shape = CanonicalFaceShape()
        self.connection_layer = ConnectionLayers(self.average_landmark)

    def forward(self, x):
        images = x.cpu().numpy()

        for i in range(self.stage_count):
            if self.stage_count == 0:
                x = self.models[i](x)
            else:
                landmark_delta = x.view(-1, 68, 2).cpu().numpy()
                predicted_landmark = self.average_landmark + landmark_delta

                x = self.connection_layer(images, predicted_landmark, self.model[i - 1].fc1_feature)
                x = self.models[i](x)

    def generate_heatmap(self, landmark, transform_matrix):
        pass

    def generate_canonical_input_image(self, images, transform_matrix):
        pass


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

        self.fc = nn.Linear(in_feature=512, out_features=3136)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(size=(112, 112))

    def forward(self, images, landmarks, previous_fc1_features):
        transform_matrix = self.estimate_transform_matrix(landmark)

        input_images = self.transform_image(images, transform_matrix)
        landmarks = self.transform_landmark(landmark, transform_matrix)
        input_heatmaps = self.generate_heatmap(landmarks)

        x = self.fc(previous_fc1_features)
        x = self.relu(x)
        x = torch.reshape(x, (-1, 56, 56, 1))
        input_feature = self.upsample(x)

        input_tensor = np.concat([input_images, input_heatmaps, input_feature])

        return input_tensor

    def estimate_transform_matrix(self, landmark):
        self.similarity_transform.estimate(landmark, self.canonical_face_landmark)
        transform_matrix = self.similarity_transform.params[0:2, :]
        return transform_matrix

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
        heatmap_image = None
        return heatmap_image
