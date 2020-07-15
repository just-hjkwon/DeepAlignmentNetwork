import torch
import torch.nn as nn
from torchvision import transforms

import cv2
import numpy as np

from skimage.transform import SimilarityTransform

from .vgg_based_model import VGGBasedModel


class DeepAlignmentNetwork(nn.Module):
    def __init__(self, average_landmark, stage_count):
        super(DeepAlignmentNetwork, self).__init__()

        self.average_landmark = average_landmark
        self.stage_count = stage_count
        self.models = torch.nn.Sequential()

        for i in range(stage_count):
            if i == 0:
                model = VGGBasedModel(in_channels=1, predefine_canonical_face_landmark=self.average_landmark)
                self.models.add_module("VGGBasedModel %d" % i, model)
            else:
                model = VGGBasedModel(in_channels=3)
                self.models.add_module("VGGBasedModel %d" % i, model)

        self.connection_layer = ConnectionLayers(self.average_landmark)
        self.models[0].load_state_dict(
            torch.load("/home/hjkwon/Desktop/DeepAlignmentNetwork-4-1st_stage/snapshots/best.weights"))

        self.transform_matrices = []
        self.untransform_matrices = []

    def forward(self, x):
        self.transform_matrices = []
        self.untransform_matrices = []

        images = x.cpu().numpy()
        images = np.transpose(np.clip(images * 255.0, 0, 255).astype(np.uint8), (0, 2, 3, 1))

        for i in range(self.stage_count):
            if i == 0:
                x = self.models[i](x)
            else:
                landmark_delta = x.view(-1, 68, 2).detach().cpu().numpy()
                predicted_landmark = self.average_landmark + landmark_delta

                x = self.connection_layer(images, predicted_landmark, self.models[i - 1].fc1_feature)

                self.transform_matrices.append(self.connection_layer.transform_matrices)
                self.untransform_matrices.append(self.connection_layer.untransform_matrices)

                trnasformed_landmarks = []
                for index, landmark in enumerate(predicted_landmark):
                    trnasformed_landmark = self.connection_layer.transform_landmark(landmark, self.transform_matrices[i-1][index])
                    trnasformed_landmarks.append(trnasformed_landmark)
                trnasformed_landmarks = np.array(trnasformed_landmarks)

                self.models[i].set_canonical_face_landmark(trnasformed_landmarks)

                x = self.models[i](x)

        return x

    def loss(self, prediction, target):
        loss = None

        for i in range(self.stage_count):
            if i == 0:
                loss = self.models[i].loss(prediction, target)
            else:
                loss += self.models[i].loss(prediction, target)

        return loss

    def untransform_landmark(self, landmark, index):
        for i in reversed(range(0, self.stage_count - 1)):
            if self.models[i].predefine_canonical_face_landmark is None:
                landmark = self.models[i + 1].predefine_canonical_face_landmark + landmark
            else:
                landmark = self.models[i + 1].canonical_face_landmark[index] + landmark

            landmark = self.connection_layer.transform_landmark(landmark, self.untransform_matrices[i][index])

        if self.models[0].predefine_canonical_face_landmark is None:
            landmark = self.models[0].predefine_canonical_face_landmark + landmark
        else:
            landmark = self.models[0].canonical_face_landmark[index] + landmark

        return landmark

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

        self.similarity_transform.estimate(self.canonical_face_landmark, landmark, )
        untransform_matrix = self.similarity_transform.params[0:2, :]

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

            x = int(round(l[0])) + 16
            y = int(round(l[1])) + 16

            heatmap_image[y - 16:y + 16 + 1, x - 16:x + 16 + 1, 0] = np.maximum(
                heatmap_image[y - 16:y + 16 + 1, x - 16:x + 16 + 1, 0], heatmap_patch)

        heatmap_image = heatmap_image[16:16+112, 16:16+112, :]

        return heatmap_image
