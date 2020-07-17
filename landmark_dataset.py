import random
import math

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch

import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from pytorch_tools.image.cropper import Cropper


class LandmarkDataset(Dataset):
    def __init__(self, datasets: list, is_train: bool, average_landmark: np.ndarray = None):
        super(LandmarkDataset, self).__init__()

        self.target_size = 112
        self.target_face_box_size_range = (self.target_size * 0.30, self.target_size * 0.75)
        self.target_average_face_box_size = 56

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.fixed_label = 0

        self.is_train = is_train

        self.datasets = []
        for dataset in datasets:
            if is_train is True:
                dataset.set_train_mode()
            else:
                dataset.set_validation_mode()

            count = dataset.count(self.fixed_label)

            if count > 0:
                self.datasets.append(dataset)

        self.count = 0

        self.index_map = None

        self.prepare_index_map()

        self.landmarks = None
        self.average_landmark = None
        self.face_boxes = None

        self.prepare_landmark_cache(average_landmark)

        self.val_cache = {}

    def prepare_index_map(self):
        self.index_map = []

        if self.is_train is True:
            count_array = []

            for dataset in self.datasets:
                dataset.set_train_mode()
                count_array.append(dataset.count(self.fixed_label))

            self.count = sum(count_array)

            for dataset_index, dataset in enumerate(self.datasets):
                fixed_label = 0
                for i in range(dataset.count(fixed_label)):
                    self.index_map.append((fixed_label, dataset_index, i))
        else:
            count_array = []

            for dataset in self.datasets:
                dataset.set_validation_mode()
                count_array.append(dataset.count(self.fixed_label))

            self.count = sum(count_array)

            for dataset_index, dataset in enumerate(self.datasets):
                fixed_label = 0
                for i in range(dataset.count(fixed_label)):
                    self.index_map.append((fixed_label, dataset_index, i))

    def prepare_landmark_cache(self, average_landmark: np.ndarray = None):
        landmarks = []
        face_boxes = []

        if average_landmark is None:
            _average_landmark = np.zeros((68, 2))

        for index in range(len(self)):
            label, dataset_index, data_index = self.index_map[index]
            if self.is_train is True:
                self.datasets[dataset_index].set_train_mode()
            else:
                self.datasets[dataset_index].set_validation_mode()

            _, annotation_filename = self.datasets[dataset_index].get_filename(label, data_index)
            annotation = self.datasets[dataset_index].parse_annotation(annotation_filename)

            landmark = np.array(annotation)
            landmarks.append(landmark)

            face_box = landmark.min(axis=0).tolist() + landmark.max(axis=0).tolist()
            face_boxes.append(face_box)

            if average_landmark is None:
                landmark_center = landmark.mean(axis=0)
                normalized_landmark = landmark - landmark_center
                _average_landmark += normalized_landmark

        if average_landmark is None:
            _average_landmark /= len(self)

        self.landmarks = landmarks
        self.face_boxes = face_boxes

        if average_landmark is None:
            average_landmark = _average_landmark

        average_face_box = face_box = average_landmark.min(axis=0).tolist() + average_landmark.max(axis=0).tolist()
        box_width = face_box[2] - face_box[0]
        box_height = face_box[3] - face_box[1]
        box_length = max(box_width, box_height)
        average_landmark_scale = self.target_average_face_box_size / box_length

        average_landmark -= average_landmark.mean(axis=0)
        average_landmark *= average_landmark_scale
        average_landmark += self.target_size / 2.0

        self.average_landmark = average_landmark

    def __getitem__(self, index: int):
        if self.is_train is True:
            label, dataset_index, data_index = self.index_map[index]

            self.datasets[dataset_index].set_train_mode()
            image, _ = self.datasets[dataset_index].get_datum(label, data_index)
            annotation = [self.face_boxes[index], self.landmarks[index]]

            image, target, pupil_distance = self.create_input_and_target(image, annotation)

            target = target.flatten()

            image = self.transforms(image)
            target = torch.tensor(target)
        else:
            if index in list(self.val_cache.keys()):
                image, target, pupil_distance = self.val_cache[index]
            else:
                label, dataset_index, data_index = self.index_map[index]

                self.datasets[dataset_index].set_validation_mode()
                image, _ = self.datasets[dataset_index].get_datum(label, data_index)
                annotation = [self.face_boxes[index], self.landmarks[index]]

                image, target, pupil_distance = self.create_input_and_target(image, annotation, random_seed=index)

                target = target.flatten()

                image = self.transforms(image)
                target = torch.tensor(target)

        return image, (target, pupil_distance)

    def create_input_and_target(self, image, annotation: list, random_seed: int = None):
        random.seed(random_seed)

        face_box = np.array(annotation[0], dtype=np.float32)
        landmark = np.array(annotation[1], dtype=np.float32)

        box_width = face_box[2] - face_box[0]
        box_height = face_box[3] - face_box[1]
        box_length = max(box_width, box_height)

        if self.is_train is True:
            scale = random.uniform(self.target_face_box_size_range[0] / box_length,
                                   self.target_face_box_size_range[1] / box_length)
        else:
            scale = self.target_average_face_box_size / box_length

        width, height = image.shape[1], image.shape[0]
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        width_scale = resized_width / width
        height_scale = resized_height / height

        image = cv2.resize(image, (resized_width, resized_height))

        landmark[:, 0] *= width_scale
        landmark[:, 1] *= height_scale
        face_box[0::2] *= width_scale
        face_box[1::2] *= height_scale
        box_center = [face_box[0::2].mean(), face_box[1::2].mean()]

        if self.is_train is True:
            box_center[0] += random.randint(-10, 10)
            box_center[1] += random.randint(-10, 10)

        crop_x = int(round(box_center[0] - 56))
        crop_y = int(round(box_center[1] - 56))

        if self.is_train is True:
            rotation_margin = int(math.ceil(((math.sqrt(2.0) * 112) - 112) / 2.0))
            crop_x -= rotation_margin
            crop_y -= rotation_margin

            image = Cropper.crop(image, crop_x, crop_y,
                                 self.target_size + (2 * rotation_margin), self.target_size + (2 * rotation_margin))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=2)

            landmark[:, 0] -= crop_x
            landmark[:, 1] -= crop_y

            augmentations = iaa.Sequential([
                iaa.Sequential([iaa.Rotate((-30, 30)),
                                iaa.CenterCropToFixedSize(height=self.target_size, width=self.target_size)]),
                iaa.Sequential([iaa.Sometimes(0.5, iaa.Add((-30, 30))),
                                iaa.Sometimes(0.5, iaa.LinearContrast((0.4, 1.6))),
                                iaa.Sometimes(0.5, iaa.Fliplr(0.5))], random_order=True)
            ])

            keypoints = KeypointsOnImage([Keypoint(x=l[0], y=l[1]) for l in landmark], shape=image.shape)
            image, keypoints = augmentations(image=image, keypoints=keypoints)

            if landmark[0, 0] != keypoints.keypoints[0].x:
                landmark_flip_mapping_indices = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                                 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                                                 27, 28, 29, 30,
                                                 35, 34, 33, 32, 31,
                                                 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40,
                                                 54, 53, 52, 51, 50, 49, 48,
                                                 59, 58, 57, 56, 55,
                                                 64, 63, 62, 61, 60, 67, 66, 65
                                                 ]

                for target_i, source_i in enumerate(landmark_flip_mapping_indices):
                    landmark[target_i, 0] = keypoints.keypoints[source_i].x
                    landmark[target_i, 1] = keypoints.keypoints[source_i].y
        else:
            image = Cropper.crop(image, crop_x, crop_y, self.target_size, self.target_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=2)

            landmark[:, 0] -= crop_x
            landmark[:, 1] -= crop_y

        left_pupil = landmark[36:42, :].mean(axis=0)
        right_pupil = landmark[42:48, :].mean(axis=0)

        pupil_distance = np.linalg.norm((left_pupil - right_pupil))

        return image, landmark, pupil_distance

    def __len__(self):
        return self.count
