import random

from torch.utils.data.dataset import Dataset
import numpy as np


class LandmarkDataset(Dataset):
    def __init__(self, datasets: list, is_train: bool, average_landmark=None):
        super(LandmarkDataset, self).__init__()

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

        self.prepare_index_map()
        self.prepare_landmark_cache(average_landmark)

    def prepare_index_map(self):
        self.index_map = []

        if self.is_train is True:
            count_array = []

            for dataset in self.datasets:
                dataset.set_validation_mode()
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

    def prepare_landmark_cache(self, average_landmark=None):
        landmark_centers = []
        landmarks = []

        if average_landmark is None:
            _average_landmark = np.zeros((68, 2))

        for index in range(len(self)):
            label, dataset_index, data_index = self.index_map[index]
            _, annotation = self.datasets[dataset_index].get_datum(label, data_index)

            landmark = np.array(annotation)
            landmarks.append(landmark)

            landmark_center = landmark.mean(axis=0)
            landmark_centers.append(landmark_center)

            normalized_landmark = landmark - landmark_center

            if average_landmark is None:
                _average_landmark += normalized_landmark

        if average_landmark is None:
            _average_landmark /= len(self)

        self.landmark_centers = landmark_centers
        self.landmarks = landmarks

        if average_landmark is None:
            self.average_landmark = _average_landmark
        else:
            self.average_landmark = average_landmark

    def __getitem__(self, index):
        if self.is_train is True:
            label, dataset_index, data_index = self.index_map[index]
            image, annotation = self.datasets[dataset_index].get_datum(label, data_index)

            image, target = self.create_input_and_target(image, annotation, label, index)

            image = self.augment(None, image)
            image = np.copy(image[:, :, ::-1])

            image = self.transform(image)
            target = int(target)

            return image, torch.LongTensor([target])
        else:
            if index in list(self.val_cache.keys()):
                image, target = self.val_cache[index]
            else:
                label, dataset_index, data_index = self.index_map[index]

                repeat_number = int(math.floor(index / self.num_data))
                self.datasets[dataset_index].set_random_salt(repeat_number)

                image, annotation = self.datasets[dataset_index].get_datum(label, data_index)

                image, target = self.create_input_and_target(image, annotation, random_seed=index)

                image = self.augment(index, image, is_infer=True)
                image = np.copy(image[:, :, ::-1])

                self.val_cache[index] = [image, target]

            image = self.transform(image)
            target = int(target)

            return image, torch.LongTensor([target])

    @staticmethod
    def create_input_and_target(image, annotation, random_seed=None):
        random.seed(random_seed)

        if target == 0 and random.randint(0, 1) == 1:
            do_sample_at_background = True
        else:
            do_sample_at_background = False

        rotating_margin_factor = (math.sqrt(2.0) - 1.0) / 2.0

        if do_sample_at_background is False:
            face_box = FDDataset.make_box_from_landmark(annotation['landmark'])

            rotating_margin_factor = (math.sqrt(2.0) - 1.0) / 2.0
            margin_need = face_box[2] * rotating_margin_factor

            x = int(round(face_box[0] - margin_need))
            y = int(round(face_box[1] - margin_need))
            width = int(round(face_box[2] + (margin_need * 2.0)))
            height = int(round(face_box[3] + (margin_need * 2.0)))
        else:
            min_length = int(math.ceil(112.0 + (2.0 * (112.0 * rotating_margin_factor))))
            image_width = image.shape[1]
            image_height = image.shape[0]

            sample_length = random.randint(min_length, min(image_width, image_height) - 1)

            x = random.randint(0, image_width - sample_length - 1)
            y = random.randint(0, image_height - sample_length - 1)
            width = sample_length
            height = sample_length

        image = Cropper.crop(image, x, y, width, height)

        return image, target

    def __len__(self):
        return self.count