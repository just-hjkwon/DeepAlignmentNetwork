from torch.utils.data.dataset import Dataset
import numpy as np


class LandmarkDataset(Dataset):
    def __init__(self, datasets: list, is_train: bool):
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


    def prepare_index_map(self):
        self.index_map = []

        if self.is_train is True:
            count_array = []

            for dataset in self.datasets:
                dataset.set_train_mode()
                count_array.append(dataset.count(self.fixed_label))

            min_count = min(count_array)
            self.count = min_count * len(self.datasets)

            for i in range(len(self)):
                dataset_index = np.random.choice(range(0, len(self.datasets)))

                fixed_label = 0
                count = self.datasets[dataset_index].count(fixed_label)
                data_index = np.random.choice(range(0, count))

                self.index_map.append((fixed_label, dataset_index, data_index))
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

                image, target = self.create_input_and_target(image, annotation, label, index)

                image = self.augment(index, image, is_infer=True)
                image = np.copy(image[:, :, ::-1])

                self.val_cache[index] = [image, target]

            image = self.transform(image)
            target = int(target)

            return image, torch.LongTensor([target])

    def __len__(self):
        return self.count