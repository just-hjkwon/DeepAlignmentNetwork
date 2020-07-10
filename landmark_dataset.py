from torch.utils.data.dataset import Dataset


class LandmarkDataset(Dataset):
    def __init__(self, datasets: list, is_train: bool):
        super(LandmarkDataset, self).__init__()

        self.is_train = is_train
        self.daatasets = datasets
