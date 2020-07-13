from datasets.afw_dataset import AFWDataset
from datasets.helen_dataset import HELENDataset
from datasets.ibug_dataset import IBUGDataset
from datasets.lfpw_dataset import LFPWDataset
from datasets.private_300w_dataset import Private300WDataset

from landmark_dataset import LandmarkDataset
from models.vgg_based_model import VGGBasedModel

from tutor import Tutor

import numpy as np


afw_database_root = "/data3/AFW/afw"
helen_database_root = "/data3/HELEN/helen"
ibug_database_root = "/data3/IBUG/ibug"
lfpw_database_root = "/data3/LFPW"
private_300w_database_root = "/data3/300W"


def main():
    train_dataset, valid_dataset = prepare_datasets()

    model = VGGBasedModel(in_channels=1)
    model.cuda()

    tutor = Tutor(model)


def prepare_datasets():
    afw_dataset = AFWDataset(afw_database_root)
    helen_dataset = HELENDataset(helen_database_root)
    ibug_dataset = IBUGDataset(ibug_database_root)
    lfpw_dataset = LFPWDataset(lfpw_database_root)
    private_300w_dataset = Private300WDataset(private_300w_database_root)

    datasets = [afw_dataset, helen_dataset, ibug_dataset, lfpw_dataset, private_300w_dataset]

    train_dataset = LandmarkDataset(datasets=datasets, is_train=True)
    valid_dataset = LandmarkDataset(datasets=datasets, is_train=False, average_landmark=train_dataset.average_landmark)

    return train_dataset, valid_dataset


if __name__ == '__main__':
    main()