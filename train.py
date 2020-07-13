import os

import torch
import numpy as np
import tqdm
import time

from datasets.afw_dataset import AFWDataset
from datasets.helen_dataset import HELENDataset
from datasets.ibug_dataset import IBUGDataset
from datasets.lfpw_dataset import LFPWDataset
from datasets.private_300w_dataset import Private300WDataset

from landmark_dataset import LandmarkDataset
from models.vgg_based_model import VGGBasedModel

from tutor import Tutor

# load_snapshot = "latest"
load_snapshot = None

batch_size = 64

gpu = "0"
gpu_count = len(gpu.split(','))
device = torch.device("cuda:%s" % gpu if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
weight_decay = 0.00005

num_workers = 4

kwargs = {'num_workers': num_workers, 'pin_memory': True}
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

afw_database_root = "/data3/AFW/afw"
helen_database_root = "/data3/HELEN/helen"
ibug_database_root = "/data3/IBUG/ibug"
lfpw_database_root = "/data3/LFPW"
private_300w_database_root = "/data3/300W"


def main():
    init_epoch = 0
    max_epoch = 20200713

    train_dataset, valid_dataset = prepare_datasets()

    model = VGGBasedModel(in_channels=1)

    tutor = Tutor(model, device, learning_rate=learning_rate, weight_decay=weight_decay)

    if load_snapshot is not None:
        tutor.load(load_snapshot)
        validate(tutor, valid_dataset)

        init_epoch = tutor.get_epoch() + 1

    for epoch in range(init_epoch, max_epoch):
        tutor.set_epoch(epoch)
        train_loss = train_a_epoch(tutor, train_dataset)
        validation_loss = validate(tutor, valid_dataset)

        tutor.update_learning_rate(validation_loss)

        time_string = get_time_string()
        description = "%s Valid.| Epoch %d, validation loss: %f" % (time_string, tutor.epoch, validation_loss)
        print(description)

        if validation_loss <= tutor.best_error:
            tutor.best_error = validation_loss
            tutor.save('best')
            tutor.save('best_at_epoch_%04d' % epoch)

            time_string = get_time_string()
            phase_string = '%s Save snapshot of best ACER (%.4f)' % (time_string, tutor.best_error)
            print(phase_string)

        tutor.save('latest')


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


def train_a_epoch(tutor, train_data_set):
    loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True, **kwargs)

    time_string = get_time_string()
    current_learning_rate = tutor.get_current_learning_rate()
    scheduler_string = tutor.make_scheduler_state_string()

    phase_string = '%s Train | Epoch %d, learning rate: %f, %s' % (
        time_string, tutor.epoch, current_learning_rate, scheduler_string)
    print(phase_string)

    epoch_loss = 0.0
    trained_count = 0

    description = get_time_string()
    epoch_bar = tqdm.tqdm(loader, desc=description)

    for batch_idx, (data, target) in enumerate(epoch_bar):
        loss = tutor.train(data, target)

        trained_count += len(data)
        epoch_loss += loss * len(data)

        time_string = get_time_string()
        average_loss = epoch_loss / trained_count

        description = "%s Train | Epoch %d, Average train loss: %f (%f)" % (
            time_string, tutor.epoch, average_loss, loss)
        epoch_bar.set_description(description)

    average_loss = epoch_loss / trained_count
    time_string = get_time_string()
    description = "%s Train | Epoch %d, Average train loss: %f" % (time_string, tutor.epoch, average_loss)
    print(description)

    return average_loss


def validate(tutor, validation_data_set):
    loader = torch.utils.data.DataLoader(validation_data_set, batch_size=100, shuffle=True, **kwargs)

    time_string = get_time_string()
    current_learning_rate = tutor.get_current_learning_rate()
    phase_string = '%s Valid.| Epoch %d, learning rate: %f' % (
        time_string, tutor.epoch, current_learning_rate)
    print(phase_string)

    validation_loss = 0.0
    validated_count = 0

    description = get_time_string()
    epoch_bar = tqdm.tqdm(loader, desc=description)

    for batch_idx, (data, target) in enumerate(epoch_bar):
        loss, output = tutor.validate(data, target)

        validated_count += len(data)
        validation_loss += loss * len(data)

        time_string = get_time_string()
        average_loss = validation_loss / validated_count
        description = "%s Valid.| Epoch %d, Average validation loss: %f (%f)" % (
            time_string, tutor.epoch, average_loss, loss)
        epoch_bar.set_description(description)
        break

    average_loss = validation_loss / validated_count
    time_string = get_time_string()

    description = "%s Valid.| Epoch %d, Average validation loss: %f, error: %f" % (
        time_string, tutor.epoch, average_loss, average_loss)
    print(description)

    return average_loss


def get_time_string():
    string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return string


if __name__ == '__main__':
    main()
