import os

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import tqdm
from tabulate import tabulate
import time

import cv2

from datasets.afw_dataset import AFWDataset
from datasets.helen_dataset import HELENDataset
from datasets.ibug_dataset import IBUGDataset
from datasets.lfpw_dataset import LFPWDataset
from datasets.private_300w_dataset import Private300WDataset

from landmark_dataset import LandmarkDataset
from models.vgg_based_model import VGGBasedModel
from models.deep_alignment_network import DeepAlignmentNetwork

from landmark_evaluator import LandmarkEvaluator

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

tensorboard_directory = os.path.basename(os.path.abspath("."))
tensorboard_logdir = "/workspace/tensorboard_DAN_logs/" + tensorboard_directory

tensorboard_tags = ['General/Learning rate', 'Loss/Train', 'Loss/Validation',
                    'Pupil/Common', 'Pupil/Challenging', 'Pupil/Full',
                    'Inter-ocular/Common', 'Inter-ocular/Challenging', 'Inter-ocular/Full',
                    'Dialgonal-box/Common', 'Dialgonal-box/Challenging', 'Dialgonal-box/Full',
                    'Public 300W/AUC', 'Public 300W/Failure rate',
                    'Private 300W/AUC', 'Private 300W/Failure rate']

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

    train_dataset, test_datasets = prepare_datasets()

    # model = VGGBasedModel(in_channels=1, predefine_canonical_face_landmark=train_dataset.average_landmark)
    model = DeepAlignmentNetwork(train_dataset.average_landmark, 2)

    tutor = Tutor(model, device, learning_rate=learning_rate, weight_decay=weight_decay)

    if load_snapshot is not None:
        tutor.load(load_snapshot)
        validate(tutor, train_dataset)

        init_epoch = tutor.get_epoch() + 1

    purge_step = init_epoch
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_logdir, purge_step=purge_step)

    for epoch in range(init_epoch, max_epoch):
        tutor.set_epoch(epoch)
        train_loss = train_a_epoch(tutor, train_dataset)
        validation_loss = validate(tutor, train_dataset)

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

        common_test = test(tutor, test_datasets[0])
        challenging_test = test(tutor, test_datasets[1])
        public_300w_test = test(tutor, test_datasets[2])
        private_300w_test = test(tutor, test_datasets[3])
        full_test = test(tutor, test_datasets[4])

        common_inter_pupil_error = common_test.get_mean_inter_pupil_distance() * 100.0
        common_inter_ocular_error = common_test.get_mean_inter_ocular_distance() * 100.0
        common_diagonal_box_error = common_test.get_mean_box_diagonal_distance() * 100.0

        challenging_inter_pupil_error = challenging_test.get_mean_inter_pupil_distance() * 100.0
        challenging_inter_ocular_error = challenging_test.get_mean_inter_ocular_distance() * 100.0
        challenging_diagonal_box_error = challenging_test.get_mean_box_diagonal_distance() * 100.0

        full_inter_pupil_error = full_test.get_mean_inter_pupil_distance() * 100.0
        full_inter_ocular_error = full_test.get_mean_inter_ocular_distance() * 100.0
        full_diagonal_box_error = full_test.get_mean_box_diagonal_distance() * 100.0

        public_300w_auc_0_08, public_300w_failure_rate = public_300w_test.get_inter_ocular_auc_and_failure_rate(0.08)
        public_300w_failure_rate *= 100.0
        private_300w_auc_0_08, private_300w_failure_rate = private_300w_test.get_inter_ocular_auc_and_failure_rate(0.08)
        private_300w_failure_rate *= 100.0

        table0_header = ["", "Common", "Challenging", "Full"]
        table0 = [["Inter pupil", common_inter_pupil_error, challenging_inter_pupil_error, full_inter_pupil_error],
                  ["Inter ocular", common_inter_ocular_error, challenging_inter_ocular_error, full_inter_ocular_error],
                  ["Diagonal box", common_diagonal_box_error, challenging_diagonal_box_error, full_diagonal_box_error]]

        table1_header = ["", "AUC 0.08", "Failure (%%)"]
        table1 = [["300W public", public_300w_auc_0_08, public_300w_failure_rate],
                  ["300W private", private_300w_auc_0_08, private_300w_failure_rate]]

        print(tabulate(table0, headers=table0_header, tablefmt='fancy_grid'))
        print(tabulate(table1, headers=table1_header, tablefmt='fancy_grid'))

        current_learning_rate = tutor.get_current_learning_rate()

        tensorboard_writer.add_scalar(tensorboard_tags[0], current_learning_rate, tutor.epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[1], train_loss, tutor.epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[2], validation_loss, tutor.epoch)

        tensorboard_writer.add_scalar(tensorboard_tags[3], common_inter_pupil_error, tutor.epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[4], challenging_inter_pupil_error, tutor.epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[5], full_inter_pupil_error, tutor.epoch)

        tensorboard_writer.add_scalar(tensorboard_tags[6], common_inter_ocular_error, tutor.epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[7], challenging_inter_ocular_error, tutor.epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[8], full_inter_ocular_error, tutor.epoch)

        tensorboard_writer.add_scalar(tensorboard_tags[9], common_diagonal_box_error, tutor.epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[10], challenging_diagonal_box_error, tutor.epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[11], full_diagonal_box_error, tutor.epoch)

        tensorboard_writer.add_scalar(tensorboard_tags[12], public_300w_auc_0_08, tutor.epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[13], public_300w_failure_rate, tutor.epoch)

        tensorboard_writer.add_scalar(tensorboard_tags[14], private_300w_auc_0_08, tutor.epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[15], private_300w_failure_rate, tutor.epoch)


def prepare_datasets():
    afw_dataset = AFWDataset(afw_database_root)
    helen_dataset = HELENDataset(helen_database_root)
    ibug_dataset = IBUGDataset(ibug_database_root)
    lfpw_dataset = LFPWDataset(lfpw_database_root)
    private_300w_dataset = Private300WDataset(private_300w_database_root)

    train_datasets = [afw_dataset, helen_dataset, lfpw_dataset]
    common_test_datasets = [lfpw_dataset, helen_dataset]
    challenging_test_datasets = [ibug_dataset]
    public_300w_test_datasets = [lfpw_dataset, helen_dataset, ibug_dataset]
    private_300w_test_datasets = [private_300w_dataset]
    full_datasets = [afw_dataset, helen_dataset, ibug_dataset, lfpw_dataset, private_300w_dataset]

    # train_dataset = LandmarkDataset(datasets=train_datasets, is_train=True)
    # np.save("average_landmark", train_dataset.average_landmark)

    train_dataset = LandmarkDataset(datasets=train_datasets, is_train=True, average_landmark=None)

    common_test_dataset = LandmarkDataset(datasets=common_test_datasets, is_train=False,
                                          average_landmark=train_dataset.average_landmark)
    challenging_test_dataset = LandmarkDataset(datasets=challenging_test_datasets, is_train=False,
                                               average_landmark=train_dataset.average_landmark)
    public_300w_test_dataset = LandmarkDataset(datasets=public_300w_test_datasets, is_train=False,
                                               average_landmark=train_dataset.average_landmark)
    private_300w_test_dataset = LandmarkDataset(datasets=private_300w_test_datasets, is_train=False,
                                                average_landmark=train_dataset.average_landmark)
    full_dataset = LandmarkDataset(datasets=full_datasets, is_train=False,
                                   average_landmark=train_dataset.average_landmark)

    return train_dataset, [common_test_dataset, challenging_test_dataset, public_300w_test_dataset,
                           private_300w_test_dataset, full_dataset]


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
    phase_string = '%s Valid.| Epoch %d, learning rate: %f' % (time_string, tutor.epoch, current_learning_rate)
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


def test(tutor: Tutor, test_data_set: LandmarkDataset):
    loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, shuffle=False, **kwargs)

    time_string = get_time_string()
    current_learning_rate = tutor.get_current_learning_rate()
    phase_string = '%s Test  | Epoch %d, learning rate: %f' % (time_string, tutor.epoch, current_learning_rate)
    print(phase_string)

    predicted_landmarks = []
    gt_landmarks = []

    description = get_time_string()
    epoch_bar = tqdm.tqdm(loader, desc=description)

    for batch_idx, (data, target) in enumerate(epoch_bar):
        _, output = tutor.validate(data, target)

        time_string = get_time_string()
        description = "%s Test  | Epoch %d" % (time_string, tutor.epoch)
        epoch_bar.set_description(description)

        for index, (_output, _target) in enumerate(zip(output, target[0])):
            landmark_delta = _output.view(-1, 2).cpu().numpy()
            predicted_landmark = tutor.network.untransform_landmark(landmark_delta, index)

            gt_landmark = _target.view(-1, 2).cpu().numpy()

            predicted_landmarks.append(predicted_landmark)
            gt_landmarks.append(gt_landmark)

    evaluator = LandmarkEvaluator(predicted_landmarks, gt_landmarks)

    time_string = get_time_string()
    description = "%s Test  | Epoch %d" % (time_string, tutor.epoch)
    print(description)

    return evaluator


def get_time_string():
    string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return string


if __name__ == '__main__':
    main()
