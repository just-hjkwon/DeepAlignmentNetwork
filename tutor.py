import sys
import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class Tutor:
    def __init__(self, network, device, learning_rate=0.01, weight_decay=0.0001, lr_decay_factor=2.0/3.0, patience=30):
        self.device = device
        self.network = network.to(self.device)

        def make_optimizer_parameters(model):
            parameters = list()

            for name, param in model.named_parameters():
                if name[-9:] == "conv.bias":
                    parameters.append({"params": param, 'lr': learning_rate * 2.0, 'weight_decay': 0.0})
                else:
                    parameters.append({"params": param, 'lr': learning_rate, 'weight_decay': weight_decay})

            return parameters

        optimizer_parameters = make_optimizer_parameters(self.network)

        self.optimizer = optim.Adam(optimizer_parameters, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=lr_decay_factor, patience=patience)

        self.snapshot_directory = "snapshots"

        self.best_error = sys.float_info.max
        self.epoch = 0

    def train(self, input_data, target):
        if self.network.training is not True:
            self.network.train()

        input_data = input_data.to(self.device)
        prediction = self.network(input_data)

        if type(target) is list:
            for i in range(len(target)):
                target[i] = target[i].to(self.device)
        elif type(target) is torch.tensor:
            target = target.to(self.device)

        loss = self.network.loss(prediction, target)
        loss.backward()

        self.optimizer.step()

        return loss.data

    def validate(self, input_data, target):
        if self.network.training is True:
            self.network.eval()

        input_data = input_data.to(self.device)
        prediction = self.network(input_data)

        if type(target) is list:
            for i in range(len(target)):
                target[i] = target[i].to(self.device)
        elif type(target) is torch.tensor:
            target = target.to(self.device)

        loss = self.network.loss(prediction, target)
        return loss.data, prediction.data

    def save(self, prefix):
        if not os.path.exists(self.snapshot_directory):
            os.mkdir(self.snapshot_directory)

        self.network.eval()

        filename = os.path.join(self.snapshot_directory, "%s.weights" % prefix)
        torch.save(self.network.state_dict(), filename)

        filename = os.path.join(self.snapshot_directory, "%s.optimizer" % prefix)
        torch.save(self.optimizer.state_dict(), filename)

        filename = os.path.join(self.snapshot_directory, "%s.scheduler" % prefix)
        torch.save(self.scheduler.state_dict(), filename)

        train_state = {'best_error': self.best_error, 'epoch': self.epoch}
        filename = os.path.join(self.snapshot_directory, "%s.state" % prefix)
        torch.save(train_state, filename)

    def load(self, prefix):
        self.network.eval()

        filename = os.path.join(self.snapshot_directory, "%s.weights" % prefix)
        self.network.load_state_dict(torch.load(filename))

        filename = os.path.join(self.snapshot_directory, "%s.optimizer" % prefix)
        self.optimizer.load_state_dict(torch.load(filename))

        filename = os.path.join(self.snapshot_directory, "%s.scheduler" % prefix)
        self.scheduler.load_state_dict(torch.load(filename))

        filename = os.path.join(self.snapshot_directory, "%s.state" % prefix)
        train_state = torch.load(filename)

        if 'best_error' in train_state.keys():
            self.best_error = train_state['best_error']
        else:
            self.best_error = sys.float_info.max

        self.epoch = train_state['epoch']

    def update_learning_rate(self, validation_loss):
        self.scheduler.step(validation_loss)

    def get_current_learning_rate(self):
        learning_rate = 0.0

        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']

        return learning_rate

    def make_scheduler_state_string(self):
        string = "Best error: %f, Best loss: %f, patience: %d/%d" % (
            self.best_error, self.scheduler.best, self.scheduler.num_bad_epochs, self.scheduler.patience)

        return string

    def get_epoch(self):
        return self.epoch

    def set_epoch(self, epoch):
        self.epoch = epoch