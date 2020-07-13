import sys

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class Tutor:
    def __init__(self, network, learning_rate=0.01, weight_decay=0.0001, lr_decay_factor=2.0/3.0, patience=30):
        self.network = network

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
