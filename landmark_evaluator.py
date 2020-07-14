import numpy as np
from scipy.integrate import simps


class LandmarkEvaluator:
    def __init__(self, predictions, ground_truths):
        self.predictions = predictions
        self.ground_truths = ground_truths

        self.inter_ocular_distances = None
        self.inter_pupil_distances = None
        self.box_diagonal_distances = None

        self.compute_distance()

    def compute_distance(self):
        predictions = np.array(self.predictions)
        ground_truths = np.array(self.ground_truths)

        distances = np.linalg.norm((predictions - ground_truths), axis=2).mean(axis=1)

        outer_eye_indices = (36, 45)
        pupil_ranges = (range(36, 42), range(42, 48))

        left_pupil_point = ground_truths[:, pupil_ranges[0], :].mean(axis=1)
        right_pupil_point = ground_truths[:, pupil_ranges[1], :].mean(axis=1)

        box_lt_point = np.array([ground_truths[:, :, 0].min(axis=1), ground_truths[:, :, 1].min(axis=1)]).T
        box_rb_point = np.array([ground_truths[:, :, 0].max(axis=1), ground_truths[:, :, 1].max(axis=1)]).T

        outer_eye_distances = np.linalg.norm(
            ground_truths[:, outer_eye_indices[1], :] - ground_truths[:, outer_eye_indices[0], :], axis=1)
        pupil_distances = np.linalg.norm(left_pupil_point - right_pupil_point, axis=1)
        box_diagonal_distances = np.linalg.norm(box_lt_point - box_rb_point, axis=1)

        self.inter_ocular_distances = distances / outer_eye_distances
        self.inter_pupil_distances = distances / pupil_distances
        self.box_diagonal_distances = distances / box_diagonal_distances

    def get_mean_inter_ocular_distance(self):
        return self.inter_ocular_distances.mean()

    def get_mean_inter_pupil_distance(self):
        return self.inter_pupil_distances.mean()

    def get_mean_box_diagonal_distance(self):
        return self.box_diagonal_distances.mean()

    def get_inter_ocular_auc_and_failure_rate(self, threshold):
        return self.get_auc_and_failure_rate(self.inter_ocular_distances, threshold)

    @staticmethod
    def get_auc_and_failure_rate(errors, threshold, step=0.0001):
        sampling_steps = list(np.arange(0., threshold + step, step))
        count = len(errors)

        ced = [float(np.count_nonzero([errors <= x])) / count for x in sampling_steps]

        auc = simps(ced, x=sampling_steps) / threshold
        failure_rate = 1.0 - ced[-1]

        return auc, failure_rate
