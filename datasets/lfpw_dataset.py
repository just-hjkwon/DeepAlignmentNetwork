from pytorch_tools.dataset import ImageDataSet


class LFPWDataset(ImageDataSet):
    def __init__(self, base_directory: str):
        super(LFPWDataset, self).__init__(base_directory, "png", "pts")

    @staticmethod
    def train_datum_filter(file_path: str):
        if "trainset" in file_path:
            return file_path
        else:
            return None

    @staticmethod
    def validation_datum_filter(file_path: str):
        if "testset" in file_path:
            return file_path
        else:
            return None

    @staticmethod
    def extract_label(file_path: str):
        return 0

    @staticmethod
    def is_valid_annotation(image_width, image_height, annotation):
        return True

    @staticmethod
    def parse_annotation(file_path: str):
        landmarks = []

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(3, 3 + 68):
                point = lines[i].strip("\n").split(" ")
                landmarks.append([float(point[0]), float(point[1])])

            return landmarks
