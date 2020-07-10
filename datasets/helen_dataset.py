from pytorch_tools.dataset import ImageDataSet


class HELENDataset(ImageDataSet):
    def __init__(self, base_directory: str):
        super(HELENDataset, self).__init__(base_directory, "png")

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
