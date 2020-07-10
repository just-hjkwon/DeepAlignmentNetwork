from pytorch_tools.dataset import ImageDataSet


class AFWDataset(ImageDataSet):
    def __init__(self, base_directory: str):
        super(AFWDataset, self).__init__(base_directory, "jpg")

    @staticmethod
    def train_datum_filter(file_path: str):
        return file_path

    @staticmethod
    def validation_datum_filter(file_path: str):
        return None

    @staticmethod
    def extract_label(file_path: str):
        return 0

    @staticmethod
    def is_valid_annotation(image_width, image_height, annotation):
        return True
