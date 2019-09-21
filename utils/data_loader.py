from pathlib import Path
from typing import Union

import tensorflow as tf
from tensorflow.data import Dataset


class DataLoader:
    """
    Loads images from a dir and preprocess them
    """

    def __init__(self, data_dir: Union[str, Path], batch_size: int = 64):
        self.data_dir = Path(data_dir)  # data dir that contains data in its subdirectories
        self.batch_size = batch_size

    def resize_and_crop(self, image_path):
        image_raw = tf.io.read_file(image_path)
        decoded_image = tf.image.decode_jpeg(image_raw, channels=3)
        resized_image = tf.image.resize(decoded_image, [256, 256])
        cropped_image = tf.image.central_crop(resized_image, 0.5) / 255
        return cropped_image

    def image_generator(self):
        all_image_paths = list(self.data_dir.glob('*/*'))
        all_image_paths = [str(image_path) for image_path in all_image_paths]
        for image_path in all_image_paths:
            yield self.resize_and_crop(image_path)

    def get_dataset(self):
        dataset = Dataset.from_generator(self.image_generator, tf.float32, self.output_shape)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(5)
        return dataset

    @property
    def num_samples(self):
        return len(list(self.data_dir.glob('*/*')))

    @property
    def output_shape(self):
        return tf.TensorShape([128, 128, 3])
