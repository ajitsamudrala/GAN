import os
import math
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    """
    Loads images from a dir
    """

    def __init__(self, data_dir: Union[str, Path], image_dir: Union[str, Path], batch_size: int = 32,
                 output_size: Tuple[int, int] = (256, 256), augment_fn: Optional[Callable] = None):
        self.data_dir = Path(data_dir)  # data dir that contains data in its subdirectories
        self.image_dir = Path(image_dir)  # dir in which data is stored
        self.batch_size = batch_size
        self.output_size = output_size
        self.augment_fn = augment_fn
        self.num_samples = len(os.listdir(self.image_dir))

    def image_generator(self):
        image_gen = ImageDataGenerator(featurewise_center=True)  # mean centering
        sample_data = self.get_sample_data()
        image_gen.fit(sample_data)  # to calculate sample stats for mean centering
        data_gen = image_gen.flow_from_directory(
            self.data_dir,
            target_size=self.output_size,
            batch_size=self.batch_size,
            class_mode=None)
        return data_gen

    def get_sample_data(self):
        random_100_indicies = np.random.permutation(self.num_samples - 1)
        sample_data_names = np.array(os.listdir(self.image_dir))[random_100_indicies]
        sample_data_paths = [self.image_dir / image_name for image_name in sample_data_names]
        sample_data = [plt.imread(path) for path in sample_data_paths]
        return sample_data

    @property
    def num_batches(self):
        return math.ceil(self.num_samples / self.batch_size)
