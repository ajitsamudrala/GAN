from pathlib import Path
from typing import Callable, Optional, Union

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import GeneratorEnqueuer


class DataLoader:
    """
    Loads images from a dir
    """

    def __init__(self, image_dir: Union[str, Path], batch_size: int = 32, augment_fn: Optional[Callable] = None):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.augment_fn = augment_fn

    def image_generator(self):
        image_gen = ImageDataGenerator(featurewise_center=True)  # mean centering
        data_gen = image_gen.flow_from_directory(
            self.image_dir,
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode=None)
        gen_enq = GeneratorEnqueuer(data_gen, use_multiprocessing=True, random_seed=None)
        return gen_enq
