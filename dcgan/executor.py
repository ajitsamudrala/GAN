import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf

from utils.data_loader import DataLoader


class Executor:
    """
    Base class to execute and quickly prototype different network architectures
    """

    def __init__(self, gen_network: Callable, dis_network: Callable, dataloader_args: Dict):
        data_loader = DataLoader(**dataloader_args)
        self.dataset = data_loader.get_dataset()
        self.generator = gen_network()
        self.discriminator = dis_network(data_loader.output_shape, (1,))

    def run(self, batch_size: int, num_epochs: int, callbacks: List = None):
        generator_optimizer = tf.train.AdamOptimizer(1e-4)
        discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
        

    @staticmethod
    def generator_loss(generated_output):
        return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

    @staticmethod
    def discriminator_loss(real_output, generated_output):
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output),
                                                         logits=generated_output)
        total_loss = real_loss + generated_loss

        return total_loss


def args_parser():
    parser = argparse.ArgumentParser(description='Run GANs on Celeb Dataset')
    parser.add_argument('epochs', help='Number of training epochs', type=int, default=15)
    return parser
