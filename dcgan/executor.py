import argparse
import os
from pathlib import Path
from typing import Callable, Dict, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

from networks import LATENT_SHAPE, generator, discriminator
from utils.data_loader import DataLoader


class Executor:
    """
    Base class to execute and quickly prototype different network architectures
    """

    def __init__(self, gen_network: Callable, dis_network: Callable, dataloader_args: Dict):
        tf.enable_eager_execution()
        self.data_loader = DataLoader(**dataloader_args)  # util to load dataset from dir
        self.dataset = self.data_loader.get_dataset()
        self.generator = gen_network()
        self.discriminator = dis_network(self.data_loader.output_shape, (1,))
        noise = Input(shape=LATENT_SHAPE)
        gen_img = self.generator(noise)
        disc_output = self.discriminator(gen_img)
        self.combined = Model(noise, disc_output)
        self.combined.layers[2].trainable = False  # combined model is only trained on generator network
        self.fixed_noise = tf.random.normal([128] + list(LATENT_SHAPE))

    def run(self, num_epochs: int):
        self.discriminator.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())
        self.combined.compile(loss=self.loss(), optimizer=self.optimizer())
        for epoch in range(num_epochs):
            for i, batch in enumerate(self.dataset):
                random_noise = tf.random.normal([batch.shape[0]] + list(LATENT_SHAPE))
                gen_images = self.generator.predict(random_noise)

                # update discriminator parameters
                disc_real_loss = self.discriminator.train_on_batch(batch, tf.ones(batch.shape[0]))
                disc_fake_loss = self.discriminator.train_on_batch(gen_images, tf.zeros(batch.shape[0]))
                disc_total_loss = tf.add(disc_real_loss[0], disc_fake_loss[0])
                disc_real_mean_prediction = self.discriminator.predict(batch).mean()
                disc_fake_mean_prediction = self.discriminator.predict(gen_images).mean()

                # update generator parameters
                gen_loss = self.combined.train_on_batch(random_noise, tf.ones(batch.shape[0]))
                disc_fake_mean_prediction_after_update = self.combined.predict(random_noise).mean()

                if i % 50 == 0:
                    print(f'[{epoch}/{num_epochs}][batch: {i}]\tLoss_D: {disc_total_loss:.4f}\tLoss_G: {gen_loss:.4f} \
                            D(x): {disc_real_mean_prediction} D(G(z)): {disc_fake_mean_prediction}/{disc_fake_mean_prediction_after_update}')

    @staticmethod
    def optimizer():
        return tf.train.AdamOptimizer(1e-4)

    @staticmethod
    def loss():
        return 'binary_crossentropy'

    @staticmethod
    def metrics():
        return ['accuracy']


def args_parser():
    parser = argparse.ArgumentParser(description='Run GANs on Celeb Dataset')
    parser.add_argument('data_dir', help='Path to dir containing data', type=str)
    parser.add_argument('epochs', help='Number of training epochs', type=int, default=15)
    return parser


if __name__ == '__main__':
    args = args_parser().parse_args()
    executor = Executor(gen_network=generator, dis_network=discriminator, dataloader_args={'data_dir': args.data_dir})
    executor.run(args.epochs)
