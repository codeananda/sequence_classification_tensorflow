import numpy as np

# from tensorflow.keras import utils
from keras.utils import Sequence
from numpy import random


class ElectroAugmenter(Sequence):
    def __init__(
        self,
        X,
        Y,
        batch_size,
        horizontal_shift,
        vertical_shift,
        noise_shift,
        noise_shift_scale=0.1,
        multiplier=10,
        shuffle=True,
        seed=None,
        augmentation_percentage=0.8,
    ):
        """
        Args:

                X: 2D numpy array shape (num_samples, timesteps). It will be reshaped
                        to (num_samples, 1, timesteps) after it has been augmented
                y: 1D or 2D numpy array shape (num_samples,) or (num_samples, num_classes)

                horizontal_shift: std of 0-centered Gaussian from which we will draw a value to
                        determine how much horizontal shift to apply to each sample.
                vertical_shift: std of 0-centered Gaussian from which we will draw a value to
                        determine how much systematic vertical shift to apply to each sample.
                noise_shift: std of 0-centered Gaussian from which we will draw many values to
                        determine how much random vertical shift to apply to each sample.
                noise_shift_scale: float which is multiplied by noise_shift to give control over
                        the random noise scaling.
                multiplier: Size multiple over original dataset size. Final number of samples will be
                        this number times original dataset size, increased to next multiple of batch_size
                augmentation_percentage: percentage of returned samples to augment
        """
        super().__init__()
        self.X = X
        self.Y = Y

        self.batch_size = batch_size
        if not shuffle:
            self.seed = None
        elif seed is None:
            self.seed = 0
        else:
            self.seed = seed

        self.horizontal_shift = horizontal_shift
        self.vertical_shift = vertical_shift
        self.noise_shift = noise_shift
        self.noise_shift_scale = noise_shift_scale

        self.multiplier = multiplier

        self.rng = random.RandomState(seed=self.seed)

        # create base ordering
        # include extra for padding final batch

        if shuffle:
            self.shuffle()
        else:
            self.order = self._create_ordering()

            batch_count = len(self)
            total_samples = batch_count * self.batch_size
            # save only the necessary number to fill all batches
            # Need to define total_samples
            self.ordering = self.ordering[:total_samples]

        # create flag for whether or not to augment a given sample
        self.should_aug = (
            self.rng.random_sample(len(self.ordering)) < augmentation_percentage
        )

    def _create_ordering(self):
        """
        Return an array containing indexes in the range [0, len(X)-1] repeated
        total_samples (or slightly larger than total_samples) times.

        We will shuffle these indexes and use them to randomly select samples from
        the X and Y data in __getitem__. This ensures each index is evenly distributed
        and makes it easy for us to select random samples.
        """
        batch_count = len(self)
        total_samples = batch_count * self.batch_size

        # Multiplier taking into account the number of blank samples to return
        filled_multiplier = total_samples // len(self.X)

        if filled_multiplier * len(self.X) < total_samples:
            filled_multiplier += 1

        # Array with values in [0, len(self.X) - 1] repeated filled_multiplier times
        return np.arange(len(self.X) * filled_multiplier) % len(self.X)

    def shuffle(self):
        """
        If self.seed isn't None then recreate the ordering and shuffle it
        """

        if self.seed is None:
            return

        self.ordering = self._create_ordering()
        self.rng.shuffle(self.ordering)

        batch_count = len(self)
        total_samples = batch_count * self.batch_size

        # save only the necessary number to fill all batches
        # truncate after shuffle to randomize extra elements
        self.ordering = self.ordering[:total_samples]

    def __len__(self):
        """
        Len is the total number of batches that will be returned.
        """

        augmented_len = len(self.X) * self.multiplier

        if augmented_len % self.batch_size == 0:
            return augmented_len // self.batch_size
        else:
            return (augmented_len // self.batch_size) + 1

    def _left_right_systematic_shift(self, a, std):
        """
        Given an array (a) and standard deviation (std), draw a random value shift the array
        (noise) from a Gaussian mean=0 and std=std. Round this value and shift the array
        left/right by noise positions. The first/last values are repeated to generate
        new elements.

        Return the new array shifted left/right.
        """
        noise = round(np.random.normal(0, std))
        if noise >= 0:
            # Cut off first noise elements
            a = a[noise:]
            # Repeat last element noise times
            last_elt = a[-1]
            extra_vals = [last_elt for _ in range(noise)]
            # Append repeated last element to original array
            new_array = np.append(a, extra_vals)
        else:
            # Cut off last noise elements
            a = a[:noise]
            # Repeat first element noise times
            first_elt = a[0]
            new_first_vals = np.array([first_elt for _ in range(noise * -1)])
            # Add repeated first element to original array
            new_array = np.append(new_first_vals, a)
        return new_array

    def _up_down_systematic_shift(self, a, std):
        """
        Given an array (a) and standard deviation (std), draw a number
        from a Gaussian distribution mean=0 and std=std. Add this
        value to every element of a and return the new array.
        """
        systematic_noise = np.random.normal(0, std)
        new_array = a + systematic_noise
        return new_array

    def _up_down_random_shift(self, a, std, scale=0.1):
        """
        Given an array (a) and a standard deviation (std), draw len(a)
        samples from a normal distribution mean=0, std=std. Multiply these
        values by scale and then add them to a to create a new_array.

        Note: We add a scaling parameter because the std likely has a
                meaning we would like to keep (e.g. the std of the mean of
                the first 200 samples). But this number is often too large
                to produce realisitic looking augmented samples, thus we
                scale them down. Default is an order of magnitude.
        """
        random_noise = np.random.normal(0, std, len(a))
        new_array = a + (random_noise * scale)
        return new_array

    def __getitem__(self, idx):
        """
        Each index returns a batch of samples ready to be fed into a model.

        Note: Every instance of sample S on epoch 1 will be either augmented or not.
                  You will not get S being augmented in batch 1 and unaugmented in batch 2.
                  But, we shuffle the ordering at the end of each epoch. Thus in the next
                  epoch, S could be either augmented or not.
        """

        idx_list = self.ordering[idx * self.batch_size : (idx + 1) * self.batch_size]

        X_batch = []
        Y_batch = []
        for idx in idx_list:

            base_X = self.X[idx]
            base_Y = self.Y[idx]

            if self.should_aug[idx]:
                # Augment X
                aug_X = self._left_right_systematic_shift(base_X, self.horizontal_shift)
                aug_X = self._up_down_systematic_shift(aug_X, self.vertical_shift)
                aug_X = self._up_down_random_shift(
                    aug_X, self.noise_shift, self.noise_shift_scale
                )
                # Don't augment Y
                aug_Y = base_Y

                X_batch.append(aug_X)
                Y_batch.append(aug_Y)
            else:
                X_batch.append(base_X)
                Y_batch.append(base_Y)

        X_batch = np.array(X_batch)
        # Reshape 3D for Keras Attention input
        X_batch = np.expand_dims(X_batch, 1)
        Y_batch = np.array(Y_batch)
        return (X_batch, Y_batch)

    def on_epoch_end(self):
        self.shuffle()
