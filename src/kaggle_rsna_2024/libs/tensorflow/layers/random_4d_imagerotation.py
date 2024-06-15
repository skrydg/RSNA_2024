import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator


class Random4DImageRotation(TFDataLayer):
    _FACTOR_VALIDATION_ERROR = (
        "The `factor` argument should be a number (or a list of two numbers) "
        "in the range [-1.0, 1.0]. "
    )
    _VALUE_RANGE_VALIDATION_ERROR = (
        "The `value_range` argument should be a list of two numbers. "
    )

    _SUPPORTED_FILL_MODE = ("reflect", "wrap", "constant", "nearest")
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
        self,
        factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        value_range=(0, 255),
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self._set_factor(factor)
        self._set_value_range(value_range)
        self.data_format = backend.standardize_data_format(data_format)
        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.fill_value = fill_value

        self.supports_jit = False

        if self.fill_mode not in self._SUPPORTED_FILL_MODE:
            raise NotImplementedError(
                f"Unknown `fill_mode` {fill_mode}. Expected of one "
                f"{self._SUPPORTED_FILL_MODE}."
            )
        if self.interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

    def _set_value_range(self, value_range):
        if not isinstance(value_range, (tuple, list)):
            raise ValueError(
                self.value_range_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        if len(value_range) != 2:
            raise ValueError(
                self.value_range_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        self.value_range = sorted(value_range)

    def _set_factor(self, factor):
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(
                    self._FACTOR_VALIDATION_ERROR + f"Received: factor={factor}"
                )
            self._check_factor_range(factor[0])
            self._check_factor_range(factor[1])
            self._factor = sorted(factor)
        elif isinstance(factor, (int, float)):
            self._check_factor_range(factor)
            factor = abs(factor)
            self._factor = [-factor, factor]
        else:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR + f"Received: factor={factor}"
            )

    def _check_factor_range(self, input_number):
        if input_number > 1.0 or input_number < -1.0:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR
                + f"Received: input_number={input_number}"
            )

    def _get_image_shapes(self, inputs):
        shape = self.backend.core.shape(inputs)
        if len(shape) == 5:
                batch_size = shape[0]
                image_depth = shape[1]
                image_height = shape[2]
                image_width = shape[3]
                image_channel = shape[4]
        else:
            assert(len(shape) == 4)
            batch_size = 1
            image_depth = shape[0]
            image_height = shape[1]
            image_width = shape[2]
            image_channel = shape[3]

        return batch_size, image_depth, image_height, image_width, image_channel
    
    def _get_rotation_matrix(self, inputs):
        shape = self.backend.core.shape(inputs)
        batch_size, image_depth, image_height, image_width, image_channel = self._get_image_shapes(inputs)


        lower = self._factor[0] * 2.0 * self.backend.convert_to_tensor(np.pi)
        upper = self._factor[1] * 2.0 * self.backend.convert_to_tensor(np.pi)

        seed_generator = self._get_seed_generator(self.backend._backend)
        angle = self.backend.random.uniform(
            shape=(batch_size,),
            minval=lower,
            maxval=upper,
            seed=seed_generator,
        )
        angle = tf.repeat(angle, image_depth)
        batch_size = batch_size * image_depth

        cos_theta = self.backend.numpy.cos(angle)
        sin_theta = self.backend.numpy.sin(angle)
        image_height = self.backend.core.cast(image_height, cos_theta.dtype)
        image_width = self.backend.core.cast(image_width, cos_theta.dtype)

        x_offset = (
            (image_width - 1)
            - (cos_theta * (image_width - 1) - sin_theta * (image_height - 1))
        ) / 2.0

        y_offset = (
            (image_height - 1)
            - (sin_theta * (image_width - 1) + cos_theta * (image_height - 1))
        ) / 2.0

        outputs = self.backend.numpy.concatenate(
            [
                self.backend.numpy.cos(angle)[:, None],
                -self.backend.numpy.sin(angle)[:, None],
                x_offset[:, None],
                self.backend.numpy.sin(angle)[:, None],
                self.backend.numpy.cos(angle)[:, None],
                y_offset[:, None],
                self.backend.numpy.zeros((batch_size, 2)),
            ],
            axis=1,
        )
        if len(shape) == 3:
            outputs = self.backend.numpy.squeeze(outputs, axis=0)
        return outputs

    def call(self, inputs, training=True):
        inputs = self.backend.cast(inputs, self.compute_dtype)
        if training:
            shape = self.backend.core.shape(inputs)
            rotation_matrix = self._get_rotation_matrix(inputs)
            batch_size, image_depth, image_height, image_width, image_channel = self._get_image_shapes(inputs)
            inputs_2d = tf.reshape(inputs, [batch_size * image_depth, image_height, image_width, image_channel])
            transformed_image_2d = self.backend.image.affine_transform(
                image=inputs_2d,
                transform=rotation_matrix,
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
                data_format=self.data_format,
            )
            return tf.reshape(transformed_image_2d, shape)
            
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self._factor,
            "value_range": self.value_range,
            "data_format": self.data_format,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}