import tensorflow as tf

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator


class Random4dImageTranslation(TFDataLayer):
    _FACTOR_VALIDATION_ERROR = (
        "The `factor` argument should be a number (or a list of two numbers) "
        "in the range [-1.0, 1.0]. "
    )
    _SUPPORTED_FILL_MODE = ("reflect", "wrap", "constant", "nearest")
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.height_factor = height_factor
        self.height_lower, self.height_upper = self._set_factor(
            height_factor, "height_factor"
        )
        self.width_factor = width_factor
        self.width_lower, self.width_upper = self._set_factor(
            width_factor, "width_factor"
        )

        if fill_mode not in self._SUPPORTED_FILL_MODE:
            raise NotImplementedError(
                f"Unknown `fill_mode` {fill_mode}. Expected of one "
                f"{self._SUPPORTED_FILL_MODE}."
            )
        if interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.data_format = backend.standardize_data_format(data_format)
        self.supports_jit = False

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

    def _set_factor(self, factor, factor_name):
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(
                    self._FACTOR_VALIDATION_ERROR
                    + f"Received: {factor_name}={factor}"
                )
            self._check_factor_range(factor[0])
            self._check_factor_range(factor[1])
            lower, upper = sorted(factor)
        elif isinstance(factor, (int, float)):
            self._check_factor_range(factor)
            factor = abs(factor)
            lower, upper = [-factor, factor]
        else:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR
                + f"Received: {factor_name}={factor}"
            )
        return lower, upper

    def _check_factor_range(self, input_number):
        if input_number > 1.0 or input_number < -1.0:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR
                + f"Received: input_number={input_number}"
            )

    def call(self, inputs, training=True):
        inputs = self.backend.cast(inputs, self.compute_dtype)
        if training:
            return self._randomly_translate_inputs(inputs)
        else:
            return inputs

    def _randomly_translate_inputs(self, inputs):
        inputs_shape = self.backend.shape(inputs)

        batch_size, image_depth, image_height, image_width, image_channel = self._get_image_shapes(inputs)
        
        seed_generator = self._get_seed_generator(self.backend._backend)
        height_translate = self.backend.random.uniform(
            minval=self.height_lower,
            maxval=self.height_upper,
            shape=[batch_size, 1],
            seed=seed_generator,
        )
        height_translate = self.backend.numpy.multiply(height_translate, image_height)
        
        width_translate = self.backend.random.uniform(
            minval=self.width_lower,
            maxval=self.width_upper,
            shape=[batch_size, 1],
            seed=seed_generator,
        )
        width_translate = self.backend.numpy.multiply(width_translate, image_width)
        height_translate = tf.repeat(height_translate, image_depth, axis=0)
        width_translate = tf.repeat(width_translate, image_depth, axis=0)
        
        translations = self.backend.cast(
            self.backend.numpy.concatenate(
                [width_translate, height_translate], axis=1
            ),
            dtype="float32",
        )

        inputs_2d = tf.reshape(inputs, [batch_size * image_depth, image_height, image_width, image_channel])
        outputs = self.backend.image.affine_transform(
            inputs_2d,
            transform=self._get_translation_matrix(translations),
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

        return tf.reshape(outputs, inputs_shape)

    def _get_translation_matrix(self, translations):
        num_translations = self.backend.shape(translations)[0]
        return self.backend.numpy.concatenate(
            [
                self.backend.numpy.ones((num_translations, 1)),
                self.backend.numpy.zeros((num_translations, 1)),
                -translations[:, 0:1],
                self.backend.numpy.zeros((num_translations, 1)),
                self.backend.numpy.ones((num_translations, 1)),
                -translations[:, 1:],
                self.backend.numpy.zeros((num_translations, 2)),
            ],
            axis=1,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "interpolation": self.interpolation,
            "seed": self.seed,
            "fill_value": self.fill_value,
            "data_format": self.data_format,
        }
        return {**base_config, **config}