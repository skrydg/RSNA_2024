import numpy as np
import tensorflow as tf

class MeanLogLoss(tf.keras.Loss):
    def __init__(self, columns=[], **kwargs):
        super(MeanLogLoss, self).__init__(**kwargs)
        self.columns = columns
        self.conditions = ['spinal', 'foraminal', 'subarticular']
        self.w = np.array([1, 2, 4])
        
        self.masks = [[(condition in column) for column in self.columns] for condition in self.conditions]
        self.spinal_mask = [("spinal" in column) for column in self.columns]
        self.log_loss = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [y_true.shape[0], y_true.shape[1] // 3, 3])
        y_pred = tf.reshape(y_pred, [y_pred.shape[0], y_pred.shape[1] // 3, 3])
        
        sample_weight = tf.math.multiply(y_true, self.w[np.newaxis, np.newaxis, :])
        sample_weight = tf.math.reduce_sum(sample_weight, axis=-1)
        
        log_losses = []
        for mask in self.masks:
            current_y_true = tf.boolean_mask(y_true, mask, axis=1)
            current_y_pred = tf.boolean_mask(y_pred, mask, axis=1)
            current_sample_weight = tf.boolean_mask(sample_weight, mask, axis=1)
            
            new_shape = [current_y_true.shape[0] * current_y_true.shape[1], 3]
            current_y_true = tf.reshape(current_y_true, new_shape)
            current_y_pred = tf.reshape(current_y_pred, new_shape)
            current_sample_weight = tf.reshape(current_sample_weight, [new_shape[0]])
            current_log_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")(
                current_y_true,
                current_y_pred,
            )
            current_log_loss = tf.math.reduce_sum(tf.math.multiply(current_log_loss, current_sample_weight), axis=0) / tf.math.reduce_sum(current_sample_weight)
            log_losses.append(current_log_loss)
            
        sever_spinal_y_true = y_true[:, :, 2:3]
        sever_spinal_y_true = tf.boolean_mask(sever_spinal_y_true, self.spinal_mask, axis=1)
        sever_spinal_y_true = tf.reduce_max(sever_spinal_y_true, axis=1)
        
        sever_spinal_y_predicted = y_pred[:, :, 2:3]
        sever_spinal_y_predicted = tf.boolean_mask(sever_spinal_y_predicted, self.spinal_mask, axis=1)
        sever_spinal_y_predicted = tf.reduce_max(sever_spinal_y_predicted, axis=1)
        
        sever_spinal_sample_weight = sample_weight
        sever_spinal_sample_weight = tf.boolean_mask(sever_spinal_sample_weight, self.spinal_mask, axis=1)
        sever_spinal_sample_weight = tf.reduce_max(sever_spinal_sample_weight, axis=1)
        
        sever_spinal_log_loss = tf.keras.losses.BinaryCrossentropy(reduction=None)(
            sever_spinal_y_true,
            sever_spinal_y_predicted,
        )
        sever_spinal_log_loss = tf.math.reduce_sum(tf.math.multiply(sever_spinal_log_loss, sever_spinal_sample_weight), axis=0) / tf.math.reduce_sum(sever_spinal_sample_weight)
        log_losses.append(sever_spinal_log_loss)

        return tf.reduce_mean(log_losses)