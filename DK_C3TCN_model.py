import tensorflow as tf
from keras import Model, layers, Input, callbacks
from keras.optimizers import Nadam
from tcn import TCN  # Ensure this package is installed: pip install keras-tcn
from custom_flood_loss import custom_flood_loss


# Spatial Attention Layer
class SpatialAttention(layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = layers.Conv1D(1, kernel_size=1, activation='sigmoid', padding='same')

    def call(self, inputs):
        attention_outputs = []
        for input_tensor in inputs:
            attention_weights = self.conv1(input_tensor)
            attention_outputs.append(input_tensor * attention_weights)
        return tf.concat(attention_outputs, axis=-1)


# Temporal Attention Layer
class TemporalAttention(layers.Layer):
    def __init__(self):
        super(TemporalAttention, self).__init__()

    def build(self, input_shape):
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(1, activation='linear')
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        temporal_features = self.dense1(inputs)
        attention_weights = self.dense2(temporal_features)
        attention_weights = tf.squeeze(attention_weights, axis=-1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        return tf.reduce_sum(inputs * attention_weights, axis=-2)


# Define model-building function
def DK_C3TCN_model(n_steps_in, n_steps_out, n_features):
    input_shape = (n_steps_in, n_features)
    inputs = Input(shape=input_shape)

    # CNN feature extractor
    conv1 = layers.Conv1D(filters=128, kernel_size=3, activation="relu")(inputs)
    conv1 = layers.Conv1D(filters=128, kernel_size=3, activation="relu")(conv1)
    conv1 = layers.MaxPooling1D(pool_size=1)(conv1)
    conv1 = layers.Flatten()(conv1)

    # Spatial Attention + TCN
    sa_output = SpatialAttention()([inputs, inputs, inputs])
    sa_output = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(sa_output)
    tcn_output = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8],
                     return_sequences=True, activation='relu')(sa_output)

    # Temporal Attention
    ta_output = TemporalAttention()(tcn_output)

    # Combine CNN and Attention outputs
    concat = layers.concatenate([conv1, ta_output])

    # Dense Output
    outputs = layers.Dense(units=n_steps_out)(concat)

    # Model and Compilation
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=custom_flood_loss)

    return model
