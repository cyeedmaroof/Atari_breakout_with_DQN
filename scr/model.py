from keras import layers, Model
import config

def create_q_model(num_actions):
    inputs = layers.Input(shape=(config.FRAME_STACK, config.SCREEN_SIZE, config.SCREEN_SIZE, 1))
    
    x = layers.TimeDistributed(layers.Conv2D(32, 8, strides=4, activation="relu"))(inputs)
    x = layers.TimeDistributed(layers.Conv2D(64, 4, strides=2, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Conv2D(64, 3, strides=1, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    x = layers.LSTM(512, activation="relu")(x)
    
    outputs = layers.Dense(num_actions, activation="linear")(x)
    
    return Model(inputs=inputs, outputs=outputs)
