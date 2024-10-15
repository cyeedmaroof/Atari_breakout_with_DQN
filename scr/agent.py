import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import Huber
import config

class DQNAgent:
    def __init__(self, model, model_target, num_actions):
        self.model = model
        self.model_target = model_target
        self.num_actions = num_actions
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_MAX
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY_FRAMES
        self.optimizer = Adam(learning_rate=config.LEARNING_RATE, clipnorm=config.CLIPNORM)
        self.loss_function = Huber()

    def get_action(self, state, frame_count):
        if frame_count < config.EPSILON_RANDOM_FRAMES or self.epsilon > np.random.rand(1)[0]:
            return np.random.choice(self.num_actions)
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            return tf.argmax(action_probs[0]).numpy()

    def train_step(self, state_sample, state_next_sample, action_sample, rewards_sample, done_sample):
        future_rewards = self.model_target.predict(state_next_sample)
        updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

        masks = tf.one_hot(action_sample, self.num_actions)

        with tf.GradientTape() as tape:
            q_values = self.model(state_sample)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_function(updated_q_values, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def update_target_network(self):
        self.model_target.set_weights(self.model.get_weights())

    def update_epsilon(self, frame_count):
        self.epsilon -= (config.EPSILON_MAX - config.EPSILON_MIN) / self.epsilon_decay
        self.epsilon = max(self.epsilon, config.EPSILON_MIN)
