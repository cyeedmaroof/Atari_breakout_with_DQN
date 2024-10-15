import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
import time

def create_q_model(num_actions):
    inputs = layers.Input(shape=(4, 84, 84, 1))
    x = layers.TimeDistributed(layers.Conv2D(32, 8, strides=4, activation="relu"))(inputs)
    x = layers.TimeDistributed(layers.Conv2D(64, 4, strides=2, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Conv2D(64, 3, strides=1, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.LSTM(512, activation="relu")(x)
    outputs = layers.Dense(num_actions, activation="linear")(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def preprocess_observation(observation):
    return np.expand_dims(observation, axis=0).astype(np.float32) / 255.0

print("Starting script...")

# Create and load the model
num_actions = 4  # For Breakout
model = create_q_model(num_actions)
model.load_weights("models/breakout_dqn_cnn_lstm.h5")
print("Model loaded successfully!")

# Test the model
test_input = tf.random.normal((1, 4, 84, 84, 1))
test_output = model(test_input)
print("Test output shape:", test_output.shape)

print("Creating environment...")
env = gym.make("BreakoutNoFrameskip-v4", render_mode=None)
env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=False)
env = FrameStack(env, 4)
print("Environment created successfully!")

num_episodes = 10

for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}...")
    start_time = time.time()
    
    observation, _ = env.reset()
    print(f"Environment reset for episode {episode + 1}")
    
    episode_reward = 0
    done = False
    steps = 0
    
    while not done:
        if steps % 100 == 0:
            print(f"Episode {episode + 1}, Step {steps}")
        
        state = preprocess_observation(observation)
        action_values = model(state)
        action = tf.argmax(action_values[0]).numpy()
        
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        steps += 1
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Episode {episode + 1} finished with reward {episode_reward} in {steps} steps")
    print(f"Episode duration: {duration:.2f} seconds")

env.close()
print("All episodes completed!")