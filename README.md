# Deep Q-Learning for Atari Breakout

This project implements a Deep Q-Network (DQN) agent to play the Atari Breakout game. It uses a convolutional neural network (CNN) combined with an LSTM layer to process the game frames and make decisions.

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/cyeedmaroof/Atari_breakout_with_DQN.git
   cd Atari_breakout_with_DQN
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: This will install TensorFlow, Keras, Gymnasium with Atari support, and other necessary packages.

3. Ensure you have the trained model weights file `breakout_dqn_cnn_lstm.h5` in the project directory.

## Project Structure

```bash
Atari_breakout_with_DQN/
│
├── .vscode/                # VSCode configuration files
│   └── settings.json        # Editor settings for the project
│
├── config/                 # Configuration files
│   └── config.py           # Hyperparameters and settings
│
├── models/                 # Directory to store trained model files
│   └── [model files]       # Saved model weights in .h5 format
│
├── src/                    # Source code for the project
│   └── agent.py            # DQN agent implementation
│   └── environment.py      # Atari Breakout environment setup
│   └── model.py            # Neural network model definition
│   └── train.py            # Training loop and logic
│
├── utils/                  # Utility functions
│   └── utils.py            # Helper functions for training and logging
│
├── load_and_test_model.py   # Script to load and test the saved model
├── main.py                  # Main entry point for training the agent
├── README.md                # Project documentation (this file)
└── requirements.txt         # Python dependencies


- `main.py`: The entry point of the program, orchestrating the entire process
- `environment.py`: Contains functions to create and configure the Atari environment
- `model.py`: Defines the Q-network architecture
- `agent.py`: Implements the DQN agent with its core functionalities
- `train.py`: Contains the main training loop and logic
- `utils.py`: Includes utility functions, specifically for plotting metrics
- `config.py`: Contains all hyperparameters and configuration settings
- `run_episodes_capped.py`: Script to run episodes with the trained agent
- `requirements.txt`: List of Python package dependencies
- `breakout_dqn_cnn_lstm.h5`: Trained model weights (not included in the repository)


### Training a New Agent

To train a new agent, you would typically use the following files:

1. Modify `config.py` to set your desired hyperparameters.
2. Run `main.py` to start the training process:
   ```
   python main.py
   ```

This will use `environment.py` to set up the game, `model.py` to create the neural network, `agent.py` to create the DQN agent, and `train.py` to run the training loop. `utils.py` will be used for plotting results.

### Evaluating a Trained Agent

To watch the trained agent play Breakout:

```
python load_and_test_model.py
```

This script will:
- Load the trained model
- Run 10 episodes of Breakout
- Print out rewards and steps for each episode

## Modifying the Scripts

- `config.py`: Adjust hyperparameters like learning rate, epsilon values, etc.
- `model.py`: Modify the neural network architecture if you want to experiment with different structures.
- `agent.py`: Adjust the agent's behavior, reward processing, or add new features like prioritized experience replay.
- `train.py`: Modify the training loop, add checkpoints, or implement different training regimes.
- `run_episodes_capped.py`: Change `num_episodes` or `max_steps_per_episode` to adjust evaluation runs.

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are correctly installed
2. Check that the model weights file is in the correct location
3. Make sure you have sufficient CPU/GPU resources
4. Verify that all imported modules are in the same directory or in your Python path


