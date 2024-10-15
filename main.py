import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from environment import create_environment
from model import create_q_model
from agent import DQNAgent
from train import train_agent
from utils import plot_metrics
import config 

def main():
    # Create environment
    env = create_environment(config.ENV_NAME, config.SEED)

    # Create main model and target model
    model = create_q_model(env.action_space.n)
    model_target = create_q_model(env.action_space.n)

    # Create DQN agent
    agent = DQNAgent(model, model_target, env.action_space.n)

    # Train the agent
    metrics = train_agent(env, agent)

    # Plot and save metrics
    plot_metrics(metrics)

    # Save the model
    agent.model.save("model/breakout_dqn_cnn_lstm.keras")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
