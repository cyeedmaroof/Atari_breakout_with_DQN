import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack, RecordVideo
import config

def create_environment(env_name=config.ENV_NAME, seed=config.SEED):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, "videos", episode_trigger=lambda x: x % config.RECORD_VIDEO_EVERY == 0)
    env = AtariPreprocessing(env, frame_skip=4, screen_size=config.SCREEN_SIZE, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=False)
    env = FrameStack(env, config.FRAME_STACK)
    env.seed(seed)
    return env
