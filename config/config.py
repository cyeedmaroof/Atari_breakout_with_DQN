# Environment settings
ENV_NAME = "BreakoutNoFrameskip-v4"
SEED = 42

# Model parameters
FRAME_STACK = 4
SCREEN_SIZE = 84

# Training hyperparameters
MAX_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 100000
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor for past rewards
EPSILON_MAX = 1.0  # Maximum epsilon greedy parameter
EPSILON_MIN = 0.1  # Minimum epsilon greedy parameter
EPSILON_RANDOM_FRAMES = 50000  # Number of frames to take random action and observe output
EPSILON_DECAY_FRAMES = 1000000.0  # Number of frames over which to decay epsilon

# Agent parameters
LEARNING_RATE = 0.00025
CLIPNORM = 1.0

# Replay buffer
MAX_MEMORY_LENGTH = 150000
UPDATE_AFTER_ACTIONS = 20
UPDATE_TARGET_NETWORK = 10000

# Solve criteria
SOLVE_CRITERIA = 15  # Running reward to consider the environment solved

# Visualization
RECORD_VIDEO_EVERY = 1000  # Record video every N episodes
