import matplotlib.pyplot as plt

def plot_metrics(metrics):
    plt.figure(figsize=(15, 5))

    # Episode Rewards
    plt.subplot(131)
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Running Rewards
    plt.subplot(132)
    plt.plot(metrics['running_rewards'])
    plt.title('Running Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Running Reward (Avg of last 100)')

    # Loss
    plt.subplot(133)
    plt.plot(metrics['losses'])
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('dqn_performance_with_loss.png')
    plt.show()
