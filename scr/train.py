import numpy as np
from collections import deque
import config

def train_agent(env, agent):
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = deque(maxlen=100)
    running_reward = 0
    episode_count = 0
    frame_count = 0

    episode_rewards = []
    running_rewards = []
    losses = []

    while True:
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []

        for timestep in range(1, config.MAX_STEPS_PER_EPISODE):
            frame_count += 1

            action = agent.get_action(state, frame_count)
            agent.update_epsilon(frame_count)

            state_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward

            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            if frame_count % config.UPDATE_AFTER_ACTIONS == 0 and len(done_history) > config.BATCH_SIZE:
                indices = np.random.choice(range(len(done_history)), size=config.BATCH_SIZE)

                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = np.array([float(done_history[i]) for i in indices])

                loss = agent.train_step(state_sample, state_next_sample, action_sample, rewards_sample, done_sample)
                episode_loss.append(loss)

            if frame_count % config.UPDATE_TARGET_NETWORK == 0:
                agent.update_target_network()

            if len(rewards_history) > config.MAX_MEMORY_LENGTH:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        episode_reward_history.append(episode_reward)
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        episode_rewards.append(episode_reward)
        running_rewards.append(running_reward)
        losses.append(np.mean(episode_loss))

        print(f"Episode {episode_count}: Reward = {episode_reward:.2f}, Running Reward = {running_reward:.2f}")

        if running_reward > config.SOLVE_CRITERIA:
            print(f"Solved at episode {episode_count}!")
            break

        if config.MAX_EPISODES > 0 and episode_count >= config.MAX_EPISODES:
            print(f"Stopped at episode {episode_count}!")
            break

    return {'episode_rewards': episode_rewards, 'running_rewards': running_rewards, 'losses': losses}
