import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
import minigrid  # Make sure gymnasium-minigrid is installed

# 1. Create and unwrap the environment
env = gym.make("MiniGrid-Empty-6x6-v0")  # or render_mode="rgb_array" / "human"
env = env.unwrapped  # Unwrap so that env.agent_pos & env.agent_dir are accessible

# 2. Define Q-learning parameters
alpha = 0.1            # Learning rate
gamma = 0.9            # Discount factor
epsilon = 1.0          # Exploration rate
epsilon_decay_rate = 0.00005  # Exploration decay rate
min_epsilon = 0.1      # Minimum exploration rate
num_episodes = 5000    # Number of episodes for training

# Initialize an empty Q-table, mapping states to action-value arrays
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

def get_state(obs):
    """
    Extracts the state from the environment.
    We directly use env.unwrapped.agent_pos and agent_dir.
    """
    agent_pos = env.agent_pos        # (row, col) of the agent
    agent_dir = env.agent_dir        # integer 0-3 representing orientation
    # Make a simple tuple (row, col, dir).
    return (agent_pos[0], agent_pos[1], agent_dir)

def select_action(state, epsilon):
    """Select an action with epsilon-greedy policy."""
    if random.random() < epsilon:
        # Explore
        return env.action_space.sample()
    else:
        # Exploit
        return np.argmax(q_table[state])

def update_q_table(state, action, reward, next_state):
    """Update the Q-table using the Q-learning update rule."""
    best_next_q = np.max(q_table[next_state])
    q_table[state][action] = (1 - alpha) * q_table[state][action] + \
        alpha * (reward + gamma * best_next_q)

def train():
    """Train the Q-learning agent."""
    global epsilon
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = get_state(obs)
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action = select_action(state, epsilon)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = get_state(obs)

            update_q_table(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)

        # Print progress
        if episode % 100 == 0:
            print(f"Episode: {episode}, Epsilon: {epsilon:.3f}, Reward: {total_reward}")
            # Check if "solved" (very naive check)
            if total_reward > 0.95:  
                print("Environment solved!")
                break

    print("Training finished.")
    return q_table

def play(q_table, num_episodes=5, render=True):
    """
    Play using the trained Q-table.
    If render=True, set the environment's render_mode to human.
    """
    # Re-make the environment with a human render (optional for visuals)
    if render:
        # We create a brand new environment in human mode just for testing
        play_env = gym.make("MiniGrid-Empty-6x6-v0", render_mode="human").unwrapped
    else:
        play_env = gym.make("MiniGrid-Empty-6x6-v0").unwrapped

    for episode in range(num_episodes):
        obs, info = play_env.reset()
        state = (play_env.agent_pos[0], play_env.agent_pos[1], play_env.agent_dir)
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            if render:
                play_env.render()
            action = np.argmax(q_table[state])  # Exploit
            obs, reward, terminated, truncated, info = play_env.step(action)

            # Update state
            state = (play_env.agent_pos[0], play_env.agent_pos[1], play_env.agent_dir)
            total_reward += reward

        if render:
            play_env.render()
        print(f"Test Episode: {episode}, Reward: {total_reward}")

    play_env.close()

if __name__ == "__main__":
    trained_q_table = train()
    play(trained_q_table, num_episodes=5, render=True)