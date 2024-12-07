import numpy as np
import gymnasium as gym
import minigrid
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
# import gymnasium_minigrid

# Parameters
env_name = 'MiniGrid-Empty-5x5-v0'
num_episodes = 1000
max_steps_per_episode = 100
alpha = 0.1       # Learning rate
gamma = 0.99      # Discount factor
epsilon = 1.0     # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999  # Decay per episode

from minigrid.wrappers import ImgObsWrapper

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)
env = gym.make("MiniGrid-Empty-16x16-v0", render_mode=None)
env = ImgObsWrapper(env)
num_actions = env.action_space.n
# Q-Table: Dictionary keyed by state (image) tuples
Q = {}

def get_state_key(observation):
    # observation is a 3D array (C, H, W)
    # Convert to a tuple so it can be used as a dictionary key
    return tuple(observation.flatten())

def get_action(state_key):
    # Epsilon-greedy action selection
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        if state_key not in Q:
            Q[state_key] = np.zeros(num_actions)
        return np.argmax(Q[state_key])

for episode in range(num_episodes):
    obs, info = env.reset()
    state_key = get_state_key(obs)
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = get_action(state_key)
        next_obs, reward, done, truncated, info = env.step(action)
        next_state_key = get_state_key(next_obs)

        if state_key not in Q:
            Q[state_key] = np.zeros(num_actions)
        if next_state_key not in Q:
            Q[next_state_key] = np.zeros(num_actions)

        # Q-Learning Update
        best_next_action = np.argmax(Q[next_state_key])
        td_target = reward + gamma * Q[next_state_key][best_next_action] * (0 if done else 1)
        td_error = td_target - Q[state_key][action]
        Q[state_key][action] += alpha * td_error

        state_key = next_state_key
        total_reward += reward

        if done:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()