import numpy as np

# Maze setup
maze = [
    [0, 0, 0, 1],  # 1 represents the goal
    [0, -1, 0, -1],  # -1 represents walls
    [0, 0, 0, 0]
]

# Parameters
num_states = len(maze) * len(maze[0])
num_actions = 4  # up, down, left, right
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000

#todo :1. immediate : understand qlearning and push this. 
#todo : 2. make maze that autoruns/does something automatically
#todo 3. check how this maps to that maze


# Initialize Q-table
Q = np.zeros((num_states, num_actions))

# Map states to coordinates
state_to_coords = {(i * len(maze[0]) + j): (i, j) for i in range(len(maze)) for j in range(len(maze[0]))}
print(state_to_coords)
# Map coordinates to state
coords_to_state = {v: k for k, v in state_to_coords.items()}

# Actions
actions = ["up", "down", "left", "right"]
action_effects = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# Reward function
def get_reward(state):
    i, j = state_to_coords[state]
    if maze[i][j] == 1:
        return 10
    elif maze[i][j] == -1:
        return -100
    else:
        return -1

# Next state function
def get_next_state(state, action):
    i, j = state_to_coords[state]
    di, dj = action_effects[action]
    ni, nj = i + di, j + dj
    if 0 <= ni < len(maze) and 0 <= nj < len(maze[0]) and maze[ni][nj] != -1:
        return coords_to_state[(ni, nj)]
    else:
        return state  # Stay in the same state if hitting a wall

# Q-learning algorithm
for _ in range(episodes):
    state = np.random.randint(0, num_states)  # Start in a random state
    
    while True:
        if state_to_coords[state] == (0, 3):  # Goal state
            break
        
        # Choose action (ε-greedy)
        if np.random.rand() < epsilon:
            action = np.random.choice(range(num_actions))
        else:
            action = np.argmax(Q[state])
        
        # Take action
        next_state = get_next_state(state, actions[action])
        reward = get_reward(next_state)

        print('current state ', state, ' action ', actions[action], ' reward ', reward)
        
        # Update Q-value
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        state = next_state  # Move to next state

print('Q = ', Q)