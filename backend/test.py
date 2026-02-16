import random
import matplotlib.pyplot as plt
import numpy as np

N = 1000

# grid
H = 3
W = 5

# start and goal to reach
START = (2, 0)
GOAL = (2, 4)

# obstacle 
WALLS = {(1, 2), (2, 2)}

ACTIONS = ["U", "D", "L", "R"]

STEP_REWARD = -1.0
GOAL_REWARD = 10.0

GAMMA = 0.9
MAX_STEPS = 50

def is_valid(cell):
    if 0 <= cell[0] < H and 0 <= cell[1] < W and cell not in WALLS:
        return True

    return False
    
def step(state, action):
    if action not in ACTIONS:
        raise ValueError(f"not an option for action")
    r, c = state
    if (r,c) == GOAL:
        return ((r,c), 0.0, True)

    if action == "U" and is_valid((r - 1, c)):
        r, c = r - 1, c
    elif action == "D" and is_valid((r + 1, c)):
        r, c = r + 1, c
    elif action == "L" and is_valid((r, c - 1)):
        r, c = r, c - 1
    elif action == "R" and is_valid((r, c + 1)):
        r, c = r, c + 1

    if (r,c) == GOAL:
        return ((r,c), GOAL_REWARD, True)

    return ((r,c), STEP_REWARD, False)


def choose_action_eps_greedy(Q, state, epsilon):
    action_values = Q[state]
    max_q = max(action_values.values())
    best_actions = [a for a, q in action_values.items() if q == max_q]

    if random.random() < epsilon:
        return random.choice(ACTIONS)
    
    return random.choice(best_actions)

def generate_episode(Q, epsilon):
    state = START
    trajectory = []
    episode_return = 0
    for t in range(MAX_STEPS):
        if state == GOAL:
            break
        action = choose_action_eps_greedy(Q, state, epsilon)
        ns, r, d = step(state, action)
        episode_return += r
        trajectory.append((state, action, r))
        state = ns
        if d:
            break
    
    return (trajectory, episode_return)

# only for debugging purposes
def calculate_returns(trajectory):
    returns = [0.0] * len(trajectory)
    G = 0.0

    for i in range(len(trajectory)-1, -1, -1):
        reward = trajectory[i][2]
        G = reward + GAMMA * G
        returns[i] = G

    return returns

def all_states():
    for r in range(H):
        for c in range(W):
            s = (r,c)
            if s not in WALLS:
                yield s
        
Q = {
    s: {a: 0.0 for a in ACTIONS}
    for s in all_states()
    if s != GOAL
}

def update_q(Q, trajectory, alpha):
    G = 0.0
    for i in reversed(range(len(trajectory))):
        state, action, reward = trajectory[i]
        G = reward + GAMMA * G
        Q[state][action] += alpha * (G-Q[state][action])

def greedy_action(Q, state):
    action_values = Q[state]
    max_q = max(action_values.values())
    best_actions = [a for a, q in action_values.items() if q == max_q]
    return random.choice(best_actions)

alpha = 0.1
reward_history = []
for episode in range(N):
    epsilon = max(0.05, 1.0 - episode / N)
    traj, reward = generate_episode(Q, epsilon)
    reward_history.append(reward)
    update_q(Q, traj, alpha)
    if (episode + 1) % 100 == 0:
        recent_mean = sum(reward_history[-100:]) / len(reward_history[-100:])
        print(f"episode {episode + 1}: {recent_mean:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
window = 50
moving_avg = [np.mean(reward_history[max(0, i-window):i+1]) for i in range(len(reward_history))]
ax.plot(moving_avg)
ax.set_xlabel('Episode')
ax.set_ylabel('Moving Average Return')
ax.set_title('Learning Curve')
ax.grid(True)
plt.tight_layout()
plt.show()
