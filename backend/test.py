import random

# grid
H = 3
W = 4

# start and goal to reach
START = (2, 0)
GOAL = (0, 3)

# obstacle 
WALLS = {(1, 1)}

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
        return ((r,c), 10, True)

    return ((r,c), STEP_REWARD, False)

def generate_episode():
    state = START
    trajectory = []
    for t in range(MAX_STEPS):
        action = random.choice(ACTIONS)
        ns, r, d = step(state, action)
        trajectory.append((state, action, r))
        state = ns
        if d:
            break
    
    return trajectory

def calculate_returns(trajectory):
    returns = [0.0] * len(trajectory)
    G = 0.0

    for i in range(len(trajectory)-1, -1, -1):
        reward = trajectory[i][2]
        G = reward + GAMMA * G
        returns[i] = G

    return returns

traj = generate_episode()
rets = calculate_returns(traj)


