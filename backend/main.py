import random
from typing import Dict, List, Tuple
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

State = Tuple[int, int]
Action = str

# env config
H, W = 3, 5
START: State = (2, 0)
GOAL: State = (2, 4)
WALLS = {(1, 2), (2, 2)}
ACTIONS: List[Action] = ["U", "D", "L", "R"]

STEP_REWARD = -1.0
GOAL_REWARD = 10.0
GAMMA = 0.9
MAX_STEPS = 50

# rl helper functions
def is_valid(cell: State) -> bool:
    r, c = cell
    return 0 <= r < H and 0 <= c < W and cell not in WALLS

def step(state: State, action: Action):
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    r, c = state
    if (r, c) == GOAL:
        return (state, 0.0, True)

    nr, nc = r, c
    if action == "U":
        nr, nc = r - 1, c
    elif action == "D":
        nr, nc = r + 1, c
    elif action == "L":
        nr, nc = r, c - 1
    elif action == "R":
        nr, nc = r, c + 1

    if is_valid((nr, nc)):
        r, c = nr, nc

    if (r, c) == GOAL:
        return ((r, c), GOAL_REWARD, True)

    return ((r, c), STEP_REWARD, False)

def choose_action_eps_greedy(Q: Dict[State, Dict[Action, float]], state: State, epsilon: float) -> Action:
    action_values = Q[state]
    max_q = max(action_values.values())
    best_actions = [a for a, q in action_values.items() if q == max_q]

    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return random.choice(best_actions)

def all_states():
    for r in range(H):
        for c in range(W):
            s = (r, c)
            if s not in WALLS:
                yield s

def init_q() -> Dict[State, Dict[Action, float]]:
    return {s: {a: 0.0 for a in ACTIONS} for s in all_states() if s != GOAL}

def generate_episode(Q, epsilon: float):
    state = START
    trajectory = []
    ep_return = 0.0

    for _ in range(MAX_STEPS):
        if state == GOAL:
            break
        action = choose_action_eps_greedy(Q, state, epsilon)
        ns, r, done = step(state, action)
        trajectory.append((state, action, r))
        ep_return += r
        state = ns
        if done:
            break

    return trajectory, ep_return

def update_q(Q, trajectory, alpha: float):
    G = 0.0
    for i in range(len(trajectory) - 1, -1, -1):
        state, action, reward = trajectory[i]
        G = reward + GAMMA * G
        Q[state][action] += alpha * (G - Q[state][action])

def greedy_action(Q, state: State) -> Action:
    action_values = Q[state]
    max_q = max(action_values.values())
    best_actions = [a for a, q in action_values.items() if q == max_q]
    return random.choice(best_actions)

class Trainer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.Q = init_q()
        self.episode = 0
        self.reward_history: List[float] = []
        self.eval_history: List[Tuple[int, float]] = []

    def epsilon_schedule(self, episode_idx: int, eps_min=0.05):
        return max(eps_min, 1.0 - episode_idx / 1000.0)

    def evaluate_greedy(self, n_eval: int):
        total_return = 0.0
        for _ in range(n_eval):
            _, episode_return = generate_episode(self.Q, epsilon=0.0)
            total_return += episode_return
        return total_return / n_eval

    def train(self, n: int, alpha: float, eval_every: int = 50, n_eval: int = 20):
        for _ in range(n):
            eps = self.epsilon_schedule(self.episode)
            traj, r = generate_episode(self.Q, eps)
            update_q(self.Q, traj, alpha)
            self.reward_history.append(r)
            self.episode += 1
            if self.episode % eval_every == 0:
                self.eval_history.append((self.episode, self.evaluate_greedy(n_eval)))

    def snapshot(self):
        policy = {}
        for r in range(H):
            for c in range(W):
                s = (r, c)
                if s in WALLS:
                    policy[f"{r},{c}"] = None
                elif s == GOAL:
                    policy[f"{r},{c}"] = "G"
                else:
                    policy[f"{r},{c}"] = greedy_action(self.Q, s)

        Q_json = {f"{s[0]},{s[1]}": self.Q[s] for s in self.Q}

        return {
            "grid": {"H": H, "W": W, "start": list(START), "goal": list(GOAL), "walls": [list(w) for w in WALLS]},
            "episode": self.episode,
            "epsilon": self.epsilon_schedule(self.episode),
            "Q": Q_json,
            "policy": policy,
            "reward_history": self.reward_history,
            "eval_history": [
                {"episode": ep, "avg_return": avg_return}
                for ep, avg_return in self.eval_history
            ],
        }

trainer = Trainer()

# fastapi app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/state")
def get_state():
    return trainer.snapshot()

@app.post("/reset")
def reset():
    trainer.reset()
    return trainer.snapshot()

@app.post("/train")
def train(
    n: int = Query(50, ge=1, le=5000),
    alpha: float = Query(0.1, gt=0.0, le=1.0),
    eval_every: int = Query(50, ge=1, le=5000),
    n_eval: int = Query(20, ge=1, le=500),
):
    trainer.train(n=n, alpha=alpha, eval_every=eval_every, n_eval=n_eval)
    return trainer.snapshot()
