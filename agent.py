import numpy as np


class TabularQLearningAgent:
    def __init__(self, action_size, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.02, seed=None):
        self.action_size = action_size
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.rng = np.random.default_rng(seed)
        self.q_table = {}

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=float)
        return self.q_table[state]

    def choose_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        values = self.get_q(state)
        max_indices = np.flatnonzero(np.isclose(values, values.max()))
        return int(self.rng.choice(max_indices))

    def learn(self, state, action, reward, next_state, done):
        q = self.get_q(state)
        target = reward if done else reward + self.gamma * self.get_q(next_state).max()
        before = float(q[action])
        q[action] += self.alpha * (target - q[action])
        return before, float(q[action])

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
