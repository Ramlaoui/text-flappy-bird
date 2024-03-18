import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from base_agent import BaseAgent

try:
    from IPython.display import clear_output
except ImportError:
    pass


class MonteCarloAgent(BaseAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, game="display"):
        super().__init__(env, "monte_carlo", game)
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}
        self.returns = {}
        self.policy = {}
        self.scores = []
        self.game = game

    def choose_action(self, state):
        state = tuple(state)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [
                self.Q.get((state, a), 0) for a in range(self.env.action_space.n)
            ]
            max_q_value = max(q_values)
            # In case there are multiple actions with the same max Q-value, we select randomly among them
            actions_with_max_q = [a for a, q in enumerate(q_values) if q == max_q_value]
            return np.random.choice(actions_with_max_q)

    def update_Q(self, episode):
        G = 0
        for state, action, reward in reversed(episode):
            state = tuple(state)
            G = self.gamma * G + reward
            if (state, action) not in self.returns:
                self.returns[(state, action)] = []
            self.returns[(state, action)].append(G)
            self.Q[(state, action)] = np.mean(self.returns[(state, action)])

    def update_policy(self):
        for state, _ in self.Q:
            state = tuple(state)
            q_values = [
                self.Q.get((state, a), 0) for a in range(self.env.action_space.n)
            ]
            best_action = np.argmax(q_values)
            self.policy[state] = best_action

    def train(self, episodes=1000, log_every=100, render=False):
        if log_every == -1:
            log_every = episodes
        pbar = tqdm(range(episodes))
        for i in range(episodes):
            rewards = 0
            episode = []
            if self.game == "display" or self.game == "screen":
                state, info = self.env.reset()
            else:
                state = self.env.reset()
                state = self.quantize_state(state)
            done = False
            j = 0
            while not done:
                j += 1
                if self.game == "screen":
                    state = state.flatten()
                action = self.choose_action(state)
                if self.game == "display" or self.game == "screen":
                    next_state, reward, done, _, info = self.env.step(action)
                else:
                    next_state, reward, done, info = self.env.step(action)
                    next_state = self.quantize_state(next_state)
                if self.game == "screen":
                    next_state = next_state.flatten()
                episode.append((state, action, reward))
                state = next_state
                rewards += reward
                if i % log_every == 0 and render:
                    clear_output(wait=True)
                    print(self.env.render())
                    time.sleep(0.05)
                # early stopping if the agent is always winning (plateau on reward, but it's ok)
                if j > 100000:
                    break
            self.scores.append(rewards)
            if i % log_every == 0:
                print(
                    f"Episode {i} - Average score: {np.mean(self.scores[-log_every:])}"
                )
            self.update_Q(episode)
            self.update_policy()
            pbar.set_description(f"Episode {i} - Reward: {rewards}")
            pbar.update(1)
        self.smoothed_scores = np.convolve(
            self.scores, np.ones(100) / 100, mode="valid"
        )
