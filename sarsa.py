import os
import sys
import time
from tqdm import tqdm
import numpy as np
from base_agent import BaseAgent

try:
    from IPython.display import clear_output
except ImportError:
    pass


class SarsaLambdaAgent(BaseAgent):
    def __init__(
        self, env, lambda_=0.9, gamma=0.99, alpha=0.1, epsilon=0.1, game="display"
    ):
        super().__init__(env, "sarsa", game)
        self.env = env
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = {}
        self.E = {}
        self.game = game

    def choose_action(self, state):
        state = tuple(state)
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore action space
        else:
            self.Q.setdefault(state, {})
            return max(
                self.Q[state],
                key=self.Q[state].get,
                default=self.env.action_space.sample(),
            )

    def update_Q_E(self, state, action, reward, next_state, next_action):
        # Ensure state-action pairs are in the Q and E tables
        state = tuple(state)
        next_state = tuple(next_state)
        self.Q.setdefault(state, {}).setdefault(action, 0)
        self.Q.setdefault(next_state, {}).setdefault(next_action, 0)
        self.E.setdefault(state, {}).setdefault(action, 0)

        # Compute the TD error
        delta = (
            reward
            + self.gamma * self.Q[next_state][next_action]
            - self.Q[state][action]
        )

        # Increment the eligibility trace for the current state-action
        self.E[state][action] += 1

        # Now, update Q and E only for visited state-action pairs
        for s, actions in self.E.items():
            for a in actions:
                s = tuple(s)
                self.Q[s][a] += (
                    self.alpha * delta * self.E[s][a]
                )  # Update Q-value using eligibility trace
                self.E[s][a] *= self.gamma * self.lambda_  # Decay eligibility trace

    def train(self, episodes=1000, log_every=100, render=False):
        if log_every == -1:
            log_every = episodes
        pbar = tqdm(range(episodes))
        for i in range(episodes):
            self.E = {}  # Reset eligibility traces
            if self.game == "display" or self.game == "screen":
                state, info = self.env.reset()
                if self.game == "screen":
                    state = state.flatten()
            else:
                state = self.env.reset()
                state = self.quantize_state(state)
            action = self.choose_action(state)
            done = False
            rewards = 0
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
                next_action = self.choose_action(next_state)
                self.update_Q_E(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
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
            pbar.set_description(f"Episode {i} - Reward: {rewards}")
            pbar.update(1)
        self.smoothed_scores = np.convolve(
            self.scores, np.ones(100) / 100, mode="valid"
        )
