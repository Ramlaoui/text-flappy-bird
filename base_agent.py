import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from IPython.display import clear_output
except ImportError:
    pass


class BaseAgent:
    def __init__(self, env, agent_type, game):
        self.env = env
        self.agent_type = agent_type
        self.scores = []
        self.game = game
        self.quantized_interval1 = None

    def choose_action(self, state):
        raise NotImplementedError

    def update_Q(self, episode):
        raise NotImplementedError

    def update_Q_E(self, state, action, reward, next_state, next_action):
        raise NotImplementedError

    def update_policy(self):
        raise NotImplementedError

    def plot_score(self, title=""):
        # Rolling average of the scores
        plt.plot(self.scores, label="Score")
        plt.plot(
            self.smoothed_scores,
            label="Average score",
        )
        plt.title(f"Scores over episodes\n{title}")
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

    def quantize_state(self, state):
        N = 1000
        if self.quantized_interval1 is None:
            self.quantized_interval1 = [i * 2 / N for i in range(-N, N + 1)]
            self.quantized_interval2 = [i * 1 / N for i in range(N)]
        state[0] = self.quantized_interval1[
            np.digitize(state[0], self.quantized_interval1)
        ]
        state[1] = self.quantized_interval2[
            np.digitize(state[1], self.quantized_interval2)
        ]
        return tuple(state)

    def play(self, render=True, slow=False):
        if self.game == "display" or self.game == "screen":
            state, info = self.env.reset()
        else:
            state = self.env.reset()
            state = self.quantize_state(state)
        done = False
        i = 0
        reward = 0
        while not done:
            i += 1
            if self.game == "screen":
                state = state.flatten()
            action = self.choose_action(state)
            if self.game == "display" or self.game == "screen":
                state, reward_, done, _, info = self.env.step(action)
            else:
                state, reward_, done, _ = self.env.step(action)
                state = self.quantize_state(state)
            reward += reward_
            if render:
                clear_output(wait=True)
                print(self.env.render())
                if slow:
                    time.sleep(0.1)
                else:
                    time.sleep(0.01)
            if i > 5000:
                print(
                    f"Stopped playing to continue running the code. Score: {info['score']}"
                )
                break
        if render:
            print("Game over!")

        return reward

    def test(self, episodes=100, render=False):
        scores = []
        for _ in range(episodes):
            score = self.play(render=render, slow=False)
            scores.append(score)
        return np.convolve(scores, np.ones(100) / 100, mode="valid")
