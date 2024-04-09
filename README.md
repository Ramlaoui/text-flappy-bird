# text-flappy-bird

This repository contains a notebook for training two different agents to play the text-based game Flappy Bird. The game can be found [here](https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym) and has two versions with different state representations.

The first agent is a Monte Carlo based agent that uses a Q-table to store the expected reward for each state-action pair. The second agent is a Sarsa-lambda agent.

While the agents are mainly trained on the first version of the game, attemps to test the performances of the agent on the real game and the second versions were made. The implementation of the real game is taken from [here](https://github.com/Talendar/flappy-bird-gym).