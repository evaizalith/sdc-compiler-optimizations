import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from lltfiUtilities import compileAndRun, interpretFIResults
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


class sdcAgent:
    def __init__(self):
        self.n_iterations = 0
        self.epsilon = 0        # randomness
        self.gamma = 0          # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = None
        self.trainer = None

    def remember(self, state, action, reward, nextState):
        self.memory.append((state, action, reward, nextState)) # memory is a deque of tuples

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else
            sample = self.memory

        states, actions, rewards, nextStates = zip(*sample)
        self.trainer.train_step(states, actions, rewards, nextStates)

    def trainShortMemory(self, state, action, reward, nextState):
        self.trainer.train_step(state, action, reward, nextState)

    # Used to select a list of opts to test
    def getAction(self, state):
        self.epsilon = 80 - n_iterations

        opts = [0] * N_OPTS

        # Randomly choose between exploration and exploitation 
        if random.randint(0, 200) < self.epsilon:
            opt = random.randint(0, N_OPTS)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            opt = torch.argmax(prediction).item()

