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
N_OPTS = 100
LR = 0.001

class Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            nextState = torch.unsqueeze(nextState, 0)

        prediction = self.model(state)

        target = prediction.clone()
        for idx in range(len(state)):
            Q_new = reward[idx] + self.gamma * torch.max(self.model(nextState[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()
            

class Linear_QNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super()__init__()

        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, fileName='trained_model.pt'):
        torch.save(self.state_dict(), fileName)

class sdcAgent:
    def __init__(self):
        self.n_iterations = 0
        self.epsilon = 0        # randomness
        self.gamma = 0          # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_Qnet(1, 256, N_OPTS)
        self.trainer = Trainer(self.model, r=LR, gamma=self.gamma)

    def remember(self, state, action, reward, nextState):
        self.memory.append((state, action, reward, nextState))

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else
            sample = self.memory

        states, actions, rewards, nextStates = zip(*sample)
        self.trainer.trainStep(states, actions, rewards, nextStates)

    def trainShortMemory(self, state, action, reward, nextState):
        self.trainer.trainStep(state, action, reward, nextState)

    # Used to select a list of opts to test
    def getAction(self, state):
        self.epsilon = 80 - n_iterations

        opts = [0] * N_OPTS

        # Randomly choose between exploration and exploitation 
        if random.randint(0, 200) < self.epsilon:
            opt = random.randint(0, N_OPTS)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            opt = torch.argmax(prediction).item()

