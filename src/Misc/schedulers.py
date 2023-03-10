from src.Misc.params import *
from math import pow, floor

def lin_schedule(episode):
    return LEARNING_RATE * (1 - episode / NUM_EPISODES)

def exp_schedule(episode):
    return LEARNING_RATE * 0.9997 ** episode

def drop_schedule(episode):
    drop = 0.5
    epi_drop = 1000
    return LEARNING_RATE * pow(drop, floor(1 + episode / epi_drop))

def discrete_schedule(episode):
    completion = episode / NUM_EPISODES
    if completion < 0.5:
        return 0.9
    elif episode < 0.9:
        return 0.5
    else:
        return 0.1