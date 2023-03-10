from params import *

def lin_schedule(rate, episode):
    return rate * (1 - episode / NUM_EPISODES)

def exp_schedule(rate, episode):
    return rate * 0.99 ** episode

def discrete_schedule(_, episode):
    completion = episode / NUM_EPISODES
    if completion < 0.5:
        return 1.0
    elif completion < 0.9:
        return 0.5
    else:
        return 0.1