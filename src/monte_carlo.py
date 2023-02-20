from environment import Env
from params import *


class MonteCarlo(object):
    def __init__(self, env, ep, gamma):
        self.env = env
        self.n_states = self.env.n_states
        self.n_actions = self.env.n_actions
        self.ep = ep
        self.gamma = gamma

    def generatePolicy(self):



if __name__ == "__main__":
    env = Env()
    mc = MonteCarlo(env, EPSILON, GAMMA)