from base_algo import *
import pandas as pd
from environment import *

class QLearning(BaseAlgo, object):
    def __init__(self, env, ep, gamma, learning_rate):
        super().__init__(env, ep, gamma)
        self.LEARNING_RATE = learning_rate
        self.Q_table = pd.DataFrame(columns=self.ACTION_SPACE)

    def add_state_q_table(self, state):
        if state not in self.Q_table.index:
            self.Q_table = self.Q_table.append(pd.Series([0] * self.N_ACTION,
                        index=self.Q_table.columns, name=state))

    def generate_episode(self):


    def test(self):
        print(self.Q_table)

if __name__ == '__main__':
    env = Env()
    ql = QLearning(env, EPSILON, GAMMA, LEARNING_RATE)
    ql.test()

