"""
This file contains the Q-Learning algorithm. It is a model-free algorithm that learns from experience.
It inherits from the BaseAlgo class.
"""

from src.Algorithms.base_algo import BaseAlgo
from src.Misc.params import *

# Initialises the Q-Learning algorithm with all variables from the BaseAlgo class.
class QLearning(BaseAlgo, object):

    # Initialise learning rate and if scheduling will be used for runs
    def __init__(self, env, ep, gamma, learning_rate):
        super().__init__(env, ep, gamma)
        self.LEARNING_RATE = learning_rate
        self.USE_GAMMA_SCHEDULE = USE_GAMMA_SCHEDULE
        self.USE_LR_SCHEDULE = USE_LR_SCHEDULE

    # This portion allows Q-Learning to learn from experience, where it accepts a state, action,
    # reward, next state. The Q-Value for the state-action pair is then updated based on the formula:
    # Q(S, A) = Q(S, A) + ALPHA * (R(S, A) + GAMMA * max(Q(S', A')) - Q(S, A))
    # Where ALPHA is the learning rate, GAMMA is the discount factor, and R(S, A) is the reward.
    # The Q-Value is then returned for cost calculation for plotting and analysis.
    # If the state-action pair is out of bounds, then the Q-value is left as -inf.
    def learn(self, state, action_idx, reward, next_state, _):
        if reward == float('-inf'):
            self.Q_table[state][action_idx] = float('-inf')
        else:
            q_target = reward + self.GAMMA * max(self.Q_table[next_state])
            q_diff = q_target - self.Q_table[state][action_idx]

            # Update Q-Value in Q-table for state action pair
            self.Q_table[state][action_idx] += q_diff * self.LEARNING_RATE
            self.Num_Visited[state] += 1
        return self.Q_table[state][action_idx]

    # This is the main function of the Q-Learning algorithm. It runs the algorithm for NUM_EPISODES
    # number of episodes specified in the params.py file. According to the params.py file, the USE_LR_SCHEDULE
    # and USE_GAMMA_SCHEDULE determines whether the learning rate and epsilon values are updated according to
    # the schedules defined in the BaseAlgo class. Finally the final optimal policy and the number of visits at
    # each state is drawn on the UI.
    def run(self, episodes, is_train=False, lr_schedule=None, gamma_schedule=None):
        for episode in range(1, episodes + 1):
            _ = self.generate_episode(episode, self.learn)
            if lr_schedule is not None and self.USE_LR_SCHEDULE:
                self.LEARNING_RATE = lr_schedule(episode)
            if gamma_schedule is not None and self.USE_GAMMA_SCHEDULE:
                self.GAMMA = gamma_schedule(episode)

            if episode != 0 and episode % ACCURACY_RANGE == 0:
                success_rate = self.goal_count / ACCURACY_RANGE
                self.Success_Rate[episode] = success_rate
                self.goal_count = 0

        print(f'Accuracy: {self.Success_Rate}')
        print(f'Rewards: {self.Rewards_List}')
        print(f'Successes: {self.Goal_Step}')
        print(f'Q_Table: {self.Q_table}')
        if not is_train:
            self.env.draw_final_policy(self.Q_table)
            # self.env.draw_number(self.Num_Visited)
            self.plot_results()
            self.plot_convergence()

# For used when running this python file by itself.
if __name__ == '__main__':
    from environment import Env
    env = Env()
    ql = QLearning(env, EPSILON, GAMMA, LEARNING_RATE)
    ql.run(NUM_EPISODES)
