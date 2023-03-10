"""
This file contains the SARSA algorithm class. The SARSA algorithm is a model-free reinforcement learning algorithm.
It inherits from the BaseAlgo class.
"""

from src.Misc.params import *
from src.Algorithms.base_algo import BaseAlgo

# Initialise the SARSA algorithm with all variables from the BaseAlgo class.
class SARSA(BaseAlgo, object):

    # Initialise learning rate and if scheduling will be used for runs.
    def __init__(self, env, ep, gamma, learning_rate):
        super().__init__(env, ep, gamma)
        self.LEARNING_RATE = learning_rate
        self.USE_GAMMA_SCHEDULE = USE_GAMMA_SCHEDULE
        self.USE_LR_SCHEDULE = USE_LR_SCHEDULE

    # This portion allows SARSA to learn from experience, where it accepts a state, action,
    # reward, next state, and next action. The Q-Value for the state-action pair is then updated based on the formula:
    # Q(S, A) = Q(S, A) + ALPHA * (R(S, A) + GAMMA * Q(S', A') - Q(S, A))
    # Where ALPHA is the learning rate, GAMMA is the discount factor, and R(S, A) is the reward.
    # The Q-Value is then returned for cost calculation for plotting and analysis.
    # If the state-action pair is out of bounds, then the Q-value is left as -inf.
    def learn(self, state, action_idx, reward, next_state, next_action_idx):
        if reward == float('-inf'):
            self.Q_table[state][action_idx] = float('-inf')
        else:
            q_target = reward + self.GAMMA * self.Q_table[next_state][next_action_idx]
            q_diff = q_target - self.Q_table[state][action_idx]

            # Update Q-Value in Q-table for state action pair
            self.Q_table[state][action_idx] += q_diff * self.LEARNING_RATE
            self.Num_Visited[state] += 1
        return self.Q_table[state][action_idx]

    # This is the main function for the SARSA algorithm. It run the algorithm for NUM_EPISODES number of episodes.
    # which is specified in the params.py file. It also accepts a learning rate schedule and a discount factor schedule.
    # The use of LR and GAMMA schedules are specified in the params.py file, whcih cna be configured in the UI as well.
    # finally the final optimal policy and number of visits can be drawn on the UI.
    def run(self, episodes, is_train=False, lr_schedule=None, gamma_schedule=None):
        for episode in range(1, episodes + 1):
            _ = self.generate_episode(episode, self.learn)
            if lr_schedule is not None and self.USE_LR_SCHEDULE:
                self.LEARNING_RATE = lr_schedule(self.LEARNING_RATE, episode)
            if gamma_schedule is not None and self.USE_GAMMA_SCHEDULE:
                self.GAMMA = gamma_schedule(self.GAMMA, episode)

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
    s = SARSA(env, EPSILON, GAMMA, LEARNING_RATE)
    s.run(NUM_EPISODES)