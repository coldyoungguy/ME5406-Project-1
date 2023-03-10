"""
This file contains the Monte Carlo algorithm. It is a model-free algorithm that learns from experience.
It inherits from the BaseAlgo class.
"""

from src.Misc.params import *
from src.Algorithms.base_algo import BaseAlgo

class MonteCarlo(BaseAlgo, object):

    # Initialize the Monte Carlo algorithm with all variables from the BaseAlgo class.
    def __init__(self, env, ep, gamma):
        super().__init__(env, ep, gamma)

    # This is the main function of the Monte Carlo algorithm. Which allows the algorithm to run
    # NUM_EPISODES number of episodes. When each episode is finished, episode_info is collected and
    # is backtracked to find the return for each state-action pair if the state-action pair is not
    # already in the state-action pair list. The return and the number of visits in this state-action
    # pair is then used to update the Q-table. The number of visits ar each state-action pair is
    # incremented by 1 every time the state-action pair is visited.
    # The return is calculated by G(S, A) = R(S, A) + GAMMA * G(S', A')
    def run(self, episodes, is_train=False):
        for episode in range(1, episodes + 1):
            episode_info = self.generate_episode(episode)
            if USE_EP_SCHEDULE: self.EPSILON = self.ep_scheduler(self.EPSILON, episode)

            state_action_pair = [(s, a) for (s, a, _) in episode_info]
            G = 0

            for i in range(len(episode_info)):
                state, action, reward = episode_info[-i - 1]
                self.Num_Visited[state] += 1

                if reward != float('-inf'):
                    G = reward + self.GAMMA * G

                if (state, action) not in state_action_pair[:i]:
                    self.Num_StateAction[state][action] += 1
                    if reward == float('-inf'):
                        self.Return_table[state][action] = float('-inf')
                        self.Q_table[state][action] = float('-inf')
                    else:
                        self.Return_table[state][action] += G
                        self.Q_table[state][action] = self.Return_table[state][action] / self.Num_StateAction[state][action]

            if episode !=0 and episode % ACCURACY_RANGE ==0:
                success_rate = self.goal_count / ACCURACY_RANGE
                self.Success_Rate[episode] = success_rate
                self.goal_count = 0

        print(f'Accuracy: {self.Success_Rate}')
        print(f'Rewards: {self.Rewards_List}')
        print(f'Successes: {self.Goal_Step}')
        print(f'Q_Table: {self.Q_table}')

        # When training, in order to not interrupt the training process, the graph is not drawn.
        # and the UI is also not updated with the final policy.
        if not is_train:
            self.env.draw_final_policy(self.Q_table)
            self.plot_results()
            self.plot_convergence()

# For used when running this python file by itself.
if __name__ == "__main__":
    from environment import Env
    env = Env()
    mc = MonteCarlo(env, EPSILON, GAMMA)
    mc.run(NUM_EPISODES)