from base_algo import BaseAlgo
from params import *

class QLearning(BaseAlgo, object):
    def __init__(self, env, ep, gamma, learning_rate):
        super().__init__(env, ep, gamma)
        self.LEARNING_RATE = learning_rate
        self.Num_Visited = {k:0 for k in list(self.Q_table.keys())}

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

    def run(self, episodes, is_train=False):
        for episode in range(1, episodes + 1):
            _ = self.generate_episode(episode, self.learn)
            if USE_LR_SCHEDULE: self.LEARNING_RATE = self.lr_scheduler(self.LEARNING_RATE, episode)
            if USE_EP_SCHEDULE: self.EPSILON = self.ep_scheduler(self.EPSILON, episode)

            if episode != 0 and episode % ACCURACY_RANGE == 0:
                success_rate = self.goal_count / ACCURACY_RANGE
                self.Success_Rate[episode] = success_rate
                self.goal_count = 0

        # print(f'Accuracy: {self.Success_Rate}')
        # print(f'Rewards: {self.Rewards_List}')
        # print(f'Successes: {self.Goal_Step}')
        print(f'Num_StateAction: {self.Num_Visited}')
        print(f'Q_Table: {self.Q_table}')
        if not is_train:
            self.env.draw_final_policy(self.Q_table)
            # self.env.draw_number(self.Num_Visited)
            self.plot_convergence()


    def test(self):
        print(self.Q_table)

# For used when running this python file by itself.
if __name__ == '__main__':
    from environment import Env
    env = Env()
    ql = QLearning(env, EPSILON, GAMMA, LEARNING_RATE)
    ql.run(NUM_EPISODES)
