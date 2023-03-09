from params import *
from base_algo import *

class SARSA(BaseAlgo, object):
    def __init__(self, env, ep, gamma, learning_rate):
        super().__init__(env, ep, gamma)
        self.LEARNING_RATE = learning_rate

    #Q(s, a) = Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))
    def learn(self, state, action_idx, reward, next_state, next_action_idx):
        if reward == float('-inf'):
            self.Q_table[state][action_idx] = float('-inf')
        else:
            q_target = reward + self.GAMMA * self.Q_table[next_state][next_action_idx]
            q_diff = q_target - self.Q_table[state][action_idx]

            # Update Q-Value in Q-table for state action pair
            self.Q_table[state][action_idx] += q_diff * self.LEARNING_RATE
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

        print(f'Accuracy: {self.Success_Rate}')
        print(f'Rewards: {self.Rewards_List}')
        print(f'Successes: {self.Goal_Step}')
        print(f'Q_Table: {self.Q_table}')
        if not is_train:
            self.env.draw_final_policy(self.Q_table)
            self.plot_convergence()
            time.sleep(1)
            # self.env.save(PATH='../Results/SARSA/')

    def test(self):
        print(self.Q_table)

# For used when running this python file by itself.
if __name__ == '__main__':
    from environment import Env
    env = Env()
    s = SARSA(env, EPSILON, GAMMA, LEARNING_RATE)
    s.run(NUM_EPISODES)