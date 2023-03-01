from base_algo import *
from environment import *

class QLearning(BaseAlgo, object):
    def __init__(self, env, ep, gamma, learning_rate):
        super().__init__(env, ep, gamma)
        self.LEARNING_RATE = learning_rate
        self.Q_table, _, _ = self.init_tables()

    def learn(self, state, action, reward, next_state):
        q_target = reward * self.GAMMA * max(self.Q_table[next_state])
        q_diff = q_target - self.Q_table[state][action]

        # Update Q-Value in Q-tabel for state action pair
        self.Q_table[state][action] += q_diff * self.LEARNING_RATE
        return self.Q_table[state][action]

    def run(self, episodes):
        for episode in range(1, episodes + 1):
            episode_info = self.generate_episode(episode, self.learn)

        print(f'Accuracy: {self.Success_Rate}')
        print(f'Rewards: {self.Rewards_List}')
        print(f'Successes: {self.Goal_Step}')

    def test(self):
        print(self.Q_table)

if __name__ == '__main__':
    env = Env()
    ql = QLearning(env, EPSILON, GAMMA, LEARNING_RATE)
    ql.run(NUM_EPISODES)
