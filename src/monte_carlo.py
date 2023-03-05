from params import *
from base_algo import *


class MonteCarlo(BaseAlgo, object):
    def __init__(self, env, ep, gamma):
        super().__init__(env, ep, gamma)
        self.Q_table, self.Return_table, self.Num_StateAction = self.init_tables()
        self.Episode_Step, self.Episode_Time = {}, {}
        self.goal_count, self.fail_count, self.total_rewards= 0, 0, 0
        self.Goal_Step, self.Fail_Step = {}, {}
        self.Success_Rate = {}


    def run(self, episodes):
        for episode in range(1, episodes + 1):
            episode_info = self.generate_episode(episode)
            state_action_pair = [(s, a) for (_, s, a) in episode_info]
            G = 0

            for i in range(len(episode_info)):
                state, action, reward = episode_info[-i - 1]

                if reward != float('-inf'):
                    G = reward + self.GAMMA * G

                if (state, action) not in state_action_pair[:i]:
                    self.Num_StateAction[state][action] = self.Num_StateAction[state][action] + 1
                    if reward == float('-inf'):
                        self.Return_table[state][action] = float('-inf')
                        self.Q_table[state][action] = float('-inf')
                    else:
                        self.Return_table[state][action] = self.Return_table[state][action] + G
                        self.Q_table[state][action] = self.Return_table[state][action] / self.Num_StateAction[state][action]

            if episode !=0 and episode % ACCURACY_RANGE ==0:
                success_rate = self.goal_count / ACCURACY_RANGE
                self.Success_Rate[episode] = success_rate
                self.goal_count = 0

        print(f'Accuracy: {self.Success_Rate}')
        print(f'Rewards: {self.Rewards_List}')
        print(f'Successes: {self.Goal_Step}')
        print(f'Q_Table: {self.Q_table}')


if __name__ == "__main__":
    from environment import Env
    env = Env()
    mc = MonteCarlo(env, EPSILON, GAMMA)
    mc.run(NUM_EPISODES)