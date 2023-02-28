import random
import time

import numpy as np
from environment import Env
from params import *


class MonteCarlo(object):
    def __init__(self, env, ep, gamma):
        self.env = env
        self.N_STATES = self.env.num_states
        self.N_ACTION = self.env.num_actions
        self.ACTION_SPACE = self.env.action_space
        self.EPSILON = ep
        self.GAMMA = gamma
        self.Q_table, self.Return_table, self.Num_StateAction = self.init_tables()
        self.Episode_Step, self.Episode_Time = {}, {}
        self.goal_count, self.fail_count, self.total_rewards= 0, 0, 0
        self.Goal_Step, self.Fail_Step = {}, {}
        self.Rewards_List = [] # average rewards over time
        self.Success_Rate = {}

    def init_tables(self):
        Q_table, Return_table, Num_StateAction = {}, {}, {}

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                Q_table[(r, c)] = [0] * self.N_ACTION
                Return_table[(r, c)] = [0] * self.N_ACTION
                Num_StateAction[(r, c)] = [0] * self.N_ACTION
        return Q_table, Return_table, Num_StateAction

    def generate_policy(self, state):
        # Generate greedy epsilon policy
        if random.uniform(0, 1) > self.EPSILON:
            return_values = self.Q_table[state]
            action_idx = return_values.index(max(return_values))
            action = self.ACTION_SPACE[action_idx]
        else:
            action = random.choice(self.ACTION_SPACE)
        return action

    def generate_episode(self, episode):
        step = 0
        observation_state = self.env.reset()
        episode_info = []
        fps = self.env.fps

        episode_start_time = time.time()

        for _ in range(NUM_STEPS):
            time.sleep((1 / fps) if fps != 0 else 0)
            action = self.generate_policy(observation_state)
            observation_state, reward, is_done = self.env.step(action)
            episode_info.append((observation_state, self.env.action_space.index(action), reward))

            step += 1
            if is_done:
                episode_elapsed_time = time.time() -  episode_start_time
                self.Episode_Step[episode] = step
                self.Episode_Time[episode] = episode_elapsed_time

                if reward > 0:
                    self.goal_count += 1
                    self.Goal_Step[episode] = step
                elif reward < 0:
                    self.fail_count += 1
                    self.Fail_Step[episode] = step

                self.total_rewards += reward
                self.Rewards_List.append(self.total_rewards / episode)

                print(f"Episode {episode} finished in {step} steps in {episode_elapsed_time}")
                break

        return episode_info

    def run(self, episodes):
        for episode in range(1, episodes + 1):
            episode_info = self.generate_episode(episode)
            state_action_pair = [(s, a) for (_, s, a) in episode_info]
            G = 0

            for i in range(len(episode_info)):
                state, action, reward = episode_info[-i - 1]

                G = reward + self.GAMMA * G

                if (state, action) not in state_action_pair[:i]:
                    self.Return_table[state][action] += G
                    self.Num_StateAction[state][action] += 1
                    self.Q_table[state][action] = self.Return_table[state][action] / self.Num_StateAction[state][action]

            if episode !=0 and episode % ACCURACY_RANGE ==0:
                success_rate = self.goal_count / ACCURACY_RANGE
                self.Success_Rate[episode] = success_rate
                self.goal_count = 0

        print(f'Accuracy: {self.Success_Rate}')
        print(f'Rewards: {self.Rewards_List}')


if __name__ == "__main__":
    env = Env()
    mc = MonteCarlo(env, EPSILON, GAMMA)
    mc.run(NUM_EPISODES)