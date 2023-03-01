import random
import time

import pandas as pd
from params import *

class BaseAlgo(object):
    def __init__(self, env, ep, gamma):
        self.env = env
        self.ACTION_SPACE = self.env.action_space
        self.N_ACTION = len(self.ACTION_SPACE)
        self.EPSILON = ep
        self.GAMMA = gamma
        self.Q_table, self.Return_table, self.Num_StateAction = self.init_tables()
        self.Q_table, self.Return_table, self.Num_StateAction = self.init_tables()
        self.Episode_Step, self.Episode_Time = {}, {}
        self.goal_count, self.fail_count, self.total_rewards= 0, 0, 0
        self.Goal_Step, self.Fail_Step = {}, {}
        self.Rewards_List = [] # average rewards over time
        self.Success_Rate = {}

    def init_tables(self):
        Q_table = pd.DataFrame(columns=self.ACTION_SPACE)
        Return_table = pd.DataFrame(columns=self.ACTION_SPACE)
        Num_StateAction = pd.DataFrame(columns=self.ACTION_SPACE)
        return Q_table, Return_table, Num_StateAction

    def add_state_to_table(self, state, table):
        if state not in table.index:
            return table.append(pd.Series([0] * self.N_ACTION,
                        index=table.columns, name=state))

    def generate_policy(self, state):
        # Generate greedy epsilon policy
        if random.uniform(0, 1) > self.EPSILON:
            return_values = self.Q_table.loc[state]
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
                episode_elapsed_time = time.time() - episode_start_time
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

