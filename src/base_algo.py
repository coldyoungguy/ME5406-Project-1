import random
import time
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
        Q_table, Return_table, Num_StateAction = {}, {}, {}

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                Q_table[(r, c)] = [0] * self.N_ACTION
                Return_table[(r, c)] = [0] * self.N_ACTION
                Num_StateAction[(r, c)] = [0] * self.N_ACTION

        return Q_table, Return_table, Num_StateAction

    def generate_policy(self, state):
        # Generate greedy epsilon policy
        if random.uniform(0, 1) < self.EPSILON:
            return_values = self.Q_table[state]
            action_idx = return_values.index(random.choice([i for i in return_values if i == max(return_values)]))
            action = self.ACTION_SPACE[action_idx]
        else:
            action = random.choice(self.ACTION_SPACE)
        return action

    def generate_episode(self, episode, method):
        step, cost = 0, 0
        current_state = self.env.reset()
        episode_info = []
        fps = self.env.fps

        episode_start_time = time.time()

        for _ in range(NUM_STEPS):
            time.sleep((1 / fps) if fps != 0 else 0)
            action = self.generate_policy(current_state)
            next_state, reward, is_done = self.env.step(action)

            if method is not None:
                cost += method(current_state, self.env.action_space.index(action), reward, next_state)

            current_state = next_state
            episode_info.append((current_state, self.env.action_space.index(action), reward))

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

