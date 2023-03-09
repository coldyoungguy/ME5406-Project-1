import copy
import random
import time

import numpy as np
from params import *
import matplotlib.pyplot as plt

class BaseAlgo(object):
    def __init__(self, env, ep, gamma):
        self.env = env
        self.ACTION_SPACE = self.env.action_space
        self.N_ACTION = len(self.ACTION_SPACE)
        self.EPSILON = ep
        self.GAMMA = gamma
        self.LEARNING_RATE = 0
        self.Q_table, self.Return_table, self.Num_StateAction = self.init_tables()
        self.Q_table, self.Return_table, self.Num_StateAction = self.init_tables()
        self.Episode_Step, self.Episode_Time = {}, {}
        self.goal_count, self.fail_count, self.total_rewards= 0, 0, 0
        self.Goal_Step, self.Fail_Step = {}, {}
        self.Rewards_List = {} # average rewards over time
        self.Success_Rate = {}
        self.Episode_Cost = {}
        self.Q_Converge = {}
        self.Optimal_Policy = {}
        self.Policy_Changes_List = {k:0 for k in range(1, NUM_EPISODES + 1)}
        self.total_policy_changes = 0

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
            max_ = max(return_values)
            if type(max_) == list:
                action_idx = random.choice([i for i in range(len(return_values)) if return_values[i] == max(return_values)])
            else:
                action_idx = return_values.index(max_)
            action = self.ACTION_SPACE[action_idx]
        else:
            action = random.choice(self.ACTION_SPACE)
        return action

    def get_optimal_policy(self):
        optimal_policy = {}
        for state in self.Q_table.keys():
            max_ = max(self.Q_table[state])
            if type(max_) == list:
                optimal_policy[state] = self.Q_table[state].index(np.random.choice(max_))
            else:
                optimal_policy[state] = self.Q_table[state].index(max_)
        return optimal_policy

    def find_valid_action(self, state):
        action = self.generate_policy(state)
        action_idx = self.ACTION_SPACE.index(action)
        while self.Q_table[state][action_idx] == float('-inf'):
            action = self.generate_policy(state)
            action_idx = self.ACTION_SPACE.index(action)
        return action, action_idx

    def get_q_convergence(self, episode):
        all_q_values = copy.deepcopy(list(self.Q_table.values()))
        for i in range(len(all_q_values)):
            for j in range(len(all_q_values[i])):
                if all_q_values[i][j] == float('-inf'):
                    all_q_values[i][j] = 0
        mse_diff = np.sum(np.power(all_q_values, 2)) / len(self.Q_table.keys())
        self.Q_Converge[episode] = mse_diff

    def get_policy_convergence(self, episode):
        self.Optimal_Policy[episode] = self.get_optimal_policy()

        if episode == 1:
            return

        for state in self.Optimal_Policy[episode].keys():
            if self.Optimal_Policy[episode][state] != self.Optimal_Policy[episode - 1][state]:
                self.total_policy_changes += 1

        self.Policy_Changes_List[episode] = self.total_policy_changes

    def lr_scheduler(self, lr, episode):
        LR_DECAY = self.LEARNING_RATE / NUM_EPISODES
        return lr / (1 + LR_DECAY * episode)

    def ep_scheduler(self, ep, episode):
        # Linear
        return ep * (1 - episode / NUM_EPISODES)

    def generate_episode(self, episode, method=None):
        cost = 0
        current_state = self.env.reset()
        action, action_idx = self.find_valid_action(current_state)
        episode_info = []
        fps = self.env.fps

        episode_start_time = time.time()

        for step in range(1, NUM_STEPS + 1):
            time.sleep((1 / fps) if fps != 0 else 0)

            next_state, reward, is_done = self.env.step(action)
            episode_info.append((current_state, action_idx, reward))
            self.get_q_convergence(episode)
            self.get_policy_convergence(episode)

            next_action, next_action_idx = self.find_valid_action(next_state)

            if method is not None:
                temp = method(current_state, action_idx, reward, next_state, next_action_idx)
                cost += temp
                self.Episode_Cost[episode] = cost
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

                self.total_rewards += (reward if reward != float('-inf') else 0)
                # Rolling rewards for more accurate representation
                self.Rewards_List[episode] = self.total_rewards / episode
                print(f"Episode {episode} finished in {step} steps in {episode_elapsed_time}")
                break

            current_state = next_state
            action = next_action
            action_idx = next_action_idx

        return episode_info

    def get_results(self):
        return self.Q_table, \
            self.Return_table, \
            self.Goal_Step, \
            self.Episode_Time, \
            self.Success_Rate, \
            self.Rewards_List, \
            self.Episode_Cost, \
            [len(self.Goal_Step), len(self.Fail_Step)]

    def plot_results(self):
        # Plot the results of accuracy, rewards, and success rate over episodes
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        label = 'Gamma=' + str(self.GAMMA) + ' Lr=' + str(self.LEARNING_RATE)

        ax[0, 0].plot(list(self.Goal_Step.keys()), list(self.Goal_Step.values()), label=label)
        ax[0, 0].set_title("Steps Taken per Successful Episode")
        ax[0, 0].set_xlabel("Episode")
        ax[0, 0].set_ylabel("Steps")
        ax[0, 0].legend()

        ax[0, 1].plot(list(self.Episode_Time.keys()), list(self.Episode_Time.values()), label=label)
        ax[0, 1].set_title("Time Taken per Episode")
        ax[0, 1].set_xlabel("Episode")
        ax[0, 1].set_ylabel("Time")
        ax[0, 1].legend()

        ax[0, 2].plot(list(self.Success_Rate.keys()), list(self.Success_Rate.values()), label=label)
        ax[0, 2].set_title(f"Average Success Rate over {ACCURACY_RANGE} episodes")
        ax[0, 2].set_xlabel("Episode")
        ax[0, 2].set_ylabel("Success Rate")
        ax[0, 2].legend()

        ax[1, 0].plot(list(self.Rewards_List.keys()), list(self.Rewards_List.values()), label=label)
        ax[1, 0].set_title(F"Average Rewards over {ROLLING_AVG} Episodes")
        ax[1, 0].set_xlabel("Episode")
        ax[1, 0].set_ylabel("Rewards")
        ax[1, 0].legend()

        ax[1, 1].plot(list(self.Episode_Cost.keys()), list(self.Episode_Cost.values()), label=label)
        ax[1, 1].set_title("Cost over Episodes")
        ax[1, 1].set_xlabel("Episode")
        ax[1, 1].set_ylabel("Cost")
        ax[1, 1].legend()

        ax[1, 2].bar(['Success', 'Fail'], (len(self.Goal_Step), len(self.Fail_Step)), label=label)
        ax[1, 2].set_title("Successes and Fails")
        ax[1, 2].set_ylabel("Number")
        ax[1, 2].legend()


        if self.LEARNING_RATE == 0:
            fig.suptitle(f"Results of {self.__class__.__name__} with Gamma={self.GAMMA}, Epsilon={self.EPSILON}")
        else:
            fig.suptitle(f"Results of {self.__class__.__name__} with Gamma={self.GAMMA}, Epsilon={self.EPSILON}, Learning Rate={self.LEARNING_RATE}")

        plt.tight_layout()
        plt.show()

    def plot_convergence(self):
        fig, ax1 = plt.subplots(figsize=(6, 4.5))
        # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        label = 'Gamma=' + str(self.GAMMA) + ' Lr=' + str(self.LEARNING_RATE)

        ax1.plot(list(self.Q_Converge.keys()), list(self.Q_Converge.values()), label=label, color='blue')
        # ax1.set_title("Q-Table Convergence")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Mean-Squared Error", color='blue')
        ax1.legend()

        ax2 = ax1.twinx()

        ax2.plot(list(self.Policy_Changes_List.keys()), list(self.Policy_Changes_List.values()), label=label, color='green')
        # ax2.set_title("Policy Convergence")
        ax2.set_ylabel("Policy Changes", color='red')
        ax2.legend()

        if self.LEARNING_RATE == 0:
            fig.suptitle(f"Convergence of {self.__class__.__name__} with Gamma={self.GAMMA}, Epsilon={self.EPSILON}")
        else:
            fig.suptitle(f"Convergence of {self.__class__.__name__} with Gamma={self.GAMMA}, Epsilon={self.EPSILON}, Learning Rate={self.LEARNING_RATE}")

        plt.tight_layout()
        plt.show()






