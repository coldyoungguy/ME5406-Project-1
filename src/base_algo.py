"""
This creates a base class for all algorithms to inherit from.
All shared functions exists here.
"""

import copy
import random
import time
import numpy as np
from params import *
import matplotlib.pyplot as plt

# Initialises the base class for all algorithms
class BaseAlgo(object):
    def __init__(self, env, ep, gamma):

        # Sets up the environment for the algorithm to interact with
        self.env = env
        # Inherit the action space from the environment and the number of actions
        self.ACTION_SPACE = self.env.action_space
        self.N_ACTION = len(self.ACTION_SPACE)
        self.grid_size = self.env.grid_size
        # Sets up hyperparameters
        self.EPSILON = ep
        self.GAMMA = gamma
        self.LEARNING_RATE = 0

        # Creates the Q-table, Return-table, and Num-StateAction-table
        # In the form of a dictionary where {state: [action1, action2, ...]} and state is a tuple (row, col)
        self.Q_table, self.Return_table, self.Num_StateAction = self.init_tables()

        # Sets up a dictionary for the number of times a node is visited
        # In the form of {state: num_visited}
        self.Num_Visited = {k: 0 for k in list(self.Q_table.keys())}

        """ Sets up the dictionaries for storing the results of the algorithm
        Episode_Step: {episode: step} for storing the number of steps taken in each episode
        Episode_Time: {episode: time} for storing the time taken in each episode
        Goal_Step: {episode: step} for storing the number of steps taken to reach the goal in each episode
        Fail_Step: {episode: step} for storing the number of steps taken to fail in each episode
        Rewards_List: {episode: reward} for storing the average reward over episodes
        Success_Rate: {episode: success_rate} for storing the success rate over episodes
        Episode_Cost: {episode: cost} for storing the cost of each episode
        Q_Converge: {episode: MSE} for storing the MSE of the Q-table over episodes
        Optimal_Policy: {episode: policy} for storing the optimal policy at each episode
        Policy_Changes_List: {episode: changes} for storing the number of policy changes over episodes
        total_policy_changes: the total number of policy changes over all episodes """
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

    # Finds the maximum value in an array and returns the list of indices where the maximum value is found
    # If there is only one maximum value, it returns the index of that value
    def max_where(self, array):
        res = np.argwhere(array == np.max(array))
        if np.shape(res)[0] > 1:
            return res[0]
        return res[0][0]

    # Initialises the Q-table, Return-table, and Num-StateAction-table
    # In the form of a dictionary where {state: [action1, action2, ...]} and state is a tuple (row, col)
    def init_tables(self):
        Q_table, Return_table, Num_StateAction = {}, {}, {}

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                Q_table[(r, c)] = [0] * self.N_ACTION
                Return_table[(r, c)] = [0] * self.N_ACTION
                Num_StateAction[(r, c)] = [0] * self.N_ACTION

        return Q_table, Return_table, Num_StateAction

    # Generate a greedy epsilon policy using epsilon, where if a random number generated
    # is more than epsilon, it will choose a random action, otherwise it will choose the
    # action with the highest Q-value, if there are multiple actions with the same Q-value,
    # it will choose one of them randomly
    def generate_policy(self, state):
        if random.uniform(0, 1) > self.EPSILON:
            return_values = self.Q_table[state]
            action_idx = self.max_where(np.array(return_values))
            if type(action_idx) == np.ndarray:
                action_idx = random.choice(action_idx)
            action = self.ACTION_SPACE[action_idx]
        else:
            action = random.choice(self.ACTION_SPACE)
        return action

    # Stores the optimal policy in a dictionary where {state: action}
    # if there are multiple actions with the same Q-value, it will choose one of them randomly
    def get_optimal_policy(self):
        optimal_policy = {}
        for state in self.Q_table.keys():
            max_ = self.max_where(self.Q_table[state])
            if type(max_) == np.ndarray:
                optimal_policy[state] = np.random.choice(max_)
            else:
                optimal_policy[state] = max_
        return optimal_policy

    # This function is used to handle the case where the reward returned and the q-value is -inf
    # That can happen when the agent traverse out of the grid map.
    # When the Q-value is -inf, it will repeatedly call the generate_policy function until it
    # finds a valid action where the q-value of the state-action pair is not -inf
    def find_valid_action(self, state):
        action = self.generate_policy(state)
        action_idx = self.ACTION_SPACE.index(action)
        while self.Q_table[state][action_idx] == float('-inf'):
            action = self.generate_policy(state)
            action_idx = self.ACTION_SPACE.index(action)
        return action, action_idx

    # Calculates the MSE of the Q-table over episodes
    # to prevent the calculation of the MSE of the Q-table from being affected by the -inf values,
    # all the -inf values are replaced with 0
    def get_q_convergence(self, episode):
        all_q_values = copy.deepcopy(list(self.Q_table.values()))
        for i in range(len(all_q_values)):
            for j in range(len(all_q_values[i])):
                if all_q_values[i][j] == float('-inf'):
                    all_q_values[i][j] = 0
        mse_diff = np.sum(np.power(all_q_values, 2)) / len(self.Q_table.keys())
        self.Q_Converge[episode] = mse_diff

    # Calculates the number of policy changes over episodes
    # If the state-action pair in the policy at the current episode is different from that of the
    # policy at the previous episode, the total number of policy changes is increased by 1
    def get_policy_convergence(self, episode):
        self.Optimal_Policy[episode] = self.get_optimal_policy()

        # Since Episodes starts at 1, there will be no previous episode to compare to
        if episode == 1:
            return

        for state in self.Optimal_Policy[episode].keys():
            if self.Optimal_Policy[episode][state] != self.Optimal_Policy[episode - 1][state]:
                self.total_policy_changes += 1

        self.Policy_Changes_List[episode] = self.total_policy_changes

    # Creates a schedule for the learning rate, to allow for lower learning rates at later episodes
    # This is to prevent the agent from overfitting to the environment, and to allow for more exploration
    def lr_scheduler(self, lr, episode, rate=1):
        LR_DECAY = self.LEARNING_RATE / NUM_EPISODES
        return lr / (1 * rate + LR_DECAY * episode)

    # Creates a schedule for the epsilon, to allow for lower epsilon at later episodes,
    # to allow for more exploration of the environment at the start of the training
    def ep_scheduler(self, ep, episode, rate=1):
        # Linear
        return ep * (episode / NUM_EPISODES) * rate

    # Generates an episode. This begins by resetting the environment, and finding a valid action
    # The episode will then run for NUM_STEPS steps, where the agent will take an action, and
    # the environment will return the next state, reward, and whether the episode is done
    # The episode's information will then be stored in a list (used for FVMC),
    # and the Q-table and policy will be updated, and the MSE of the Q-table and
    # the number of policy changes will be calculated
    def generate_episode(self, episode, method=None):
        cost = 0
        current_state = self.env.reset()
        action, action_idx = self.find_valid_action(current_state)
        episode_info = []

        episode_start_time = time.time()

        for step in range(1, NUM_STEPS + 1):
            time.sleep((1 / self.env.fps) if self.env.fps != 0 else 0)

            next_state, reward, is_done = self.env.step(action)
            episode_info.append((current_state, action_idx, reward))
            self.get_q_convergence(episode)
            self.get_policy_convergence(episode)

            next_action, next_action_idx = self.find_valid_action(next_state)

            # Used to update the Q-Table for SARSA and Q-Learning
            # If the method is not specified, it will be skipped for FVMC
            if method is not None:
                cost += method(current_state, action_idx, reward, next_state, next_action_idx)
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
                self.Rewards_List[episode] = self.total_rewards / episode
                print(f"Episode {episode} finished in {step} steps in {episode_elapsed_time}")
                break

            current_state = next_state
            action = next_action
            action_idx = next_action_idx

        return episode_info

    # Used to pull the results of the training for scripted training in plots.py
    def get_results(self):
        return self.Q_table, \
            self.Return_table, \
            self.Goal_Step, \
            self.Episode_Time, \
            self.Success_Rate, \
            self.Rewards_List, \
            self.Episode_Cost, \
            [len(self.Goal_Step), len(self.Fail_Step)], \
            self.Q_Converge, \
            self.Policy_Changes_List

    # Plots the results of the training of Steps taken per successful episode, Time taken per episode,
    # Success rate per episode, and Rewards per episode, Costs per episode, and the number of successes
    # and failures and saves them to the results folder
    def plot_results(self):
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

    # Plots the convergence of the Q-Table and the number of policy changes per episode on the same plot
    def plot_convergence(self):
        fig, ax1 = plt.subplots(figsize=(7, 5))
        label = 'Gamma=' + str(self.GAMMA) + ' Lr=' + str(self.LEARNING_RATE)

        ax1.plot(list(self.Q_Converge.keys()), list(self.Q_Converge.values()), label=label, color='blue')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Mean-Squared Error", color='blue')
        ax1.legend()

        ax2 = ax1.twinx()

        ax2.plot(list(self.Policy_Changes_List.keys()), list(self.Policy_Changes_List.values()), label=label, color='green')
        ax2.set_ylabel("Policy Changes", color='red')
        ax2.legend()

        if self.LEARNING_RATE == 0:
            fig.suptitle(f"Convergence of {self.__class__.__name__} with Gamma={self.GAMMA}, Epsilon={self.EPSILON}")
        else:
            fig.suptitle(f"Convergence of {self.__class__.__name__} with Gamma={self.GAMMA}, Epsilon={self.EPSILON}, Learning Rate={self.LEARNING_RATE}")

        plt.tight_layout()
        plt.show()






