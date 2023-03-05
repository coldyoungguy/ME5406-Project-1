import random
import time
from params import *
import matplotlib.pyplot as plt

class BaseAlgo(object):
    def __init__(self, env, ep, gamma, learning_rate=None):
        self.env = env
        self.ACTION_SPACE = self.env.action_space
        self.N_ACTION = len(self.ACTION_SPACE)
        self.EPSILON = ep
        self.GAMMA = gamma
        self.learning_rate = learning_rate
        self.Q_table, self.Return_table, self.Num_StateAction = self.init_tables()
        self.Q_table, self.Return_table, self.Num_StateAction = self.init_tables()
        self.Episode_Step, self.Episode_Time = {}, {}
        self.goal_count, self.fail_count, self.total_rewards= 0, 0, 0
        self.Goal_Step, self.Fail_Step = {}, {}
        self.Rewards_List = {} # average rewards over time
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
            max_ = max(return_values)
            if type(max_) == list:
                action_idx = random.choice([i for i in range(len(return_values)) if return_values[i] == max(return_values)])
            else:
                action_idx = return_values.index(max_)
            action = self.ACTION_SPACE[action_idx]
        else:
            action = random.choice(self.ACTION_SPACE)
        return action

    def find_valid_action(self, state):
        action = self.generate_policy(state)
        action_idx = self.ACTION_SPACE.index(action)
        while self.Q_table[state][action_idx] == float('-inf'):
            action = self.generate_policy(state)
            action_idx = self.ACTION_SPACE.index(action)
        return action, action_idx

    def generate_episode(self, episode, method=None):
        step, cost = 0, 0
        current_state = self.env.reset()
        action, action_idx = self.find_valid_action(current_state)
        episode_info = []
        fps = self.env.fps

        episode_start_time = time.time()

        for _ in range(NUM_STEPS):
            time.sleep((1 / fps) if fps != 0 else 0)
            next_state, reward, is_done = self.env.step(action)
            next_action, next_action_idx = self.find_valid_action(next_state)

            if method is not None:
                cost += method(current_state, action_idx, reward, next_state, next_action_idx)

            current_state = next_state
            action = next_action
            action_idx = next_action_idx
            episode_info.append((current_state, action_idx, reward))

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

                self.total_rewards += reward if reward != float('-inf') else 0
                self.Rewards_List[episode] = self.total_rewards / episode

                print(f"Episode {episode} finished in {step} steps in {episode_elapsed_time}")
                break

        return episode_info

    def plot_results(self, path=None):
        # Plot the results of accuracy, rewards, and success rate over episodes
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        ax[0, 0].plot(list(self.Episode_Step.keys()), list(self.Episode_Step.values()))
        ax[0, 0].set_title("Steps Taken per Episode")
        ax[0, 0].set_xlabel("Episode")
        ax[0, 0].set_ylabel("Steps")

        ax[0, 1].plot(list(self.Episode_Time.keys()), list(self.Episode_Time.values()))
        ax[0, 1].set_title("Time Taken per Episode")
        ax[0, 1].set_xlabel("Episode")
        ax[0, 1].set_ylabel("Time")

        ax[1, 0].plot(list(self.Success_Rate.keys()), list(self.Success_Rate.values()))
        ax[1, 0].set_title(f"Average Success Rate over {ACCURACY_RANGE} episodes")
        ax[1, 0].set_xlabel("Episode")
        ax[1, 0].set_ylabel("Success Rate")

        ax[1, 1].plot(list(self.Rewards_List.keys()), list(self.Rewards_List.values()))
        ax[1, 1].set_title(f"Average Rewards over Episodes")
        ax[1, 1].set_xlabel("Episode")
        ax[1, 1].set_ylabel("Rewards")

        if self.learning_rate is None:
            fig.suptitle(f"Results of {self.__class__.__name__} with Gamma={self.GAMMA}, Epsilon={self.EPSILON}")
        else:
            fig.suptitle(f"Results of {self.__class__.__name__} with Gamma={self.GAMMA}, Epsilon={self.EPSILON}, Learning Rate={self.learning_rate}")
        plt.tight_layout()

        if path is not None:
            fig.savefig(path)
        else:
            plt.show()







