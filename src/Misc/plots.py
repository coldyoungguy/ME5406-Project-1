from src.Algorithms.Q_learning import QLearning
from src.Algorithms.SARSA import SARSA
import matplotlib.pyplot as plt
from schedulers import *

gamma_testing_set = [0.1, 0.3, 0.5, 0.7, 0.9]
learning_rate_testing_set = [0.001, 0.01, 0.1, 0.3, 0.5]
ep_testing_set = [0.1, 0.3, 0.5, 0.7, 0.9]
schedule_testing_set = {"Linear Schedule": lin_schedule,
                        "Exponential Schedule": exp_schedule,
                        "Discrete Schedule": discrete_schedule,
                        "Drop Schedule": drop_schedule}

def iterate_lr(env, model):
    # Goal_Steps_All, Episode_Time_All, Success_Rate_All, RewardsList_All, Episode_Cost_All, Results_All= {}, {}, {}, {}, {}, {}
    Q_Converge_All, Policy_Changes_List_All = {}, {}
    gamma = 0.9
    for lr in learning_rate_testing_set:
        algo = model(env, EPSILON, gamma, lr)
        algo.run(NUM_EPISODES, is_train=True)
        _, _, _, _, _, _, _, _, Q_Converge_All[lr], Policy_Changes_List_All[lr] = algo.get_results()

    for label in schedule_testing_set.keys():
        algo = model(env, EPSILON, gamma, LEARNING_RATE)
        algo.run(NUM_EPISODES, is_train=True, lr_schedule=schedule_testing_set[label])
        _, _, _, _, _, _, _, _, Q_Converge_All[label], Policy_Changes_List_All[label] = algo.get_results()

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    print(Q_Converge_All)
    print(Policy_Changes_List_All)
    for lr in learning_rate_testing_set:
        label = f"{lr}"
        ax[0].plot(list(Q_Converge_All[lr].keys()), list(Q_Converge_All[lr].values()), label=label)
        ax[0].set_title("Convergence of Q Values")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("MSE of Q Values")
        ax[0].legend()

        ax[1].plot(list(Policy_Changes_List_All[lr].keys()), list(Policy_Changes_List_All[lr].values()), label=label)
        ax[1].set_title("Convergence of Policy Changes")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Number of Policy Changes")
        ax[1].legend()

    for label in schedule_testing_set.keys():
        fig_label = f"{label}"
        ax[0].plot(list(Q_Converge_All[label].keys()), list(Q_Converge_All[label].values()), label=fig_label)
        ax[0].set_title("Convergence of Q Values")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("MSE of Q Values")
        ax[0].legend()

        ax[1].plot(list(Policy_Changes_List_All[label].keys()), list(Policy_Changes_List_All[label].values()), label=fig_label)
        ax[1].set_title("Convergence of Policy Changes")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Number of Policy Changes")
        ax[1].legend()

    fig.suptitle(f"Results of {model.__name__} with Epsilon={EPSILON}, Gamma={gamma}")
    plt.tight_layout()
    PATH_LR_SUMMARY = '../../Results/' + model.__name__ + '/Summary_lr.png'
    fig.savefig(PATH_LR_SUMMARY, dpi=300)

def iterate_gamma(env, model):
    # Goal_Steps_All, Episode_Time_All, Success_Rate_All, RewardsList_All, Episode_Cost_All, Results_All= {}, {}, {}, {}, {}, {}
    Q_Converge_All, Policy_Changes_List_All = {}, {}
    lr = 0.1

    for gamma in gamma_testing_set:
        algo = model(env, EPSILON, gamma, lr)
        algo.run(NUM_EPISODES, is_train=True)
        _, _, _, _, _, _, _, _, Q_Converge_All[gamma], Policy_Changes_List_All[gamma] = algo.get_results()

    for label in schedule_testing_set.keys():
        algo = model(env, EPSILON, GAMMA, lr)
        algo.run(NUM_EPISODES, is_train=True, gamma_schedule=schedule_testing_set[label])
        _, _, _, _, _, _, _, _, Q_Converge_All[label], Policy_Changes_List_All[label] = algo.get_results()

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for gamma in gamma_testing_set:
        label = f"{gamma}"
        ax[0].plot(list(Q_Converge_All[gamma].keys()), list(Q_Converge_All[gamma].values()), label=label)
        ax[0].set_title("Convergence of Q Values")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("MSE of Q Values")
        ax[0].legend()

        ax[1].plot(list(Policy_Changes_List_All[gamma].keys()), list(Policy_Changes_List_All[gamma].values()), label=label)
        ax[1].set_title("Convergence of Policy Changes")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Number of Policy Changes")
        ax[1].legend()

    for label in schedule_testing_set.keys():
        fig_label = f"{label}"
        ax[0].plot(list(Q_Converge_All[label].keys()), list(Q_Converge_All[label].values()), label=fig_label)
        ax[0].set_title("Convergence of Q Values")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("MSE of Q Values")
        ax[0].legend()

        ax[1].plot(list(Policy_Changes_List_All[label].keys()), list(Policy_Changes_List_All[label].values()), label=fig_label)
        ax[1].set_title("Convergence of Policy Changes")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Number of Policy Changes")
        ax[1].legend()

    fig.suptitle(f"Results of {model.__name__} with Epsilon={EPSILON}, Lr={lr}")
    plt.tight_layout()
    PATH_GAMMA_SUMMARY = '../../Results/' + model.__name__ + '/Summary_gamma.png'
    fig.savefig(PATH_GAMMA_SUMMARY, dpi=300)

def iterate_ep(env, model):
    # Goal_Steps_All, Episode_Time_All, Success_Rate_All, RewardsList_All, Episode_Cost_All, Results_All= {}, {}, {}, {}, {}, {}
    Q_Converge_All, Policy_Changes_List_All = {}, {}
    gamma= 0.9
    lr = 0.1

    for ep in ep_testing_set:
        algo = model(env, ep, gamma, lr)
        algo.run(NUM_EPISODES, is_train=True)
        _, _, _, _, _, _, _, _, Q_Converge_All[ep], Policy_Changes_List_All[ep] = algo.get_results()

    for label in schedule_testing_set.keys():
        algo = model(env, 1, gamma, lr)
        algo.run(NUM_EPISODES, is_train=True, ep_schedule=schedule_testing_set[label])
        _, _, _, _, _, _, _, _, Q_Converge_All[label], Policy_Changes_List_All[label] = algo.get_results()

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for ep in ep_testing_set:
        label = f"{ep}"
        ax[0].plot(list(Q_Converge_All[ep].keys()), list(Q_Converge_All[ep].values()), label=label)
        ax[0].set_title("Convergence of Q Values")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("MSE of Q Values")
        ax[0].legend()

        ax[1].plot(list(Policy_Changes_List_All[ep].keys()), list(Policy_Changes_List_All[ep].values()), label=label)
        ax[1].set_title("Convergence of Policy Changes")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Number of Policy Changes")
        ax[1].legend()

    for label in schedule_testing_set.keys():
        fig_label = f"{label}"
        ax[0].plot(list(Q_Converge_All[label].keys()), list(Q_Converge_All[label].values()), label=fig_label)
        ax[0].set_title("Convergence of Q Values")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("MSE of Q Values")
        ax[0].legend()

        ax[1].plot(list(Policy_Changes_List_All[label].keys()), list(Policy_Changes_List_All[label].values()), label=fig_label)
        ax[1].set_title("Convergence of Policy Changes")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Number of Policy Changes")
        ax[1].legend()

    fig.suptitle(f"Results of {model.__name__} with Gamma={gamma}, Lr={lr}")
    plt.tight_layout()
    PATH_GAMMA_SUMMARY = '../../Results/' + model.__name__ + '/Summary_EP.png'
    fig.savefig(PATH_GAMMA_SUMMARY, dpi=300)

# For used when running this python file by itself.
if __name__ == '__main__':
    from src.Environment.environment import Env
    env = Env()
    # iterate_lr(env, QLearning)
    # iterate_gamma(env, QLearning)
    # iterate_lr(env, SARSA)
    # iterate_gamma(env, SARSA)
    iterate_ep(env, QLearning)

