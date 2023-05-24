import argparse
import pickle
from sklearn.utils import shuffle
from sklearn.linear_model import Ridge
import numpy as np
import sys
sys.path.append(".")
from exp_utils import LoggingPolicyCont
from exp_params import cont_loss, reward_model_proportion, loss_width

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_bandit_feedback_data_path", type=str, help="the full-info data to simulate "
                                                                             "bandit feedback")
    parser.add_argument("--logging_policy_path", type=str, help="path of logging policy")
    parser.add_argument("--N", type=float, help="number of passes")
    parser.add_argument('--train_data_path_IPW', type=str, help="path to save the training data for IPW")
    parser.add_argument('--valid_data_path_IPW', type=str, help="path to save the validation data for IPW")
    parser.add_argument('--train_data_path_DR', type=str, help="path to save the training data for DR")
    parser.add_argument('--valid_data_path_DR', type=str, help="path to save the validation data for DR")

    args = parser.parse_args()
    with open(args.simulate_bandit_feedback_data_path, 'rb') as f:
        X, Y = pickle.load(f)
    with open(args.logging_policy_path, 'rb') as f:
        logging_policy = pickle.load(f)
    X_bandit = []
    L_bandit = []
    A_bandit = []
    data_size = X.shape[0]
    sweeps = int(args.N % 1)
    rest_N = int((args.N - sweeps) * data_size)
    for sweep in range(sweeps):
        X, Y = shuffle(X, Y)
        for i in range(data_size):
            X_bandit.append(X[i])
            A_example = logging_policy.sample_action(X[i])
            L_bandit.append(cont_loss(Y[i], A_example))
            A_bandit.append(A_example)

    X, Y = shuffle(X, Y)
    for i in range(rest_N):
        X_bandit.append(X[i])
        A_example = logging_policy.sample_action(X[i])
        L_bandit.append(cont_loss(Y[i], A_example))
        A_bandit.append(A_example)

    X_bandit = np.array(X_bandit)
    A_bandit = np.array(A_bandit)
    L_bandit = np.array(L_bandit)

    bandit_data_size = X_bandit.shape[0]
    train_reward_data_size = int(bandit_data_size * reward_model_proportion)
    X_reward_model = X_bandit[:train_reward_data_size]
    L_reward_model = L_bandit[:train_reward_data_size]
    reward_model = Ridge()
    reward_model.fit(X_reward_model, L_reward_model)
    X_bandit_DR = X_bandit[train_reward_data_size:]
    A_bandit_DR = A_bandit[train_reward_data_size:]
    L_bandit_DR = L_bandit[train_reward_data_size:]

    loss_shift_DR = reward_model.predict(X_bandit_DR)

    data_size_IPW = X_bandit.shape[0]
    train_data_size_IPW = data_size_IPW // 2
    with open(args.train_data_path_IPW, 'wb') as f:
        pickle.dump((X_bandit[:train_data_size_IPW], A_bandit[:train_data_size_IPW], L_bandit[:train_data_size_IPW], 1.), f)
    with open(args.valid_data_path_IPW, 'wb') as f:
        pickle.dump((X_bandit[train_data_size_IPW:], A_bandit[train_data_size_IPW:], L_bandit[train_data_size_IPW:], 1.), f)

    data_size_DR = X_bandit_DR.shape[0]
    train_data_size_DR = data_size_DR // 2
    with open(args.train_data_path_DR, 'wb') as f:
        pickle.dump((X_bandit_DR[:train_data_size_DR], A_bandit_DR[:train_data_size_DR], L_bandit_DR[:train_data_size_DR], loss_shift_DR[:train_data_size_DR]), f)
    with open(args.valid_data_path_DR, 'wb') as f:
        pickle.dump((X_bandit_DR[train_data_size_DR:], A_bandit_DR[train_data_size_DR:], L_bandit_DR[train_data_size_DR:], loss_shift_DR[train_data_size_DR:]), f)
