import argparse
import pickle
from sklearn.utils import shuffle
from scipy.special import softmax
from sklearn.linear_model import Ridge
import numpy as np
import sys
sys.path.append(".")
from src.oracles import LinearRegressionCSCOracle
from exp_params import SOFT_MAX_POLICY, BINARY_COST, reward_model_proportion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_bandit_feedback_data_path", type=str, help="the full-info data to simulate "
                                                                             "bandit feedback")
    parser.add_argument("--logging_policy_path", type=str, help="path of logging policy")
    parser.add_argument("--N", type=float, help="number of passes")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon in epsilon greedy logging policy")
    parser.add_argument('--train_data_path_IPW', type=str, help="path to save the training data for IPW")
    parser.add_argument('--valid_data_path_IPW', type=str, help="path to save the validation data for IPW")
    parser.add_argument('--train_data_path_DR', type=str, help="path to save the training data for DR")
    parser.add_argument('--valid_data_path_DR', type=str, help="path to save the validation data for DR")

    args = parser.parse_args()
    with open(args.simulate_bandit_feedback_data_path, 'rb') as f:
        X, L = pickle.load(f)
    with open(args.logging_policy_path, 'rb') as f:
        logging_policy = pickle.load(f)
    X_bandit = []
    A_bandit = []
    L_bandit = []
    P_bandit = []
    data_size = X.shape[0]
    num_labels = L.shape[1]
    sweeps = int(args.N % 1)
    rest_N = int((args.N - sweeps) * data_size)

    if SOFT_MAX_POLICY:
        outputs = logging_policy.predict_score(X)
        outputs = (1. - outputs) / args.epsilon
        P = softmax(outputs, axis=1)
        for sweep in range(sweeps):
            X, L, P = shuffle(X, L, P)
            for i in range(data_size):
                X_bandit.append(X[i])
                A_example = np.random.choice(num_labels, p=P[i])
                if BINARY_COST:
                    L_bandit.append(np.random.binomial(1, L[i, A_example]))
                else:
                    L_bandit.append(L[i, A_example])
                A_bandit.append(A_example)
                P_bandit.append(P[i])

        X, L, P = shuffle(X, L, P)
        for i in range(rest_N):
            X_bandit.append(X[i])
            A_example = np.random.choice(num_labels, p=P[i])
            if BINARY_COST:
                L_bandit.append(np.random.binomial(1, L[i, A_example]))
            else:
                L_bandit.append(L[i, A_example])
            A_bandit.append(A_example)
            P_bandit.append(P[i])
    else:
        A_hat = logging_policy.predict(X)
        A_hat = np.eye(L.shape[1])[A_hat]
        P = A_hat * (1 - args.epsilon) + args.epsilon / num_labels
        for sweep in range(sweeps):
            X, L, A_hat, P = shuffle(X, L, A_hat, P)
            for i in range(data_size):
                X_bandit.append(X[i])
                random_number = np.random.rand()
                if random_number < args.epsilon:
                    A_example = np.random.randint(0, num_labels)
                else:
                    A_example = np.argmax(A_hat[i])
                if BINARY_COST:
                    L_bandit.append(np.random.binomial(1, L[i, A_example]))
                else:
                    L_bandit.append(L[i, A_example])
                A_bandit.append(A_example)
                P_bandit.append(P[i])

        X, L, A_hat, P = shuffle(X, L, A_hat, P)
        for i in range(rest_N):
            X_bandit.append(X[i])
            random_number = np.random.rand()
            if random_number < args.epsilon:
                A_example = np.random.randint(0, num_labels)
            else:
                A_example = np.argmax(A_hat[i])
            if BINARY_COST:
                L_bandit.append(np.random.binomial(1, L[i, A_example]))
            else:
                L_bandit.append(L[i, A_example])
            A_bandit.append(A_example)
            P_bandit.append(P[i])

    X_bandit = np.array(X_bandit)
    A_bandit = np.array(A_bandit)
    L_bandit = np.array(L_bandit)
    P_bandit = np.array(P_bandit)

    bandit_data_size = X_bandit.shape[0]
    train_reward_data_size = int(bandit_data_size * reward_model_proportion)
    X_reward_model = X_bandit[:train_reward_data_size]
    L_reward_model = L_bandit[:train_reward_data_size]
    reward_model = Ridge(alpha=1e-3)
    reward_model.fit(X_reward_model, L_reward_model)
    X_bandit_DR = X_bandit[train_reward_data_size:]
    A_bandit_DR = A_bandit[train_reward_data_size:]
    L_bandit_DR = L_bandit[train_reward_data_size:]
    P_bandit_DR = P_bandit[train_reward_data_size:]

    loss_shift_DR = reward_model.predict(X_bandit_DR)
    data_size_IPW = X_bandit.shape[0]
    train_data_size_IPW = data_size_IPW // 2
    with open(args.train_data_path_IPW, 'wb') as f:
        pickle.dump((X_bandit[:train_data_size_IPW], A_bandit[:train_data_size_IPW], L_bandit[:train_data_size_IPW],
                     P_bandit[:train_data_size_IPW], 1.), f)
    with open(args.valid_data_path_IPW, 'wb') as f:
        pickle.dump((X_bandit[train_data_size_IPW:], A_bandit[train_data_size_IPW:], L_bandit[train_data_size_IPW:],
                     P_bandit[train_data_size_IPW:], 1.), f)

    data_size_DR = X_bandit_DR.shape[0]
    train_data_size_DR = data_size_DR // 2
    with open(args.train_data_path_DR, 'wb') as f:
        pickle.dump((X_bandit_DR[:train_data_size_DR], A_bandit_DR[:train_data_size_DR], L_bandit_DR[:train_data_size_DR],
                     P_bandit_DR[:train_data_size_DR], loss_shift_DR[:train_data_size_DR]), f)
    with open(args.valid_data_path_DR, 'wb') as f:
        pickle.dump((X_bandit_DR[train_data_size_DR:], A_bandit_DR[train_data_size_DR:], L_bandit_DR[train_data_size_DR:],
                     P_bandit_DR[train_data_size_DR:], loss_shift_DR[train_data_size_DR:]), f)
