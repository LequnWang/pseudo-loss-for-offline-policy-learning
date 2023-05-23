import argparse
import pickle
import numpy as np
import time
import sys
sys.path.append(".")
from src.oracles import LinearSoftmaxCSCOracle
import copy
from exp_utils import generate_cost, calculate_IPW, calculate_pseudo_loss, calculate_sample_variance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, help="the training bandit feeedback data path")
    parser.add_argument("--valid_data_path", type=str, help="the validation bandit feedback data path")
    parser.add_argument("--test_data_path", type=str, help="the full-info test data path")
    parser.add_argument("--model_path", type=str, help="the path to save the best model")
    parser.add_argument("--result_path", type=str, help="the path to save the validation results of the best model")
    parser.add_argument("--beta", type=float, help="weight of the pseudo loss")
    parser.add_argument("--weight_decay", type=float, help="l2 regularization strength")
    parser.add_argument("--lrs", nargs='+', type=float, help="list of learning rates")
    parser.add_argument("--n_restarts", type=int, default=1, help="number of restarts")

    args = parser.parse_args()
    with open(args.train_data_path, 'rb') as f:
        X_train, A_train, L_train, P_train, loss_shift_train = pickle.load(f)
        L_train = L_train - loss_shift_train
    C_train = generate_cost(A_train, L_train, P_train, args.beta)
    best_loss = float("inf")
    best_lr = None
    best_oracle = None
    start_time = time.time()
    for lr in args.lrs:
        for restart in range(args.n_restarts):
            oracle = LinearSoftmaxCSCOracle(weight_decay=args.weight_decay, lr=lr)
            oracle.fit(X_train, C_train)
            loss = oracle.loss(X_train, C_train)
            if loss < best_loss:
                best_loss = loss
                best_lr = lr
                best_oracle = copy.deepcopy(oracle)

    results = {}
    results["best_lr"] = best_lr
    results["time"] = time.time() - start_time
    with open(args.valid_data_path, 'rb') as f:
        X_valid, A_valid, L_valid, P_valid, loss_shift_valid = pickle.load(f)

    A_hat_valid = best_oracle.predict(X_valid)
    A_hat_valid = np.eye(P_valid.shape[1])[A_hat_valid]
    IPW_estimate = calculate_IPW(np.eye(P_valid.shape[1])[A_valid], A_hat_valid, L_valid, loss_shift_valid, P_valid)
    pseodo_loss = calculate_pseudo_loss(A_hat_valid, P_valid)
    sample_variance = calculate_sample_variance(np.eye(P_valid.shape[1])[A_valid], A_hat_valid, L_valid, loss_shift_valid, P_valid)
    results["IPW"] = IPW_estimate
    results["pseudo_loss"] = pseodo_loss
    results["sample_variance"] = sample_variance
    with open(args.test_data_path, 'rb') as f:
        X_test, L_test = pickle.load(f)
    A_hat_test = best_oracle.predict(X_test)
    A_hat_test = np.eye(L_test.shape[1])[A_hat_test]
    test_loss = np.sum(A_hat_test * L_test) / L_test.shape[0]
    results["test_loss"] = test_loss
    results["valid_data_size"] = X_valid.shape[0]
    print(results)
    with open(args.result_path, 'wb') as f:
        pickle.dump(results, f)
    with open(args.model_path, 'wb') as f:
        pickle.dump(best_oracle, f)



