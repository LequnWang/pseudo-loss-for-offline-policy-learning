import argparse
import pickle
import numpy as np
import sys
import time
sys.path.append(".")
from src.oracles import LinearSoftmaxEBOracle
import copy
from exp_utils import generate_cost, calculate_IPW, calculate_pseudo_loss, calculate_sample_variance
from exp_params import cont_loss

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
    parser.add_argument("--k", type=int, help="number of discrete actions")
    parser.add_argument("--h", type=float, help="the bandwidth")
    parser.add_argument("--logging_policy_path", type=str, help="the logging policy path")
    parser.add_argument("--n_restarts", type=int, default=1, help="number of restarts")

    args = parser.parse_args()
    with open(args.logging_policy_path, 'rb') as f:
        logging_policy = pickle.load(f)
    with open(args.train_data_path, 'rb') as f:
        X_train, A_train, L_train, loss_shift_train = pickle.load(f)
        L_train = L_train - loss_shift_train
    if not type(loss_shift_train) == np.ndarray:
        loss_shift_train = np.array([loss_shift_train] * X_train.shape[0])

    C_train = generate_cost(X_train, A_train, L_train, logging_policy, args.k, args.h, args.beta)
    best_loss = float("inf")
    best_lr = None
    best_oracle = None
    start_time = time.time()
    for lr in args.lrs:
        for restart in range(args.n_restarts):
            oracle = LinearSoftmaxEBOracle(weight_decay=args.weight_decay, lr=lr)
            oracle.fit(X_train, C_train, loss_shift_train, args.beta)
            loss = oracle.loss(X_train, C_train, loss_shift_train, args.beta)
            if loss < best_loss:
                best_loss = loss
                best_lr = lr
                best_oracle = copy.deepcopy(oracle)

    results = {}
    results["time"] = time.time() - start_time
    results["best_lr"] = best_lr
    with open(args.valid_data_path, 'rb') as f:
        X_valid, A_valid, L_valid, loss_shift_valid = pickle.load(f)

    A_hat_valid = best_oracle.predict(X_valid)
    print(A_hat_valid.shape)
    IPW_estimate = calculate_IPW(X_valid, A_valid, A_hat_valid, L_valid, loss_shift_valid, args.k, args.h, logging_policy)
    pseodo_loss = calculate_pseudo_loss(X_valid, A_hat_valid, args.k, args.h, logging_policy)
    sample_variance = calculate_sample_variance(X_valid, A_valid, A_hat_valid, L_valid, loss_shift_valid, args.k, args.h, logging_policy)

    results["IPW"] = IPW_estimate
    results["pseudo_loss"] = pseodo_loss
    results["sample_variance"] = sample_variance

    with open(args.test_data_path, 'rb') as f:
        X_test, Y_test = pickle.load(f)
    A_hat_test = best_oracle.predict(X_test)
    A_hat_test = (A_hat_test + 0.5) / (args.k - 1)
    test_loss = np.average([cont_loss(Y_test[i], A_hat_test[i]) for i in range(Y_test.shape[0])])
    results["test_loss"] = test_loss
    results["valid_data_size"] = X_valid.shape[0]
    results["h"] = args.h
    print(results)
    with open(args.result_path, 'wb') as f:
        pickle.dump(results, f)
    with open(args.model_path, 'wb') as f:
        pickle.dump(best_oracle, f)



