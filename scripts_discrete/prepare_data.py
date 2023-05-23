from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import pickle
import os
import sys
import numpy as np
sys.path.append(".")
from exp_params import *
from src.oracles import LinearRegressionCSCOracle



def generate_real_cost(Y):
    num_classes_original = np.max(Y) + 1
    num_classes = num_classes_original * TIMES_NUM_CLASSES
    cost_matrix = np.random.rand(num_classes_original, num_classes)
    for i in range(num_classes_original):
        for j in range(TIMES_NUM_CLASSES):
            cost_matrix[i, num_classes_original * j + i] = 0.
    C = []
    for y in Y:
        C.append(cost_matrix[y, :])
    C = np.array(C)
    return C

def generate_cost(Y):
    num_classes_original = np.max(Y) + 1
    num_classes = num_classes_original * TIMES_NUM_CLASSES
    cost_matrix = np.ones((num_classes_original, num_classes))
    for i in range(num_classes_original):
        for j in range(TIMES_NUM_CLASSES):
            cost_matrix[i, num_classes_original * j + i] = 0.
    C = []
    for y in Y:
        C.append(cost_matrix[y, :])
    C = np.array(C)
    return C


def fetch_data(data_id, data_home, train_logging_policy_data_path, test_data_path, simulate_bandit_feedback_data_path,
               train_logging_policy_data_size, test_proportion):
    dataset = fetch_openml(data_id=data_id, data_home=data_home, cache=True, as_frame=False)
    X = dataset.data
    Y = dataset.target
    test_size = int((Y.shape[0] - train_logging_policy_data_size) * test_proportion)

    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    if GENERATE_REAL_COST:
        C = generate_real_cost(Y)
    else:
        C = generate_cost(Y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X, C = shuffle(X, C)
    print(X.shape)
    # print("valid data size {}".format(int((X.shape[0] - train_logging_policy_data_size) * (1 - test_proportion) / 2)))
    print(C.shape)
    with open(train_logging_policy_data_path, 'wb') as f:
        pickle.dump((X[: train_logging_policy_data_size], C[: train_logging_policy_data_size]), f)
    with open(simulate_bandit_feedback_data_path, 'wb') as f:
        pickle.dump((X[train_logging_policy_data_size: -test_size], C[train_logging_policy_data_size: -test_size]), f)
    with open(test_data_path, 'wb') as f:
        pickle.dump((X[-test_size:], C[-test_size:]), f)
    return X[: train_logging_policy_data_size], C[: train_logging_policy_data_size], X[-test_size:], C[-test_size:]


if __name__ == "__main__":
    for data_id in data_ids:
        print(data_id)
        data_name = data_id_to_name[data_id]
        print(data_name)
        train_logging_policy_data_path = os.path.join(exp_dir, data_name + ".train_logging_policy.pkl")
        test_data_path = os.path.join(exp_dir, data_name + ".test.pkl")
        simulate_bandit_feedback_data_path = os.path.join(exp_dir, data_name + ".simulate_bandit_feedback.pkl")
        X_log, C_log, X_test, C_test = fetch_data(data_id, exp_dir, train_logging_policy_data_path, test_data_path,
                                  simulate_bandit_feedback_data_path, train_logging_policy_data_size, test_proportion)
        logging_policy = LinearRegressionCSCOracle(epochs=10)
        if TEST_BAD_POLICY:
            logging_policy.fit(X_log, 1. - C_log)
        else:
            logging_policy.fit(X_log, C_log)
        logging_policy_path = os.path.join(exp_dir, data_name + ".logging_policy.pkl")
        with open(logging_policy_path, 'wb') as f:
            pickle.dump(logging_policy, f)

        # evaluate logging policy performance
        Y_hat_test = logging_policy.predict(X_test)
        Y_hat_test = np.eye(C_test.shape[1])[Y_hat_test]
        average_cost = np.sum(Y_hat_test * C_test) / C_test.shape[0]
        print(data_name, "average cost of logging policy: {}".format(average_cost))

