from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import Ridge
import pickle
import os
import numpy as np
from collections import defaultdict
from exp_params import *
from exp_utils import LoggingPolicyCont
from scipy import stats


def floatorzero(element):
    try:
        return float(element)
    except ValueError:
        return 0.0
def isfloat(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def fetch_data(data_name, data_home, train_logging_policy_data_path, test_data_path, simulate_bandit_feedback_data_path,
               train_logging_policy_data_size, test_proportion):
    extras = defaultdict(set)
    csv_filename = os.path.join(data_home, data_name + ".csv")
    if not os.path.isfile(csv_filename):
        print(csv_filename)
        if not os.path.isdir("./data"):
            os.system("mkdir ./data")
        os.system("wget https://www.openml.org/data/get_csv/150677/BNG_wisconsin.arff -O ./data/BNG_wisconsin.csv")
        os.system("sed '/^$/d' -i ./data/BNG_wisconsin.csv")
        os.system("python ./scripts_continuous/preprocess_data.py --csv_file ./data/BNG_wisconsin.csv")
        os.system("wget https://www.openml.org/data/get_csv/150680/BNG_cpu_act.arff -O ./data/BNG_cpu_act.csv")
        os.system("sed '/^$/d' -i ./data/BNG_cpu_act.csv")
        os.system("python ./scripts_continuous/preprocess_data.py --csv_file ./data/BNG_cpu_act.csv")
        os.system("wget https://www.openml.org/data/get_csv/150679/BNG_auto_price.arff -O ./data/BNG_auto_price.csv")
        os.system("sed '/^$/d' -i ./data/BNG_auto_price.csv")
        os.system("python ./scripts_continuous/preprocess_data.py --csv_file ./data/BNG_auto_price.csv")
        os.system("wget https://www.openml.org/data/get_csv/21230845/file639340bd9ca9.arff -O ./data/black_friday.csv")
        os.system("sed '/^$/d' -i ./data/black_friday.csv")
        os.system("python ./scripts_continuous/preprocess_data.py --csv_file ./data/black_friday.csv")
        os.system("wget https://www.openml.org/data/get_csv/5698591/file62a9329beed2.arff -O ./data/zurich.csv")
        os.system("sed '/^$/d' -i ./data/zurich.csv")
        os.system("python ./scripts_continuous/preprocess_data.py --csv_file ./data/zurich.csv")
    filename = os.path.join(data_home, data_name + ".dat")
    with open(filename, 'r') as f:
        for line in f:
            targetstr, rest = line.strip().split('|')
            target = float(targetstr)
            stringfeatures = rest.split()
            for col, (isnum, v) in enumerate((isfloat(x), x) for x in stringfeatures):
                if not isnum:
                    extras[col].add(v)

    onehotmap = {}
    for col, values in extras.items():
        for v in values:
            if (col, v) not in onehotmap:
                onehotmap[col, v] = len(onehotmap)

    print(f'creating {len(onehotmap)} additional one-hot columns')

    Y = []
    X = []
    with open(filename, 'r') as f:
        for line in f:
            targetstr, rest = line.strip().split('|')
            target = float(targetstr)
            stringfeatures = rest.split()
            features = [0] * len(onehotmap) + [floatorzero(x) for x in stringfeatures]
            for col, v in enumerate(stringfeatures):
                if (col, v) in onehotmap:
                    features[onehotmap[col, v]] = 1

            Y.append(target)
            X.append(features)
    X = np.array(X)
    Y = np.array(Y)
    print(filename)
    print(X.shape)
    print(Y.shape)
    test_size = int((Y.shape[0] - train_logging_policy_data_size) * test_proportion)
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    if use_percentile:
        Y = (stats.rankdata(Y) - 1.) / (len(Y) - 1.)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X, Y = shuffle(X, Y)
    print(X.shape)
    print(Y.shape)
    with open(train_logging_policy_data_path, 'wb') as f:
        pickle.dump((X[: train_logging_policy_data_size], Y[: train_logging_policy_data_size]), f)
    with open(simulate_bandit_feedback_data_path, 'wb') as f:
        pickle.dump((X[train_logging_policy_data_size: -test_size], Y[train_logging_policy_data_size: -test_size]), f)
    with open(test_data_path, 'wb') as f:
        pickle.dump((X[-test_size:], Y[-test_size:]), f)
    return X[: train_logging_policy_data_size], Y[: train_logging_policy_data_size], X[-test_size:], Y[-test_size:]


if __name__ == "__main__":
    if not os.path.isdir(exp_dir):
        os.system("mkdir " + exp_dir)
    for data_name in data_names:
        train_logging_policy_data_path = os.path.join(exp_dir, data_name + ".train_logging_policy.pkl")
        test_data_path = os.path.join(exp_dir, data_name + ".test.pkl")
        simulate_bandit_feedback_data_path = os.path.join(exp_dir, data_name + ".simulate_bandit_feedback.pkl")
        X_log, Y_log, X_test, Y_test = fetch_data(data_name, data_home, train_logging_policy_data_path, test_data_path,
                                  simulate_bandit_feedback_data_path, train_logging_policy_data_size, test_proportion)
        logging_regression_model = Ridge(alpha=0.1)
        logging_regression_model.fit(X_log, Y_log)
        logging_policy = LoggingPolicyCont(logging_regression_model, logging_h, epsilon)
        logging_policy_path = os.path.join(exp_dir, data_name + ".logging_policy.pkl")
        with open(logging_policy_path, 'wb') as f:
            pickle.dump(logging_policy, f)
        #
        # evaluate logging policy performance
        Y_hat_test = logging_regression_model.predict(X_test)
        average_cost = np.average([cont_loss(Y_test[i], Y_hat_test[i]) for i in range(Y_test.shape[0])])
        print(data_name, "average cost of logging policy: {}".format(average_cost))

