import os
import numpy as np
import random


class LoggingPolicyCont:
    def __init__(self, regression_model, logging_h, epsilon):
        self.regression_model = regression_model
        self.logging_h = logging_h
        self.epsilon = epsilon

    def sample_action(self, x):
        y_hat = self.regression_model.predict(x.reshape(1, -1))[0]
        y_hat = min(y_hat, 1.)
        y_hat = max(y_hat, 0.)
        greedy_start = max(y_hat - self.logging_h / 2., 0.)
        greedy_end = min(y_hat + self.logging_h / 2., 1.)
        logging_h_x = greedy_end - greedy_start

        # first sample interval, then sample action
        prob_each_interval = [(greedy_start - 0.) * self.epsilon,
                              (greedy_end - greedy_start) * (self.epsilon + (1. - self.epsilon) / logging_h_x),
                              (1. - greedy_end) * self.epsilon]
        interval = np.random.choice(3, p=prob_each_interval)
        if interval == 0:
            a = np.random.uniform(0., greedy_start)
        elif interval == 1:
            a = np.random.uniform(greedy_start, greedy_end)
        else:
            a = np.random.uniform(greedy_end, 1.)
        return a

    def probability_density(self, x, a):
        y_hat = self.regression_model.predict(x.reshape(1, -1))[0]
        y_hat = min(y_hat, 1.)
        y_hat = max(y_hat, 0.)
        if y_hat - self.logging_h / 2. < a < y_hat + self.logging_h / 2.:
            greedy_start = max(y_hat - self.logging_h / 2., 0.)
            greedy_end = min(y_hat + self.logging_h / 2., 1.)
            logging_h_x = greedy_end - greedy_start
            return self.epsilon + (1. - self.epsilon) / logging_h_x
        else:
            return self.epsilon

    def inverse_density_integral(self, x, start, end):
        y_hat = self.regression_model.predict(x.reshape(1, -1))[0]
        y_hat = min(y_hat, 1.)
        y_hat = max(y_hat, 0.)
        greedy_start = max(y_hat - self.logging_h / 2., 0.)
        greedy_end = min(y_hat + self.logging_h / 2., 1.)
        logging_h_x = greedy_end - greedy_start
        overlap = 0.
        if start >= greedy_end or end <= greedy_start:
            overlap = 0.
        elif start <= greedy_start and end >= greedy_end:
            overlap = greedy_end - greedy_start
        elif greedy_start <= start and greedy_end >= end:
            overlap = end - start
        elif start <= greedy_end and end >= greedy_end:
            overlap = greedy_end - start
        elif end >= greedy_start and start <= greedy_start:
            overlap = end - greedy_start
        else:
            raise ValueError("inverse mass integral case incomplete")

        return overlap / (self.epsilon + (1. - self.epsilon) / logging_h_x) + (end - start - overlap) / self.epsilon

def generate_cost(X_train, A_train, L_train, logging_policy, k, h, beta):
    num_examples = X_train.shape[0]
    C_train = np.zeros((num_examples, k))
    for i in range(num_examples):
        for j in range(k):
            smooth_action = j * 1. / (k - 1)
            smooth_action_start = max(smooth_action - h / 2, 0.)
            smooth_action_end = min(smooth_action + h / 2, 1.)
            smooth_action_h = smooth_action_end - smooth_action_start
            if smooth_action_start <= A_train[i] <= smooth_action_end:
                C_train[i, j] += L_train[i] / smooth_action_h / logging_policy.probability_density(X_train[i], A_train[i])
            C_train[i, j] += beta * logging_policy.inverse_density_integral(X_train[i], smooth_action_start, smooth_action_end) / smooth_action_h
    return C_train

def calculate_IPW(X, A, A_hat, L, loss_shift, k, h, logging_polcy):
    # A_hat selected action from k actions
    # A is in [0, 1]
    L = L - loss_shift
    num_examples = X.shape[0]
    results = np.zeros((num_examples))
    for i in range(num_examples):
        smooth_action = A_hat[i] * 1. / (k - 1)
        smooth_action_start = max(smooth_action - h / 2, 0.)
        smooth_action_end = min(smooth_action + h / 2, 1.)
        smooth_action_h = smooth_action_end - smooth_action_start
        if smooth_action_start <= A[i] <= smooth_action_end:
            results[i] += L[i] / smooth_action_h / logging_polcy.probability_density(X[i], A[i])
    results = results + loss_shift
    return np.mean(results)

def calculate_pseudo_loss(X, A_hat, k, h, logging_policy):
    num_examples = X.shape[0]
    results = np.zeros((num_examples))
    for i in range(num_examples):
        smooth_action = A_hat[i] * 1. / (k - 1)
        smooth_action_start = max(smooth_action - h / 2, 0.)
        smooth_action_end = min(smooth_action + h / 2, 1.)
        smooth_action_h = smooth_action_end - smooth_action_start
        results[i] += logging_policy.inverse_density_integral(X[i], smooth_action_start, smooth_action_end) / smooth_action_h
    return np.average(results)

def calculate_sample_variance(X, A, A_hat, L, loss_shift, k, h, logging_polcy):
    L = L - loss_shift
    num_examples = X.shape[0]
    results = np.zeros((num_examples))
    for i in range(num_examples):
        smooth_action = A_hat[i] * 1. / (k - 1)
        smooth_action_start = max(smooth_action - h / 2, 0.)
        smooth_action_end = min(smooth_action + h / 2, 1.)
        smooth_action_h = smooth_action_end - smooth_action_start
        if smooth_action_start <= A[i] <= smooth_action_end:
            results[i] += L[i] / smooth_action_h / logging_polcy.probability_density(X[i], A[i])
    results = results + loss_shift
    sample_mean = np.mean(results)
    return np.sum((results - sample_mean) ** 2) / (results.shape[0] - 1)


def data_generation_commands(exp_dir, data_names, data_sizes, n_runs):
    """generate a list of commands to simulate bandit feedback from the experiment setup"""
    commands = []
    for data_name in data_names:
        logging_policy_path = os.path.join(exp_dir, data_name + ".logging_policy.pkl")
        simulate_bandit_feedback_data_path = os.path.join(exp_dir, data_name + ".simulate_bandit_feedback.pkl")
        for data_size in data_sizes:
            for run in range(n_runs):
                feedback_identity_string = "_".join([data_name, str(data_size), str(run)])
                train_data_path_IPW = os.path.join(exp_dir, feedback_identity_string + "_train_data_IPW.pkl")
                valid_data_path_IPW = os.path.join(exp_dir, feedback_identity_string + "_valid_data_IPW.pkl")
                train_data_path_DR = os.path.join(exp_dir, feedback_identity_string + "_train_data_DR.pkl")
                valid_data_path_DR = os.path.join(exp_dir, feedback_identity_string + "_valid_data_DR.pkl")
                simulate_bandit_feedback_command = "python ./scripts_continuous/simulate_bandit_feedback.py " \
                                                   "--simulate_bandit_feedback_data_path {} " \
                                                   "--logging_policy_path {} " \
                                                   "--N {} " \
                                                   "--train_data_path_IPW {} " \
                                                   "--valid_data_path_IPW {}" \
                                                   " --train_data_path_DR {}" \
                                                   " --valid_data_path_DR {}".format(simulate_bandit_feedback_data_path,
                                                                                     logging_policy_path,
                                                                                     data_size,
                                                                                     train_data_path_IPW,
                                                                                     valid_data_path_IPW,
                                                                                     train_data_path_DR,
                                                                                     valid_data_path_DR)
                commands.append(simulate_bandit_feedback_command)
    return commands

def model_fitting_commands(exp_dir, data_names, data_sizes, max_data_size_for_EB, estimators, oracles, lrs, weight_decays, betas, ks, hs, n_runs):
    """generate a list of model fitting commands from the experiment setup"""
    commands = []
    for data_name in data_names:
        test_data_path = os.path.join(exp_dir, data_name + ".test.pkl")
        logging_policy_path = os.path.join(exp_dir, data_name + ".logging_policy.pkl")
        for data_size in data_sizes:
            for estimator in estimators:
                for run in range(n_runs):
                    feedback_identity_string = "_".join([data_name, str(data_size), str(run)])
                    train_data_path = os.path.join(exp_dir, feedback_identity_string + "_train_data_" + estimator + ".pkl")
                    valid_data_path = os.path.join(exp_dir, feedback_identity_string + "_valid_data_" + estimator + ".pkl")
                    for beta in betas:
                        for k in ks:
                            for h in hs:
                                for weight_decay in weight_decays:
                                    model_identity_string = "_".join([data_name, str(data_size), str(estimator), str(run), str(beta), str(k), str(h), str(weight_decay)])
                                    LR_PL_model_path = os.path.join(exp_dir, model_identity_string + "_LR_PL_model,pkl")
                                    LR_PL_result_path = os.path.join(exp_dir, model_identity_string + "_LR_PL_result.pkl")
                                    LR_PL_command = "python ./scripts_continuous/train_LinearRegressionCSCOracle.py " \
                                                           "--train_data_path {} " \
                                                           "--valid_data_path {} " \
                                                           "--test_data_path {} " \
                                                           "--model_path {} " \
                                                           "--result_path {} " \
                                                           "--beta {} " \
                                                           "--weight_decay {} " \
                                                           "--k {} " \
                                                           "--h {} " \
                                                           "--logging_policy_path {} ".format(
                                        train_data_path,
                                        valid_data_path,
                                        test_data_path,
                                        LR_PL_model_path,
                                        LR_PL_result_path,
                                        beta,
                                        weight_decay,
                                        k,
                                        h,
                                        logging_policy_path
                                    ) + " --lrs " + " ".join(lrs)
                                    if "LR" in oracles:
                                        commands.append(LR_PL_command)

                                    PG_PL_model_path = os.path.join(exp_dir, model_identity_string + "_PG_PL_model.pkl")
                                    PG_PL_result_path = os.path.join(exp_dir, model_identity_string + "_PG_PL_result.pkl")
                                    PG_PL_command = "python ./scripts_continuous/train_LinearSoftmaxCSCOracle.py " \
                                                           "--train_data_path {} " \
                                                           "--valid_data_path {} " \
                                                           "--test_data_path {} " \
                                                           "--model_path {} " \
                                                           "--result_path {} " \
                                                           "--beta {} " \
                                                           "--weight_decay {} " \
                                                           "--k {} " \
                                                           "--h {} " \
                                                           "--logging_policy_path {} ".format(
                                        train_data_path,
                                        valid_data_path,
                                        test_data_path,
                                        PG_PL_model_path,
                                        PG_PL_result_path,
                                        beta,
                                        weight_decay,
                                        k,
                                        h,
                                        logging_policy_path
                                    ) + " --lrs " + " ".join(lrs)
                                    if "PG" in oracles:
                                        commands.append(PG_PL_command)

                                    PG_EB_model_path = os.path.join(exp_dir,
                                                                        model_identity_string + "_PG_EB_model.pkl")
                                    PG_EB_result_path = os.path.join(exp_dir,
                                                                         model_identity_string + "_PG_EB_result.pkl")
                                    PG_EB_command = "python ./scripts_continuous/train_LinearSoftmaxEBOracle.py " \
                                                        "--train_data_path {} " \
                                                        "--valid_data_path {} " \
                                                        "--test_data_path {} " \
                                                        "--model_path {} " \
                                                        "--result_path {} " \
                                                        "--beta {} " \
                                                        "--weight_decay {} " \
                                                        "--k {} " \
                                                        "--h {} " \
                                                        "--logging_policy_path {} ".format(
                                        train_data_path,
                                        valid_data_path,
                                        test_data_path,
                                        PG_EB_model_path,
                                        PG_EB_result_path,
                                        beta,
                                        weight_decay,
                                        k,
                                        h,
                                        logging_policy_path
                                    ) + " --lrs " + " ".join(lrs)
                                    if "PG" in oracles and data_size <= max_data_size_for_EB:
                                        commands.append(PG_EB_command)
    return commands

def submit_commands(token, exp_dir, split_size, commands, submit, shuffle):
    """
    submit commands to server
    """
    if shuffle:
        random.shuffle(commands)
    split_len = int((len(commands) - 1) / split_size) + 1
    current_idx = 0
    while True:
        stop = 0
        start = current_idx * split_len
        end = (current_idx + 1) * split_len
        if end >= len(commands):
            stop = 1
            end = len(commands)
        with open(os.path.join(exp_dir, "scripts_{}_{}.sh".format(token, current_idx)), "w") as f:
            for command in commands[start:end]:
                f.write(command + "\n")
        current_idx += 1
        if stop:
            break

    scripts = [os.path.join(exp_dir, "scripts_{}_{}.sh".format(token, idx)) for idx in range(current_idx)]
    cnt = 0
    for script in scripts:
        submission_command = "sbatch --partition=default_partition --requeue -N1 -n1 -c1 --mem=8G " \
                             "-t 24:00:00 -J %s -o %s.o -e %s.e --wrap=\"sh %s\"" % (token + str(cnt), script,
                                                                                     script, script)
        cnt += 1
        if submit:
            os.system(submission_command)
    return

