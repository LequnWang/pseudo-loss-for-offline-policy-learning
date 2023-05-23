import os
import random
import numpy as np


def generate_cost(A, L, P, beta):
    data_size = A.shape[0]
    num_classes = P.shape[1]
    C = np.zeros((data_size, num_classes))
    for i in range(data_size):
        for j in range(num_classes):
            if A[i] == j:
                C[i, j] += L[i] / P[i, j]
            C[i, j] += beta / P[i, j]
    return C


def calculate_IPW(A, A_hat, L, loss_shift, P):
    return np.mean(np.sum(A * A_hat, axis=1) * (L - loss_shift) / np.sum((P * A), axis=1) + loss_shift)


def calculate_pseudo_loss(A_hat, P):
    return np.mean(1 / np.sum(A_hat * P, axis=1))


def calculate_sample_variance(A, A_hat, L, loss_shift, P):
    sample = np.sum(A * A_hat, axis=1) * (L - loss_shift) / np.sum((P * A), axis=1) + loss_shift
    sample_mean = np.mean(sample)
    return np.sum((sample - sample_mean) ** 2) / (sample.shape[0] - 1)


def data_generation_commands(exp_dir, data_names, data_sizes, epsilon, n_runs):
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
                simulate_bandit_feedback_command = "python ./scripts_discrete/simulate_bandit_feedback.py " \
                                                   "--simulate_bandit_feedback_data_path {} " \
                                                   "--logging_policy_path {} " \
                                                   "--N {} " \
                                                   "--epsilon {} " \
                                                   "--train_data_path_IPW {} " \
                                                   "--valid_data_path_IPW {} " \
                                                   "--train_data_path_DR {} " \
                                                   "--valid_data_path_DR {} ".format(simulate_bandit_feedback_data_path,
                                                                                 logging_policy_path,
                                                                                 data_size,
                                                                                 epsilon,
                                                                                 train_data_path_IPW,
                                                                                 valid_data_path_IPW,
                                                                                train_data_path_DR,
                                                                                valid_data_path_DR)
                commands.append(simulate_bandit_feedback_command)
    return commands

def model_fitting_commands(exp_dir, data_names, data_sizes, max_data_size_for_EB, estimators, oracles, lrs, weight_decays, betas, n_runs):
    """generate a list of model fitting commands from the experiment setup"""
    commands = []
    for data_name in data_names:
        test_data_path = os.path.join(exp_dir, data_name + ".test.pkl")
        for data_size in data_sizes:
            for estimator in estimators:
                for run in range(n_runs):
                    feedback_identity_string = "_".join([data_name, str(data_size), str(run)])
                    train_data_path = os.path.join(exp_dir, feedback_identity_string + "_train_data_" + estimator + ".pkl")
                    valid_data_path = os.path.join(exp_dir, feedback_identity_string + "_valid_data_" + estimator + ".pkl")
                    for beta in betas:
                        for weight_decay in weight_decays:
                            model_identity_string = "_".join([data_name, str(data_size), str(estimator), str(run), str(beta), str(weight_decay)])

                            LR_PL_model_path = os.path.join(exp_dir, model_identity_string + "_LR_PL_model.pkl")
                            LR_PL_result_path =os.path.join(exp_dir, model_identity_string + "_LR_PL_result.pkl")
                            LR_PL_command = "python ./scripts_discrete/train_LinearRegressionCSCOracle.py " \
                                                   "--train_data_path {} " \
                                                   "--valid_data_path {} " \
                                                   "--test_data_path {} " \
                                                   "--model_path {} " \
                                                   "--result_path {} " \
                                                   "--beta {} " \
                                                   "--weight_decay {} " \
                                                   "--loss_type {}".format(
                                train_data_path,
                                valid_data_path,
                                test_data_path,
                                LR_PL_model_path,
                                LR_PL_result_path,
                                beta,
                                weight_decay,
                                "l2",
                            ) + " --lrs " + " ".join(lrs)
                            if "LR" in oracles:
                                commands.append(LR_PL_command)

                            PG_PL_model_path = os.path.join(exp_dir, model_identity_string + "_PG_PL_model.pkl")
                            PG_PL_result_path =os.path.join(exp_dir, model_identity_string + "_PG_PL_result.pkl")
                            PG_PL_command = "python ./scripts_discrete/train_LinearSoftmaxCSCOracle.py " \
                                                   "--train_data_path {} " \
                                                   "--valid_data_path {} " \
                                                   "--test_data_path {} " \
                                                   "--model_path {} " \
                                                   "--result_path {} " \
                                                   "--beta {} " \
                                                   "--weight_decay {}".format(
                                train_data_path,
                                valid_data_path,
                                test_data_path,
                                PG_PL_model_path,
                                PG_PL_result_path,
                                beta,
                                weight_decay
                            ) + " --lrs " + " ".join(lrs)
                            if "PG" in oracles:
                                commands.append(PG_PL_command)

                            PG_EB_model_path = os.path.join(exp_dir, model_identity_string + "_PG_EB_model.pkl")
                            PG_EB_result_path =os.path.join(exp_dir, model_identity_string + "_PG_EB_result.pkl")
                            PG_EB_command = "python ./scripts_discrete/train_LinearSoftmaxEBOracle.py " \
                                                   "--train_data_path {} " \
                                                   "--valid_data_path {} " \
                                                   "--test_data_path {} " \
                                                   "--model_path {} " \
                                                   "--result_path {} " \
                                                   "--beta {} " \
                                                   "--weight_decay {}".format(
                                train_data_path,
                                valid_data_path,
                                test_data_path,
                                PG_EB_model_path,
                                PG_EB_result_path,
                                beta,
                                weight_decay
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
        submission_command = "sbatch --partition=thorsten,default_partition --exclude=sablab-cpu-01,sablab-cpu-02,sablab-gpu-04,sablab-gpu-06,g2-cpu-11,g2-cpu-01,g2-cpu-07,g2-cpu-08,g2-cpu-27,luxlab-cpu-01 --requeue -N1 -n1 -c1 --mem=12G " \
                             "-t 24:00:00 -J %s -o %s.o -e %s.e --wrap=\"sh %s\"" % (token + str(cnt), script,
                                                                                     script, script)
        cnt += 1
        if submit:
            os.system(submission_command)
    return
