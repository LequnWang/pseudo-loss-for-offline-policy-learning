from exp_params import *
import os
import pickle
import numpy as np

def select_by_PL(d, alpha=0.1):
    best_param = None
    best_lr = None
    best_objective = None
    test_loss = None
    best_pseudo_loss = None
    best_confidence = None
    for param in d["params"]:
        result = d["params"][param]
        objective = result["IPW"] + np.sqrt(2 * result["pseudo_loss"] * np.log(4 / alpha) / result["valid_data_size"])
        if best_objective is None or objective < best_objective:
            best_param = param
            best_lr = result["best_lr"]
            best_objective = objective
            test_loss = result["test_loss"]
            best_pseudo_loss = result["pseudo_loss"]
            best_confidence = np.sqrt(2 * result["pseudo_loss"] * np.log(4  / alpha) / result["valid_data_size"])

    d["selected_by_PL"] = {}
    d["selected_by_PL"]["param"] = best_param
    d["selected_by_PL"]["lr"] = best_lr
    d["selected_by_PL"]["test_loss"] = test_loss
    d["selected_by_PL"]["pseudo_loss"] = best_pseudo_loss
    d["selected_by_PL"]["confidence"] = best_confidence


def select_by_None(d):
    best_param = None
    best_lr = None
    best_objective = None
    test_loss = None
    for param in d["params"]:
        result = d["params"][param]
        objective = result["IPW"]
        if best_objective is None or objective < best_objective:
            best_param = param
            best_lr = result["best_lr"]
            best_objective = objective
            test_loss = result["test_loss"]

    d["selected_by_None"] = {}
    d["selected_by_None"]["param"] = best_param
    d["selected_by_None"]["lr"] = best_lr
    d["selected_by_None"]["test_loss"] = test_loss



def select_by_EB(d, alpha=0.1):
    best_param = None
    best_lr = None
    best_objective = None
    test_loss = None
    best_sample_variance = None
    best_confidence = None
    for param in d["params"]:
        result = d["params"][param]
        objective = result["IPW"] + np.sqrt(2 * result["sample_variance"] * np.log(2 / alpha) / result["valid_data_size"])
        if best_objective is None or objective < best_objective:
            best_param = param
            best_lr = result["best_lr"]
            best_objective = objective
            test_loss = result["test_loss"]
            best_sample_variance = result["sample_variance"]
            best_confidence = np.sqrt(2 * result["sample_variance"] * np.log(2 / alpha) / result["valid_data_size"])
    d["selected_by_EB"] = {}
    d["selected_by_EB"]["param"] = best_param
    d["selected_by_EB"]["lr"] = best_lr
    d["selected_by_EB"]["test_loss"] = test_loss
    d["selected_by_EB"]["sample_variance"] = best_sample_variance
    d["selected_by_EB"]["confidence"] = best_confidence


def select_by_best(d):
    best_param = None
    best_lr = None
    best_objective = None
    test_loss = None
    for param in d["params"]:
        result = d["params"][param]
        objective = result["test_loss"]
        if best_objective is None or objective < best_objective:
            best_param = param
            best_lr = result["best_lr"]
            best_objective = objective
            test_loss = result["test_loss"]
    d["selected_by_best"] = {}
    d["selected_by_best"]["param"] = best_param
    d["selected_by_best"]["lr"] = best_lr
    d["selected_by_best"]["test_loss"] = test_loss

if __name__ == "__main__":
    results_dict = {}
    confidences = ["PL", "EB", "None"]
    for data_name in data_names:
        results_dict[data_name] = {}
        for oracle in oracles:
            results_dict[data_name][oracle] = {}
            for estimator in estimators:
                results_dict[data_name][oracle][estimator] = {}
                for confidence in confidences:
                    if confidence == "EB" and oracle == "LR":
                        continue
                    results_dict[data_name][oracle][estimator][confidence] = {}
                    if confidence == "None":
                        betas_for_method = [0.]
                    else:
                        betas_for_method = betas
                    for data_size in data_sizes:
                        if confidence == "EB" and data_size > max_data_size_for_EB:
                            continue
                        results_dict[data_name][oracle][estimator][confidence][data_size] = {}
                        for run in range(n_runs):
                            results_dict[data_name][oracle][estimator][confidence][data_size][run] = {}
                            results_dict[data_name][oracle][estimator][confidence][data_size][run]["params"] = {}
                            feedback_identity_string = "_".join([data_name, str(data_size), str(run)])
                            for beta in betas_for_method:
                                for weight_decay in weight_decays:
                                    param_string = "_".join([str(beta), str(weight_decay)])
                                    model_identity_string = "_".join(
                                        [data_name, str(data_size), str(estimator), str(run), str(beta),
                                         str(weight_decay)])
                                    if confidence == "None":
                                        result_path = os.path.join(exp_dir, model_identity_string +
                                                                   "_{}_{}_result.pkl".format(oracle, "PL"))
                                    else:
                                        result_path = os.path.join(exp_dir, model_identity_string +
                                                                   "_{}_{}_result.pkl".format(oracle, confidence))
                                    try:
                                        with open(result_path, 'rb') as f:
                                            result = pickle.load(f)
                                            results_dict[data_name][oracle][estimator][confidence][data_size][run][
                                                "params"][param_string] = result
                                    except:
                                        print(result_path)
                                        continue

                            select_by_EB(results_dict[data_name][oracle][estimator][confidence][data_size][run])

    stats_for_plot = {}
    for data_name in data_names:
        print(data_name)
        stats_for_plot[data_name] = {}
        for data_size in data_sizes:
            stats_for_plot[data_name][data_size] = {}
            print("data size: {}".format(data_size))
            for oracle in oracles:
                for estimator in estimators:
                    for confidence in confidences:
                        if confidence == "EB" and oracle == "LR":
                            continue
                        if confidence == "EB" and data_size > max_data_size_for_EB:
                            print("NA")
                            continue
                        selection_method = "selected_by_EB"
                        method_name = "_".join([oracle, estimator, confidence])
                        performance_across_runs = [results_dict[data_name][oracle][estimator][confidence][data_size][run][selection_method]["test_loss"] for run in range(n_runs)]
                        performance_without_None = []
                        for performance in performance_across_runs:
                            if performance is None:
                                continue
                            performance_without_None.append(performance)
                        stats_for_plot[data_name][data_size][method_name] = performance_without_None
                        print("{:.1f}±{:.1f}".format(np.mean(performance_without_None) * 100, 200 * np.std(performance_without_None, ddof=1) / np.sqrt(len(performance_without_None))))
    with open("./plots/" + exp_token + "_plot_stats.pkl", 'wb') as f:
        pickle.dump(stats_for_plot, f)
