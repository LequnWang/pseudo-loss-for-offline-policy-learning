import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_constants import *
plt.rcParams.update(params)
# algorithm_marker = {
#     "PG+IPW+PL": ".",
#     "PG+DR+PL": "x",
#     "LR+IPW+PL": "|",
#     "LR+DR+PL": "1"
# }
n_runs = 50
data_ids = [247, 261, 1183, 1214]
data_sizes = [0.01, 0.1, 1]

data_id_to_name = {
    247: "BNGLetter",
    261: "BNGPenDigits",
    1183: "BNGSatImage",
    1214: "BNGJPVowel"
}


data_names = [data_id_to_name[data_id] for data_id in data_ids]

exp_tokens = ["d_real_eps0.1", "d_real_eps0.01", "d_real_bad", "d_real_large_action", "d_binary_eps0.1", "d_binary_eps0.01", "d_binary_bad", "d_binary_large_action"]

results = {}
oracles = ["PG", "LR"]
estimators = ["IPW", "DR"]
for exp_token in exp_tokens:
    with open("./plots/" + exp_token + "_plot_stats.pkl", 'rb') as f:
        plot_stats = pickle.load(f)
    for data_name in data_names:
        for data_size in data_sizes:
            condition = "+".join([exp_token, data_name, str(data_size)])
            for oracle in oracles:
                for estimator in estimators:
                    for confidence in ["PL", "EB"]:
                        if confidence == "EB" and oracle == "LR":
                            continue
                        method_name = "_".join([oracle, estimator, confidence])
                        None_method_name = "_".join([oracle, estimator, "None"])
                        if method_name not in results:
                            results[method_name] = []
                        confidence_performances = []
                        None_performances = []
                        improvements = []
                        for run in range(n_runs):
                            if run >= len(plot_stats[data_name][data_size][method_name]):
                                break
                            confidence_performance = plot_stats[data_name][data_size][method_name][run]
                            confidence_performances.append(confidence_performance)
                            None_performance = plot_stats[data_name][data_size][None_method_name][run]
                            None_performances.append(None_performance)
                            improvements.append((None_performance - confidence_performance)/None_performance)

                        results[method_name].append((np.mean(confidence_performances), np.mean(None_performances), np.mean(improvements), condition))






def plot_empirical_cdf(data, method_print_name):
    # Sort data and compute the corresponding probabilities
    sorted_data = np.sort(data)[::-1]
    probabilities = np.arange(1, len(data) + 1) / float(len(data))
    # Plot empirical CDF
    plt.plot(sorted_data, probabilities, marker='.', linewidth=line_width, label=method_print_name)

legend = []
for (k, v) in results.items():
    # print(k)
    if k[-2:] == 'EB':
        continue
    lst = [x[2] for x in v]
    # print(lst)
    method_print_name = k.split("_")
    method_print_name = "+".join(method_print_name)
    plot_empirical_cdf(lst, method_print_name)
    legend.append(method_print_name)

xlabel = "Relative Performance of PL vs baseline"
ylabel = "Survival function (1-CDF)"
title = "Relative performance across conditions"
# plt.title(title)
plt.xlabel(xlabel, fontsize=font_size)
plt.ylabel(ylabel, fontsize=font_size)
# Add grid and display the plot
plt.grid(True)
plt.legend(legend)
plt.tight_layout()

plt.savefig('./plots/discrete_empirical_cdf.pdf', dpi=dpi)
