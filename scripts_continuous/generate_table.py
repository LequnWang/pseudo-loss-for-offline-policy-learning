import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exp_params import *

data_print_name = {
"BNG_wisconsin": "Wisconsin",
"BNG_auto_price": "AutoPrice",
"BNG_cpu_act": "CPUAct",
"zurich": "Zurich",
"black_friday": "BlackFriday"
}

with open("./plots/" + exp_token + "_plot_stats.pkl", 'rb') as f:
    plot_stats = pickle.load(f)


confidences = ["PL", "EB", "None"]
for data_size in data_sizes:
    table_path = "./plots/" + exp_token + "_size_" + str(data_size) + ".txt"
    with open(table_path, 'w') as f:
        # first line
        f.write("Risk * 100" + "\t")
        for data_name in data_names:
            f.write("& " + data_print_name[data_name] + "\t")
        # f.write("\\\\ \\midrule\n")
        f.write("\\\\\n")
        for oracle in oracles:
            for estimator in estimators:
                for confidence in confidences:
                    if confidence == "EB" and oracle == "LR":
                        continue
                    if confidence == "EB" and data_size > max_data_size_for_EB:
                        continue
                    method_name = "_".join([oracle, estimator, confidence])
                    method_print_name = ""
                    if oracle == "PG":
                        method_print_name += "PG"
                    elif oracle == "LR":
                        method_print_name += "LR"
                    if estimator == "IPW":
                        method_print_name += "+IPW"
                    else:
                        method_print_name += "+DR"
                    if confidence == "PL":
                        method_print_name += "+PL"
                    elif confidence == "EB":
                        method_print_name += "+EB"
                    else:
                        method_print_name += ""
                    f.write(method_print_name + "\t")
                    for data_name in data_names:
                        performance = plot_stats[data_name][data_size][method_name]
                        f.write("& " + "{:.1f}±{:.1f}\t".format(np.mean(performance) * 100,
                                                     200 * np.std(performance, ddof=1) / np.sqrt(
                                                         len(performance))))
                    f.write("\\\\\n")
