import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_constants import *
plt.rcParams.update(params)
from exp_params import *


data_print_name={
    "BNGLetter": "Letter",
    "BNGPenDigits": "PenDigits",
    "BNGSatImage": "SatImage",
    "BNGJPVowel": "JPVowel"
}

with open("./plots/" + exp_token + "_plot_stats.pkl", 'rb') as f:
    plot_stats = pickle.load(f)


for data_name in data_names:
    for data_size in data_sizes:
        improvements = []
        for run in range(n_runs):
            best_PL = None
            best_None = None
            for method_name in plot_stats[data_name][data_size]:
                if run >= len(plot_stats[data_name][data_size][method_name]):
                    break
                if "PL" in method_name:
                    if best_PL is None or best_PL > plot_stats[data_name][data_size][method_name][run]:
                        best_PL = plot_stats[data_name][data_size][method_name][run]
                if "None" in method_name:
                    if best_None is None or best_None > plot_stats[data_name][data_size][method_name][run]:
                        best_None = plot_stats[data_name][data_size][method_name][run]
            if best_None is None:
                continue
            improvements.append((best_None - best_PL) / best_None)
        plot_stats[data_name][data_size]["improvements"] = improvements


# fig = plt.figure(0)
plot_path = "./plots/" + exp_token + "_improvement.pdf"

# x = [str(data_size) for data_size in data_sizes]
# for data_name in data_names:
#     y_mean = []
#     y_std = []
#     for data_size in data_sizes:
#         y_mean.append(np.mean(plot_stats[data_name][data_size]["improvements"]))
#         y_std.append(np.std(plot_stats[data_name][data_size]["improvements"], ddof=1))
#     plt.plot(x, y_mean, label=data_name)
#     plt.legend()
#     plt.xlabel("Data Size")
#     plt.ylabel("Improvement")
#     plt.tight_layout()
#     plt.savefig(plot_path, format="jpeg", dpi=dpi)


width = 0.2
fig, ax = plt.subplots()
X = np.arange(3)
shift = -0.4
for data_name in data_names:
    y_mean = []
    y_std_err = []
    for data_size in data_sizes:
        y_mean.append(100 * np.mean(plot_stats[data_name][data_size]["improvements"]))
        y_std_err.append(200 * np.std(plot_stats[data_name][data_size]["improvements"], ddof=1) / np.sqrt(len(plot_stats[data_name][data_size]["improvements"])))
    ax.bar(X + shift, y_mean, yerr=y_std_err, width=width, label=data_print_name[data_name])
    shift += width
ax.legend()
ax.set_xlabel("Data Size", fontsize=font_size)
ax.set_ylabel("Improvement %", fontsize=font_size)
ax.set_xticks(X, ("0.01", "0.1", "1.0"))

plt.tight_layout()
plt.savefig(plot_path, format="pdf", dpi=dpi)
