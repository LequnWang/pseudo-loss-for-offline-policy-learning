import pickle
import numpy as np
from exp_params import *


for data_size in data_sizes:
    table_path = "./plots/" + exp_token + "_size_" + str(data_size) + ".txt"
    transformed_table_path = "./plots/" + exp_token + "_size_" + str(data_size) + "_transformed.txt"
    with open(table_path, 'r') as f:
        table = []
        first_line = True
        for line in f.readlines():
            if first_line:
                title_line = line
                first_line = False
            else:
                items = line.split("&")
                items = [items[0].strip()] + [item[1:item.find("\t")] for item in items[1:]]
                table.append(items)
        m = len(table)
        n = len(table[0])
        bold_indexes = []
        box_indexes = []
        for i in range(1, n):
            best_performance = None
            best_performance_indexes = None
            best_performance_oracle_estimator = {}
            best_performance_indexes_oracle_estimator = {}
            for oracle in oracles:
                for estimator in estimators:
                    best_performance_oracle_estimator[oracle + "+" + estimator] = None
                    best_performance_indexes_oracle_estimator[oracle + "+" + estimator] = None
            for j in range(m):
                oracle, estimator = table[j][0].split("+")[:2]
                oracle_estimator = oracle + "+" + estimator
                mean_performance = float(table[j][i][:table[j][i].find("±")])
                if best_performance is None or mean_performance < best_performance:
                    best_performance = mean_performance
                    best_performance_indexes = {j}
                elif best_performance == mean_performance:
                    best_performance_indexes.add(j)
                if best_performance_oracle_estimator[oracle_estimator] is None or mean_performance < best_performance_oracle_estimator[oracle_estimator]:
                    best_performance_oracle_estimator[oracle_estimator] = mean_performance
                    best_performance_indexes_oracle_estimator[oracle_estimator] = {j}
                elif best_performance_oracle_estimator[oracle_estimator] == mean_performance:
                    best_performance_indexes_oracle_estimator[oracle_estimator].add(j)
            box_indexes.append(best_performance_indexes)
            bold_indexes.append(set())
            for oracle in oracles:
                for estimator in estimators:
                    bold_indexes[-1] = bold_indexes[-1].union(best_performance_indexes_oracle_estimator[oracle + "+" + estimator])

    with open(transformed_table_path, 'w') as f:
        f.write(title_line)
        last_oracle_estimator = None
        for i in range(m):
            oracle, estimator = table[i][0].split("+")[:2]
            oracle_estimator = oracle + "+" + estimator
            if last_oracle_estimator != oracle_estimator:
                f.write("\\midrule\n")
                last_oracle_estimator = oracle_estimator
            f.write(table[i][0])
            f.write("\t& ")
            for j in range(1, n-1):
                if i in bold_indexes[j-1] and i in box_indexes[j-1]:
                    f.write("\\fbox{\\textbf{")
                    f.write(table[i][j])
                    f.write("}}")
                    f.write("\t& ")
                elif i in bold_indexes[j-1]:
                    f.write("\\textbf{")
                    f.write(table[i][j])
                    f.write("}")
                    f.write("\t& ")
                else:
                    f.write(table[i][j])
                    f.write("\t& ")
            if i in bold_indexes[n-2] and i in box_indexes[n-2]:
                f.write("\\fbox{\\textbf{")
                f.write(table[i][n-1])
                f.write("}}")
            elif i in bold_indexes[n-2]:
                f.write("\\textbf{")
                f.write(table[i][n-1])
                f.write("}")
            else:
                f.write(table[i][n-1])

            f.write("\t\\\\\n")

