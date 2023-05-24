import numpy as np
exp_token = "c_eps0.1"
exp_dir = "./exp_c_eps0.1"
split_size = 5000
submit = True
data_ids = [1187, 1189, 1190, 40753, 44057]
data_sizes = [0.01, 0.1, 1.]
max_data_size_for_EB = 0.2
data_id_to_name = {
    1187: "BNG_wisconsin",
    1189: "BNG_auto_price",
    1190: "BNG_cpu_act",
    40753: "zurich",
    44057: "black_friday"
}
data_names = [data_id_to_name[id] for id in data_ids]
data_home = "./data"
train_logging_policy_data_size = 1000
logging_h = 0.1
epsilon = 0.1
test_proportion = 0.3
reward_model_proportion = 0.1
estimators = ["IPW", "DR"]
oracles = ["PG", "LR"]
betas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.]
betas = [0., 1e-3, 1e-2, 1e-1]
lrs = [1e-4, 1e-3, 1e-2, 1e-1]
lrs = [1e-4, 1e-3]
lrs = [str(lr) for lr in lrs]
weight_decays = [1e-6]
ks = [11, 21, 51, 101]
ks = [11, 21]
hs = [1e-2, 2e-2, 5e-2, 1e-1]
hs = [5e-2, 1e-1]
n_runs = 5
loss_width = 1.0
def cont_loss(y, a, h=loss_width):
    return min(np.absolute(y - a) / h, 1.)
use_percentile = True



# import numpy as np
# exp_token = "c_eps0.01"
# exp_dir = "./exp_c_eps0.01"
# split_size = 5000
# submit = True
# data_ids = [1187, 1189, 1190, 40753, 44057]
# data_sizes = [0.01, 0.1, 1.]
# max_data_size_for_EB = 0.2
# data_id_to_name = {
#     1187: "BNG_wisconsin",
#     1189: "BNG_auto_price",
#     1190: "BNG_cpu_act",
#     40753: "zurich",
#     44057: "black_friday"
# }
# data_names = [data_id_to_name[id] for id in data_ids]
# data_home = "./data"
# train_logging_policy_data_size = 1000
# logging_h = 0.1
# epsilon = 0.01
# test_proportion = 0.3
# reward_model_proportion = 0.1
# estimators = ["IPW", "DR"]
# oracles = ["PG", "LR"]
# betas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.]
# lrs = [1e-4, 1e-3, 1e-2, 1e-1]
# lrs = [str(lr) for lr in lrs]
# weight_decays = [1e-6]
# ks = [11, 21, 51, 101]
# hs = [1e-2, 2e-2, 5e-2, 1e-1]
# n_runs = 10
# loss_width = 1.0
# def cont_loss(y, a, h=loss_width):
#     return min(np.absolute(y - a) / h, 1.)
# use_percentile = True
