exp_token = "d_real_eps0.1"
exp_dir ="./exp_d_real_eps0.1"
split_size = 5000
submit = True
data_ids = [247, 261, 1183, 1214]
data_sizes = [0.01, 0.1, 1.]
max_data_size_for_EB = 2.
data_id_to_name = {
    247: "BNGLetter",
    261: "BNGPenDigits",
    1183: "BNGSatImage",
    1214: "BNGJPVowel"
}
data_names = [data_id_to_name[data_id] for data_id in data_ids]
data_home = "./data"
train_logging_policy_data_size = 1000
epsilon = 0.1
test_proportion = 0.3
reward_model_proportion = 0.1
estimators = ["IPW", "DR"]
oracles = ["PG", "LR"]
TEST_BAD_POLICY = False
GENERATE_REAL_COST = True
TIMES_NUM_CLASSES = 1
SOFT_MAX_POLICY = False
BINARY_COST = True
betas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.]
lrs = [1e-3, 1e-2, 1e-1, 1., 10.]
lrs = [str(lr) for lr in lrs]
weight_decays = [1e-6]
n_runs = 50

# exp_token = "d_real_eps0.01"
# exp_dir ="./exp_d_real_eps0.01"
# split_size = 5000
# submit = True
# data_ids = [247, 261, 1183, 1214]
# data_sizes = [0.01, 0.1, 1.]
# max_data_size_for_EB = 2.
# data_id_to_name = {
#     247: "BNGLetter",
#     261: "BNGPenDigits",
#     1183: "BNGSatImage",
#     1214: "BNGJPVowel"
# }
# data_names = [data_id_to_name[data_id] for data_id in data_ids]
# data_home = "./data"
# train_logging_policy_data_size = 1000
# epsilon = 0.01
# test_proportion = 0.3
# reward_model_proportion = 0.1
# estimators = ["IPW", "DR"]
# oracles = ["PG", "LR"]
# TEST_BAD_POLICY = False
# GENERATE_REAL_COST = True
# TIMES_NUM_CLASSES = 1
# SOFT_MAX_POLICY = False
# BINARY_COST = True
# betas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.]
# lrs = [1e-3, 1e-2, 1e-1, 1., 10.]
# lrs = [str(lr) for lr in lrs]
# weight_decays = [1e-6]
# n_runs = 50

# exp_token = "d_real_bad"
# exp_dir ="./exp_d_real_bad"
# split_size = 5000
# submit = True
# data_ids = [247, 261, 1183, 1214]
# data_sizes = [0.01, 0.1, 1.]
# max_data_size_for_EB = 2.
# data_id_to_name = {
#     247: "BNGLetter",
#     261: "BNGPenDigits",
#     1183: "BNGSatImage",
#     1214: "BNGJPVowel"
# }
# data_names = [data_id_to_name[data_id] for data_id in data_ids]
# data_home = "./data"
# train_logging_policy_data_size = 1000
# epsilon = 0.1
# test_proportion = 0.3
# reward_model_proportion = 0.1
# estimators = ["IPW", "DR"]
# oracles = ["PG", "LR"]
# TEST_BAD_POLICY = True
# GENERATE_REAL_COST = True
# TIMES_NUM_CLASSES = 1
# SOFT_MAX_POLICY = False
# BINARY_COST = True
# betas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.]
# lrs = [1e-3, 1e-2, 1e-1, 1., 10.]
# lrs = [str(lr) for lr in lrs]
# weight_decays = [1e-6]
# n_runs = 50

# exp_token = "d_real_large_action"
# exp_dir ="./exp_d_real_large_action"
# split_size = 5000
# submit = True
# data_ids = [247, 261, 1183, 1214]
# data_sizes = [0.01, 0.1, 1.]
# max_data_size_for_EB = 2.
# data_id_to_name = {
#     247: "BNGLetter",
#     261: "BNGPenDigits",
#     1183: "BNGSatImage",
#     1214: "BNGJPVowel"
# }
# data_names = [data_id_to_name[data_id] for data_id in data_ids]
# data_home = "./data"
# train_logging_policy_data_size = 1000
# epsilon = 0.1
# test_proportion = 0.3
# reward_model_proportion = 0.1
# estimators = ["IPW", "DR"]
# oracles = ["PG", "LR"]
# TEST_BAD_POLICY = False
# GENERATE_REAL_COST = True
# TIMES_NUM_CLASSES = 5
# SOFT_MAX_POLICY = False
# BINARY_COST = True
# betas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.]
# lrs = [1e-3, 1e-2, 1e-1, 1., 10.]
# lrs = [str(lr) for lr in lrs]
# weight_decays = [1e-6]
# n_runs = 50

# exp_token = "d_binary_eps0.1"
# exp_dir ="./exp_d_binary_eps0.1"
# split_size = 5000
# submit = True
# data_ids = [247, 261, 1183, 1214]
# data_sizes = [0.01, 0.1, 1.]
# max_data_size_for_EB = 2.
# data_id_to_name = {
#     247: "BNGLetter",
#     261: "BNGPenDigits",
#     1183: "BNGSatImage",
#     1214: "BNGJPVowel"
# }
# data_names = [data_id_to_name[data_id] for data_id in data_ids]
# data_home = "./data"
# train_logging_policy_data_size = 1000
# epsilon = 0.1
# test_proportion = 0.3
# reward_model_proportion = 0.1
# estimators = ["IPW", "DR"]
# oracles = ["PG", "LR"]
# TEST_BAD_POLICY = False
# GENERATE_REAL_COST = False
# TIMES_NUM_CLASSES = 1
# SOFT_MAX_POLICY = False
# BINARY_COST = True
# betas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.]
# lrs = [1e-3, 1e-2, 1e-1, 1., 10.]
# lrs = [str(lr) for lr in lrs]
# weight_decays = [1e-6]
# n_runs = 50

# exp_token = "d_binary_eps0.01"
# exp_dir ="./exp_d_binary_eps0.01"
# split_size = 5000
# submit = True
# data_ids = [247, 261, 1183, 1214]
# data_sizes = [0.01, 0.1, 1.]
# max_data_size_for_EB = 2.
# data_id_to_name = {
#     247: "BNGLetter",
#     261: "BNGPenDigits",
#     1183: "BNGSatImage",
#     1214: "BNGJPVowel"
# }
# data_names = [data_id_to_name[data_id] for data_id in data_ids]
# data_home = "./data"
# train_logging_policy_data_size = 1000
# epsilon = 0.01
# test_proportion = 0.3
# reward_model_proportion = 0.1
# estimators = ["IPW", "DR"]
# oracles = ["PG", "LR"]
# TEST_BAD_POLICY = False
# GENERATE_REAL_COST = False
# TIMES_NUM_CLASSES = 1
# SOFT_MAX_POLICY = False
# BINARY_COST = True
# betas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.]
# lrs = [1e-3, 1e-2, 1e-1, 1., 10.]
# lrs = [str(lr) for lr in lrs]
# weight_decays = [1e-6]
# n_runs = 50

# exp_token = "d_binary_bad"
# exp_dir ="./exp_d_binary_bad"
# split_size = 5000
# submit = True
# data_ids = [247, 261, 1183, 1214]
# data_sizes = [0.01, 0.1, 1.]
# max_data_size_for_EB = 2.
# data_id_to_name = {
#     247: "BNGLetter",
#     261: "BNGPenDigits",
#     1183: "BNGSatImage",
#     1214: "BNGJPVowel"
# }
# data_names = [data_id_to_name[data_id] for data_id in data_ids]
# data_home = "./data"
# train_logging_policy_data_size = 1000
# epsilon = 0.1
# test_proportion = 0.3
# reward_model_proportion = 0.1
# estimators = ["IPW", "DR"]
# oracles = ["PG", "LR"]
# TEST_BAD_POLICY = True
# GENERATE_REAL_COST = False
# TIMES_NUM_CLASSES = 1
# SOFT_MAX_POLICY = False
# BINARY_COST = True
# betas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.]
# lrs = [1e-3, 1e-2, 1e-1, 1., 10.]
# lrs = [str(lr) for lr in lrs]
# weight_decays = [1e-6]
# n_runs = 50

# exp_token = "d_binary_large_action"
# exp_dir ="./exp_d_binary_large_action"
# split_size = 5000
# submit = True
# data_ids = [247, 261, 1183, 1214]
# data_sizes = [0.01, 0.1, 1.]
# max_data_size_for_EB = 2.
# data_id_to_name = {
#     247: "BNGLetter",
#     261: "BNGPenDigits",
#     1183: "BNGSatImage",
#     1214: "BNGJPVowel"
# }
# data_names = [data_id_to_name[data_id] for data_id in data_ids]
# data_home = "./data"
# train_logging_policy_data_size = 1000
# epsilon = 0.1
# test_proportion = 0.3
# reward_model_proportion = 0.1
# estimators = ["IPW", "DR"]
# oracles = ["PG", "LR"]
# TEST_BAD_POLICY = False
# GENERATE_REAL_COST = False
# TIMES_NUM_CLASSES = 5
# SOFT_MAX_POLICY = False
# BINARY_COST = True
# betas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.]
# lrs = [1e-3, 1e-2, 1e-1, 1., 10.]
# lrs = [str(lr) for lr in lrs]
# weight_decays = [1e-6]
# n_runs = 50
