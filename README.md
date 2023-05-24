# Oracle-Efficient Pessimism: Offline Policy Optimization In Contextual Bandits

This repo contains the code for the empirical evaluation in the paper Pessimism: Offline Policy Optimization In Contextual Bandits.  
We implement several offline policy optimization methods with inverse probability weighting and doubly robust estimators, policy-gradient-based and linear-regression-based cost-sensitive classification oracles, pseudo loss and sample variance regularizers. 

### Create environment

Make sure [conda](https://docs.conda.io/en/latest/) is installed. Run
```angular2html
conda env create -f environment.yml
source activate cb-learn
```

### Choose experimental setting for discrete-action experiments in 
```angular2html
./scripts_discrete/exp_params.py
```


### Prepare data for discrete-action experiments

Run
```angular2html
python ./scripts_discrete/prepare_data.py
```

### Simulate bandit feedback data for discrete-action experiments

On a cluster with [Slurm](https://slurm.schedmd.com/documentation.html) workload manager, run
```angular2html
python ./scripts_discrete/run_simulate_bandit_feedback.py
```

### Offline policy learning with different methods for the discrete-action experiments

On a cluster with [Slurm](https://slurm.schedmd.com/documentation.html) workload manager, run
```angular2html
python ./scripts_discrete/run_OPO.py
```

### Model selection for the discrete-action experiments

Run

```angular2html
python ./scripts_discrete/model_selection.py
```

### Generate figure and tables for the discrete-action experiments

Run

```angular2html
python ./scripts_discrete/plot_improvement_figure.py
python ./scripts_discrete/generate_table.py
python ./scripts_discrete/transform_table.py
```


### Choose experimental setting for continuous-action experiments in 
```angular2html
./scripts_continuous/exp_params.py
```


### Prepare data for continuous-action experiments

Run
```angular2html
python ./scripts_continuous/prepare_data.py
```

### Simulate bandit feedback data for continuous-action experiments

On a cluster with [Slurm](https://slurm.schedmd.com/documentation.html) workload manager, run
```angular2html
python ./scripts_continuous/run_simulate_bandit_feedback.py
```

### Offline policy learning with different methods for the continuous-action experiments

On a cluster with [Slurm](https://slurm.schedmd.com/documentation.html) workload manager, run
```angular2html
python ./scripts_continuous/run_OPO.py
```

### Model selection for the continuous-action experiments

Run

```angular2html
python ./scripts_continuous/model_selection.py
```

### Generate figure and tables for the continuous-action experiments

Run

```angular2html
python ./scripts_continuous/plot_improvement_figure.py
python ./scripts_continuous/generate_table.py
python ./scripts_continuous/transform_table.py
```