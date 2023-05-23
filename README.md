# Oracle-Efficient Pessimism: Offline Policy Optimization In Contextual Bandits

This repo contains the code for the empirical evaluation in the paper Pessimism: Offline Policy Optimization In Contextual Bandits.  
We implement several offline policy optimization methods with inverse probability weighting and doubly robust estimators, policy-gradient-based and linear-regression-based cost-sensitive classification oracles, pseudo loss and sample variance regularizers. 

### Create environment

Mane sure [conda](https://docs.conda.io/en/latest/) is installed. Run
```angular2html
conda env create -f environment.yml
source activate cb-learn
```

### Prepare data for discrete-action experiments

Run
```angular2html
python ./scripts_discrete/prepare_data.py
```