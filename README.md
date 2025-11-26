# VAE Anomaly Detection Project

This project implements a Variational Autoencoder (VAE) from scratch for detecting anomalies in high-dimensional data.

## Contents
- data_generation.py  
- vae_model.py  
- train.py  
- evaluate.py  
- report.md  
- submission.md  
- requirements.txt  

## Quick start
1. Create venv and install requirements:
```

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

```
2. Generate data:
```

python data_generation.py --n_samples 10000 --n_features 10 --anomaly_frac 0.02

```
3. Train with a given beta:
```

python train.py --beta 1.0 --epochs 50

```
4. Evaluate and produce `results/anomaly_scores_100_20.csv`:
```

python evaluate.py --results_dir results

```
