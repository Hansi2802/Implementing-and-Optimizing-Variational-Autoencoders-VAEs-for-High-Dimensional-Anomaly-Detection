# Report — Implementing and Optimizing VAE for High-Dimensional Anomaly Detection

## Summary
This project implements a Variational Autoencoder (VAE) to perform unsupervised anomaly detection on a synthetic 10-dimensional dataset. The dataset contains 10,000 samples with ~2% anomalies. We trained the VAE while varying the KL-divergence weight (beta) and optionally using a simple linear annealing schedule.

## Dataset
- Samples: 10,000
- Features: 10 numeric features
- Normal data: Mixture of 3 Gaussian components (covariance = 0.5 I)
- Anomalies: 2% samples drawn from a separate Gaussian with shifted mean and larger variance
- Train/test split: 80/20 stratified

## Model architecture
- Encoder: [10 -> 64 -> 32] with ReLU
- Latent: z_dim = 8 (configurable)
- Decoder: [8 -> 32 -> 64 -> 10]
- Loss: MSE reconstruction + beta * KL

## Training details
- Optimizer: Adam, lr=1e-3
- Batch size: 128
- Epochs: 50 (default)
- Beta experiments suggested: 0.0, 0.5, 1.0, 4.0 and linear annealing for first 30 epochs

## Evaluation
- Anomaly score: per-sample reconstruction MSE
- Threshold heuristic: 95th percentile of reconstruction MSE on normal test samples
- Metrics: Precision, Recall, F1, ROC-AUC

## Observations (to fill after running experiments)
- Low beta (near 0) typically allows very low reconstruction error but may overfit and make anomalies harder to separate.
- High beta (>>1) enforces stronger latent regularization, which can blur reconstructions and increase reconstruction error for both normal and anomalous samples; however, it can help separate anomalies if latent structure becomes more compact.
- Annealing beta from 0 -> 1 over early epochs often yields a good trade-off: early-stage good reconstruction, later-stage regularized latent space.

## Suggested next steps
- Try Gaussian likelihood (negative log-likelihood) instead of MSE for reconstruction.
- Use more advanced anomaly scoring like combining reconstruction probability and latent Mahalanobis distance.
- Visualize latent space with PCA or t-SNE to inspect separation of anomalies.
