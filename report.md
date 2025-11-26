# Analysis Report: VAE for High-Dimensional Anomaly Detection

## 1. Dataset Generation
A synthetic dataset with 10,000 samples and 10 features was generated. Normal data was drawn from a multivariate Gaussian distribution, while anomalies were created by sampling from a shifted Gaussian distribution.

## 2. VAE Architecture
- Input dimension: 10  
- Latent dimension: 2–8 (configurable)  
- Encoder: Linear → ReLU → Linear  
- Decoder: Linear → ReLU → Linear  
- Reparameterization trick implemented  

## 3. Loss Function
The VAE loss = reconstruction loss + β · KL divergence.

- Reconstruction: MSE  
- KL divergence: standard closed-form  
- β values tested: 0.1, 1.0, 5.0  

## 4. Experiments with KL Weighting

### β = 0.1
- Weak regularization  
- Good reconstruction but weak anomaly separation  

### β = 1.0
- Balanced latent space  
- Best anomaly detection performance  

### β = 5.0
- Very strong regularization  
- Latent collapse  
- Poor anomaly separation  

## 5. Evaluation Metrics
Metrics computed using reconstruction error-based anomaly scores:

- Precision  
- Recall  
- F1-Score  

## 6. Conclusion
A moderate KL weight (β ≈ 1) produced the best trade-off between reconstruction quality and anomaly detection accuracy.
