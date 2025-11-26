import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def make_dataset(n_samples=10000, n_features=10, anomaly_frac=0.02, random_seed=0):
    np.random.seed(random_seed)
    # Normal data: mixture of 3 Gaussians with different means
    n_norm = int(n_samples * (1 - anomaly_frac))
    n_each = n_norm // 3
    cov_base = np.eye(n_features) * 0.5

    means = [np.random.uniform(-2, 2, size=n_features) for _ in range(3)]
    X_norm_parts = [np.random.multivariate_normal(m, cov_base, size=n_each) for m in means]
    X_norm = np.vstack(X_norm_parts)
    # If rounding left some samples out, append from first component
    if X_norm.shape[0] < n_norm:
        extra = n_norm - X_norm.shape[0]
        X_norm = np.vstack([X_norm, np.random.multivariate_normal(means[0], cov_base, size=extra)])

    # Anomalies: larger variance and shifted mean
    n_anom = n_samples - n_norm
    anom_mean = np.random.uniform(5, 8, size=n_features)
    X_anom = np.random.multivariate_normal(anom_mean, np.eye(n_features) * 4.0, size=n_anom)

    X = np.vstack([X_norm, X_anom])
    y = np.hstack([np.zeros(len(X_norm), dtype=int), np.ones(len(X_anom), dtype=int)])

    # shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # train/test split (stratify to keep anomaly fraction similar)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic dataset for VAE anomaly detection")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--anomaly_frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = make_dataset(n_samples=args.n_samples,
                                                   n_features=args.n_features,
                                                   anomaly_frac=args.anomaly_frac,
                                                   random_seed=args.seed)

    pd.DataFrame(X_train).to_csv(os.path.join(args.out_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(args.out_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train, columns=["label"]).to_csv(os.path.join(args.out_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test, columns=["label"]).to_csv(os.path.join(args.out_dir, "y_test.csv"), index=False)

    print(f"Saved data to {args.out_dir}: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
