import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch

from vae_model import VAE


def compute_reconstruction_scores(model, X, device='cpu'):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X).float().to(device)
        recon, mu, logvar = model(x_t)
        per_sample_mse = torch.mean((recon - x_t) ** 2, dim=1).cpu().numpy()
    return per_sample_mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--out_csv', type=str, default=os.path.join('results', 'anomaly_scores_100_20.csv'))
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    X_test = pd.read_csv(os.path.join(args.data_dir, 'X_test.csv')).values
    y_test = pd.read_csv(os.path.join(args.data_dir, 'y_test.csv'))['label'].values

    input_dim = X_test.shape[1]

    if args.model_path is None:
        # pick the latest final model in results_dir
        candidates = [f for f in os.listdir(args.results_dir) if f.endswith('.pt')]
        if not candidates:
            raise FileNotFoundError('No .pt model file found in results_dir; pass --model_path')
        candidates = sorted(candidates)
        args.model_path = os.path.join(args.results_dir, candidates[-1])

    model = VAE(input_dim=input_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    scores = compute_reconstruction_scores(model, X_test, device=args.device)

    # compute threshold: use 95th percentile of reconstruction on test normal samples (simple heuristic)
    normal_mask = (y_test == 0)
    if normal_mask.sum() < 100:
        raise ValueError('Not enough normal samples in test set to pick threshold')

    threshold = np.percentile(scores[normal_mask], 95)
    preds = (scores >= threshold).astype(int)

    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, scores)

    print(f"Threshold (95th percentile over normal test samples): {threshold:.6f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

    # prepare 100 normal and 20 anomalous samples
    normal_idx = np.where(y_test == 0)[0]
    anom_idx = np.where(y_test == 1)[0]

    if len(normal_idx) < 100:
        raise ValueError('Less than 100 normal samples in test set')
    if len(anom_idx) < 20:
        raise ValueError('Less than 20 anomalous samples in test set')

    sel_norm = normal_idx[:100]
    sel_anom = anom_idx[:20]

    norm_scores = scores[sel_norm]
    anom_scores = scores[sel_anom]

    combined = np.concatenate([norm_scores, anom_scores])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    # Save as single-row CSV with 120 comma-separated values
    pd.DataFrame([combined]).to_csv(args.out_csv, header=False, index=False)

    print('Saved anomaly scores (100 normal + 20 anomalous) to', args.out_csv)
