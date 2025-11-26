import os
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from vae_model import VAE, reconstruction_loss_mse, kl_divergence


def train_vae(X_train, X_val, input_dim=10, z_dim=8, hidden_dims=(64,32),
              beta=1.0, anneal=False, anneal_epochs=30, epochs=50, batch_size=128,
              lr=1e-3, device='cpu', out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)

    model = VAE(input_dim=input_dim, hidden_dims=hidden_dims, z_dim=z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train).float()
    train_ds = TensorDataset(X_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {'epoch':[], 'recon_loss':[], 'kl_loss':[], 'total_loss':[], 'beta':[]}

    for epoch in range(1, epochs+1):
        model.train()
        running_recon = 0.0
        running_kl = 0.0
        running_total = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            # per-sample kl and recon
            kl_per = kl_divergence(mu, logvar)  # shape (batch,)
            recon_per = torch.mean((recon - x) ** 2, dim=1)

            # reductions
            kl_term = kl_per.mean()
            recon_term = recon_per.mean()

            # compute beta (anneal if needed)
            if anneal:
                current_beta = float(min(1.0, epoch / max(1, anneal_epochs))) * beta
            else:
                current_beta = beta

            loss = recon_term + current_beta * kl_term
            loss.backward()
            optimizer.step()

            running_recon += recon_term.item() * x.size(0)
            running_kl += kl_term.item() * x.size(0)
            running_total += loss.item() * x.size(0)

        # epoch averages
        n_train = len(X_train)
        epoch_recon = running_recon / n_train
        epoch_kl = running_kl / n_train
        epoch_total = running_total / n_train
        history['epoch'].append(epoch)
        history['recon_loss'].append(epoch_recon)
        history['kl_loss'].append(epoch_kl)
        history['total_loss'].append(epoch_total)
        history['beta'].append(current_beta)

        # simple validation if X_val provided
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val).float().to(device)
                recon_v, mu_v, logvar_v = model(xv)
                recon_per_v = torch.mean((recon_v - xv) ** 2, dim=1)
                kl_per_v = kl_divergence(mu_v, logvar_v)
                val_recon = recon_per_v.mean().item()
                val_kl = kl_per_v.mean().item()
        else:
            val_recon = None
            val_kl = None

        print(f"Epoch {epoch}/{epochs} | recon={epoch_recon:.6f} kl={epoch_kl:.6f} beta={current_beta:.4f}")

        # checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            ckpt_path = os.path.join(out_dir, f"vae_beta{beta}_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

    # save final model
    final_path = os.path.join(out_dir, f"vae_beta{beta}_final.pt")
    torch.save(model.state_dict(), final_path)

    # save history
    hist_path = os.path.join(out_dir, f"history_beta{beta}.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--z_dim', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--anneal', action='store_true')
    parser.add_argument('--anneal_epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # load data
    X_train = pd.read_csv(os.path.join(args.data_dir, 'X_train.csv')).values
    X_test = pd.read_csv(os.path.join(args.data_dir, 'X_test.csv')).values
    y_test = pd.read_csv(os.path.join(args.data_dir, 'y_test.csv'))['label'].values

    model, history = train_vae(X_train, X_test, input_dim=X_train.shape[1], z_dim=args.z_dim,
                              beta=args.beta, anneal=args.anneal, anneal_epochs=args.anneal_epochs,
                              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                              out_dir=args.results_dir)

    print('Training finished. Model and history saved to', args.results_dir)
