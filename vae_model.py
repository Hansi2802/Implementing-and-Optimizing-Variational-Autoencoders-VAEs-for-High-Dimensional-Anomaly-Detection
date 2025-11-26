import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_dim=10, hidden_dims=(64, 32), z_dim=8):
        super().__init__()
        self.input_dim = input_dim
        # Encoder
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        last_h = hidden_dims[-1]
        self.fc_mu = nn.Linear(last_h, z_dim)
        self.fc_logvar = nn.Linear(last_h, z_dim)

        # Decoder (mirror)
        dec_layers = []
        in_dim = z_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))  # output reconstruction
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# loss utilities
def kl_divergence(mu, logvar):
    # summed over latent dims per batch element, returns mean over batch
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl

def reconstruction_loss_mse(recon, x, reduction='mean'):
    # per-sample MSE (mean over features), returns per-sample values
    per_sample_mse = torch.mean((recon - x) ** 2, dim=1)
    if reduction == 'mean':
        return per_sample_mse.mean()
    elif reduction == 'sum':
        return per_sample_mse.sum()
    else:
        return per_sample_mse
