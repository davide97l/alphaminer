import torch
import torch.nn.functional as F
import torch.optim as optim
from ding.torch_utils.network import MLP
from alphaminer.rl.env import DataSource
from alphaminer.utils.torch_utils import EarlyStopper
import numpy as np
import qlib
from qlib.contrib.data.handler import Alpha158
from easydict import EasyDict
import argparse
import copy
import os
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm


class Encoder(torch.nn.Module):
    def __init__(self, d_in, h, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, h)
        self.linear2 = torch.nn.Linear(h, h)
        self.enc_mu = torch.nn.Linear(h, latent_size)
        self.enc_log_sigma = torch.nn.Linear(h, latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        return torch.distributions.Normal(loc=mu, scale=sigma)


class Decoder(torch.nn.Module):
    def __init__(self, d_in, h, d_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, h)
        self.linear2 = torch.nn.Linear(h, d_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = torch.tanh(self.linear2(x))
        return torch.distributions.Normal(mu, torch.ones_like(mu))


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, state):
        q_z = self.encoder(state)
        z = q_z.rsample()
        return self.decoder(z), q_z, z


def train(model, ds, args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    model.train()
    # init early stopping
    early_stopper = None
    if args.patience is not None and args.patience >= 0:
        early_stopper = EarlyStopper(patience=args.patience)
    # generate dataset
    dataset = []
    for data in tqdm(ds._obs_data.values()):
        if args.flatten_obs:
            if data.values.shape[0] != args.len_index:  # filter day with missing obs
                continue
            dataset.append(data.values.reshape(1, -1))
        else:
            dataset.append(data.values)
    dataset = np.concatenate(dataset, axis=0)  # shape (n_days * n_instruments, n_features)
    print('N samples:', dataset.shape[0])
    n_samples = dataset.shape[0]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_loss, best_weights = np.inf, model.state_dict()
    loss_values = []
    for epoch in tqdm(range(1, args.epochs + 1)):
        e_loss, e_lp, e_kl = 0, 0, 0
        for data in dataloader:
            x = data
            x = torch.nan_to_num(x.to(device))
            optimizer.zero_grad()
            p_x, q_z, _ = model(x)
            log_likelihood = p_x.log_prob(x).sum(-1).mean()
            kl = torch.distributions.kl_divergence(
                q_z,
                torch.distributions.Normal(0, 1.)
            ).sum(-1).mean()
            loss = -(log_likelihood - kl)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            e_loss += loss
            e_kl += kl.item()
            e_lp += log_likelihood.item()
        e_loss /= n_samples
        loss_values.append(e_loss)
        print("Epoch: {} | Loss: {} | Log prob: {} | KL div: {}".
              format(epoch, e_loss, e_lp / n_samples, e_kl / n_samples))
        if e_loss < best_loss:
            best_loss = e_loss
            best_weights = copy.deepcopy(model.state_dict())
            if args.save_path:
                os.makedirs(args.save_path, exist_ok=True)
                model_path = os.path.join(args.save_path, 'vae_{}.pth'.
                                          format(args.identifier))
                torch.save(model.state_dict(), model_path)
            print('New best loss: {}. Model saved at: {}'.format(best_loss, model_path))
        if early_stopper and early_stopper.early_stop(e_loss):
            print('Training stopped due to early stopping: loss did not improve for {} epochs'.
                  format(args.patience))
            break

    model.load_state_dict(best_weights)
    print('Training finished. Restored best state with loss {}'.format(best_loss))
    return loss_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train vae')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--n_features', type=int, default=158)
    parser.add_argument('--len_index', type=int, default=500)
    parser.add_argument('--latent_space', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--start_date', type=str, default='2008-01-01')
    parser.add_argument('--end_date', type=str, default='2022-06-30')
    parser.add_argument('--patience', type=int, default=10)  # used for early stopping
    parser.add_argument('--save_path', type=str, default='./weights')  # None if don't want to save model
    parser.add_argument('--flatten_obs', action='store_true')
    args = parser.parse_args()
    args.identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def get_data_path() -> str:
        dirname = os.path.dirname(os.path.realpath(__file__))
        return os.path.realpath(os.path.join(dirname, "../tests/data"))

    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region="cn")
    #qlib.init(provider_uri=get_data_path(), region="cn")

    data_handler = dict(
        instruments='csi500',
        start_time=args.start_date,
        end_time=args.end_date,
        fit_start_time=args.start_date,
        fit_end_time=args.end_date,
    )
    config = EasyDict(data_handler)
    dh = Alpha158(**config)
    ds = DataSource(start_date=args.start_date,
                    end_date=args.end_date,
                    market="csi500",
                    data_handler=dh)
    if args.flatten_obs:
        assert args.latent_space % args.len_index == 0
        e = Encoder(args.n_features * args.len_index, 8196, args.latent_space)
        d = Decoder(args.latent_space, 8196, args.n_features * args.len_index)
    else:
        e = Encoder(args.n_features, 64, args.latent_space)
        d = Decoder(args.latent_space, 64, args.n_features)
    model = VAE(e, d)
    loss_values = train(model, ds, args)
    plt.plot(loss_values)
    plt.savefig('vae_loss_{}.png'.format(args.identifier))
    plt.show()
