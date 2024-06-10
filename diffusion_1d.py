import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

import matplotlib.pyplot as plt


class ErrorModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.f = nn.Sequential(
            nn.Linear(2, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x, t):
        x_ = torch.cat([x, t], dim=1)
        return self.f(x_)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def get_alpha_beta_bar(alpha, t):
    assert np.any(0 < t)

    alpha_bar = alpha ** t
    beta_bar = 1 - alpha_bar

    return alpha_bar, beta_bar


if __name__ == '__main__':
    # training hyper param
    n_iter = 30_000
    batch_size = 200 * 3
    t_max = 5
    alpha = 0.9
    beta = 1 - alpha
    sigma = np.sqrt(beta)

    # data
    N = 200
    d1 = 0.1 * np.random.randn(N).reshape((N, 1))
    d2 = 0.3 * np.random.randn(N).reshape((N, 1)) + 2
    d3 = 0.2 * np.random.randn(N).reshape((N, 1)) - 2
    data = np.concatenate([d1, d2, d3], axis=0)
    data = np.random.permutation(data)

    # model & optimizer
    noise_model = ErrorModel().to(torch.float32)
    noise_model.apply(init_weights)

    noise_model.train()
    optimizer = torch.optim.AdamW(params=noise_model.parameters(), lr=0.001, weight_decay=0.000001)
    mse = nn.MSELoss()

    # training (weight params are fixed to 1)
    loss_hist = []
    for n in range(n_iter):
        noise_model.train()
        loss = 0
        for batch in range(0, int(len(data) / batch_size), batch_size):
            t = np.random.randint(1, t_max + 1, (batch_size, 1))
            noise = np.random.randn(batch_size, 1)

            x0 = data[batch:batch+batch_size]
            alpha_bar, beta_bar = get_alpha_beta_bar(alpha, t)
            x = torch.tensor(np.sqrt(alpha_bar) * x0 + np.sqrt(beta_bar) * noise, dtype=torch.float32)
            t_ = torch.tensor(t, dtype=torch.float32)
            noise_pred = noise_model(x, t_)

            loss += mse(torch.tensor(noise, dtype=torch.float32), noise_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_hist.append(loss.item())
        print(f"LOSSS @ {n}: {loss.item()}")

        if n % 100 == 0:
            # sampling test
            noise_model.eval()
            data_sampled = np.zeros(N)
            for n_sample in range(N):
                x_sample = np.random.randn()
                for t in reversed(range(t_max)):
                    t_sample = t + 1
                    u = np.random.randn()
                    if t == 1:
                        u = 0.0

                    alpha_bar, beta_bar = get_alpha_beta_bar(alpha, t_sample)
                    x = torch.tensor([[x_sample]], dtype=torch.float32)
                    t_ = torch.tensor([[t_sample]], dtype=torch.float32)
                    noise_pred_ = noise_model(x, t_).item()

                    x_sample = (1 / np.sqrt(alpha)) * (x_sample - (beta / np.sqrt(beta_bar)) * noise_pred_) + sigma * u

                data_sampled[n_sample] = x_sample

            # plotting
            plt.clf()
            plt.subplot(311)
            plt.title("Target")
            plt.hist(data, bins=100, range=(-3, 3), density=True)

            plt.subplot(312)
            plt.title("Sampled")
            plt.hist(data_sampled, bins=100, range=(-3, 3), density=True)

            plt.subplot(313)
            plt.ylabel("Loss")
            plt.plot(loss_hist)
            plt.yscale("log")

            plt.tight_layout()
            if n < n_iter - 1:
                plt.pause(0.0001)

    plt.show()
