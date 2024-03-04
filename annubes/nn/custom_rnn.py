from math import sqrt

import torch
from numpy.typing import NDArray
from torch import nn


class CustomRNN(nn.Module):
    """Custom Recurrent Neural Network."""

    def __init__(  # TODO: fix type annotations, these are guesses
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        param: dict,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ReLU = nn.ReLU()
        self.n_ex = int(hidden_size * param["ex_in_ratio"])
        self.n_in = hidden_size - self.n_ex
        self.rec_noise_std = param["rec_noise_std"]

        # I2H
        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        # turn off updates for input to hidden weights
        self.i2h.weight.requires_grad = False

        # H2H
        # NOTE: I think we may not be setting the weights for inhibitory neurons correctly
        # Unlike Song, do not put self recurrence on 0
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        with torch.no_grad():  # just for the multiplication
            self.h2h.weight[:, : self.n_in] = (
                self.h2h.weight[:, : self.n_in] * 4
            )  # Why -> avoid breaking of convergence

        # H2O
        self.h2o = nn.Linear(self.hidden_size - self.n_in, self.output_size)
        # turn off updates for hidden to output weights
        self.h2o.weight.requires_grad = False
        self.h2o.weight.data.uniform_(-0.1, 0.1)
        self.h2o.bias.data.uniform_(-0.1, 0.1)

        # self.noise = torch.distributions.normal.Normal(0, self.rec_noise_std)  # noqa: ERA001
        self.noise = torch.distributions.normal.Normal(0, 1)
        self.set_weights()

    def set_weights(self) -> None:  # noqa: D102
        with torch.no_grad():
            # constrain inhibitory weights to have negative values
            torch.clamp(self.h2h.weight[:, : self.n_in], max=0, out=self.h2h.weight[:, : self.n_in])
            # constrain excitatory weights to have positive values
            torch.clamp(self.h2h.weight[:, self.n_in :], min=0, out=self.h2h.weight[:, self.n_in :])

    def forward(  # noqa: D102, TODO: fix type annotations, these are guesses
        self,
        x: NDArray,
        tau: float,
        dt: float,
        rnn_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_sz, trial_length, _ = x.shape
        alpha = dt / tau
        rnn_input = torch.zeros(batch_sz, 1, self.hidden_size)

        if rnn_output is None:
            rnn_output = torch.zeros(batch_sz, 1, self.hidden_size)
        network_output = torch.zeros(batch_sz, 1, self.output_size)

        for t in range(trial_length):
            # eq 7
            rnn_input_t = alpha * (self.i2h(x[:, t, :]) + self.h2h(rnn_output[:, t, :]))
            +((1 - alpha) * rnn_input[:, t, :])

            # Noise
            internal_noise = self.noise.sample(rnn_input_t.size())
            rnn_input_t = rnn_input_t + (internal_noise * sqrt(2 * alpha) * self.rec_noise_std)

            # eq 8
            rnn_output_t = self.ReLU(rnn_input_t)
            rnn_output_t_ex = rnn_output_t[:, self.n_in :]  # only read out from excitatory neurons
            # eq 9
            network_output_t = self.h2o(rnn_output_t_ex)

            rnn_input = torch.cat(
                [rnn_input, rnn_input_t[:, None, :]],
                dim=1,
            )  # Appends along time axis, such that next iteration has access at [:,t,:] to basically [:,t-1,:]
            rnn_output = torch.cat([rnn_output, rnn_output_t[:, None, :]], dim=1)
            network_output = torch.cat([network_output, network_output_t[:, None, :]], dim=1)

        return network_output, rnn_output
