"""The whole FlyWire connectome, run continuously with fly-brain's PyTorch LIF model."""

from . import config  # noqa: F401  (puts fly-brain/code on sys.path)

import pyarrow  # noqa: F401  (fly-brain: import before torch to avoid a libarrow conflict)
import torch

from benchmark import path_comp, path_con, path_wt
from run_pytorch import DT, MODEL_PARAMS, TorchModel, get_weights


class ConnectomeModel(TorchModel):
    """fly-brain's TorchModel, with the recurrent input computed as W @ s.

    Same dynamics; this avoids transposing the CSR weight matrix on every 0.1 ms step.
    """

    def forward(self, rates, conductance, delay_buffer, spikes, v, refrac):
        voltage_stim = self.scale * self.poisson(rates)
        recurrent_input = self.scale * torch.sparse.mm(self.weights, spikes.T.contiguous()).T
        return self.neurons(recurrent_input, voltage_stim, conductance, delay_buffer, spikes, v, refrac)


class LiveBrain:
    """Keeps the brain's state between calls so it runs as one continuous life."""

    def __init__(self, groups, stimulated, batch=1, device=None):
        """
        groups: name -> list of neuron indices (inputs and readouts)
        stimulated: indices that receive Poisson input (refractory period set to 0, as in Shiu et al.)
        batch: independent brains run in parallel (used by calibration)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        weights = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(self.device)
        self.num_neurons = weights.shape[0]
        self.batch = batch
        self.model = ConnectomeModel(batch, self.num_neurons, DT, MODEL_PARAMS, weights,
                                     exc_indices=list(stimulated), device=self.device)
        self.state = self.model.state_init()
        self.rates = torch.zeros(batch, self.num_neurons, device=self.device)
        self.groups = {name: torch.tensor(idx, dtype=torch.long, device=self.device)
                       for name, idx in groups.items()}

    def set_input(self, rates_hz, trial=None):
        """Set Poisson input rates (Hz) per group; groups not named get no input."""
        target = self.rates if trial is None else self.rates[trial:trial + 1]
        target.zero_()
        for name, hz in rates_hz.items():
            target[:, self.groups[name]] = hz

    @torch.no_grad()
    def run(self, duration_ms):
        """Advance the brain; return spike counts per neuron, shape (batch, num_neurons)."""
        counts = torch.zeros(self.batch, self.num_neurons, device=self.device)
        state = self.state
        for _ in range(int(round(duration_ms / DT))):
            state = self.model(self.rates, *state)
            counts += state[2]
        self.state = state
        return counts

    def spikes(self, counts, trial=0):
        """(indices, spike counts) of the neurons that fired, for one trial."""
        row = counts[trial]
        fired = row.nonzero().squeeze(1)
        return fired.tolist(), row[fired].int().tolist()

    def group_rates(self, counts, duration_ms, names):
        """Mean firing rate (Hz) of each named group, one value per batch trial."""
        scale = 1000.0 / duration_ms
        return {name: (counts[:, self.groups[name]].mean(dim=1) * scale).tolist()
                if len(self.groups[name]) else [0.0] * self.batch
                for name in names}
