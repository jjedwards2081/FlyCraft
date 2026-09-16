"""The whole FlyWire connectome, run continuously with fly-brain's PyTorch LIF model."""

from . import config  # noqa: F401  (puts fly-brain/code on sys.path)

import logging

import pyarrow  # noqa: F401  (fly-brain: import before torch to avoid a libarrow conflict)
import torch

from benchmark import path_comp, path_con, path_wt
from run_pytorch import DT, MODEL_PARAMS, TorchModel, get_weights

log = logging.getLogger(__name__)

EDGE_BUDGET = 1 << 15  # edges a step may gather; a typical step needs ~1,000, the worst seen 4,287


class ConnectomeModel(TorchModel):
    """fly-brain's TorchModel, with the recurrent input computed as W @ s.

    Same dynamics; this avoids transposing the CSR weight matrix on every 0.1 ms step.
    """

    def forward(self, rates, conductance, delay_buffer, spikes, v, refrac):
        voltage_stim = self.scale * self.poisson(rates)
        recurrent_input = self.scale * torch.sparse.mm(self.weights, spikes.T.contiguous()).T
        return self.neurons(recurrent_input, voltage_stim, conductance, delay_buffer, spikes, v, refrac)


class GatherModel(ConnectomeModel):
    """Same sum, but over the edges leaving the few neurons that spiked, not all 15 million.

    About 3 of 138,639 neurons spike in a 0.1 ms step, so W @ s reads the whole connectome to add
    up ~1,000 numbers. This adds up those same numbers, edge for edge (it is exact, not an
    approximation), and keeps every shape fixed so the step can be captured as a CUDA graph.
    """

    def prepare(self):
        """Index the edges by the neuron they leave, which is how spikes reach them."""
        columns = self.weights.to_sparse_coo().t().coalesce().to_sparse_csr()
        self.crow, self.col, self.val = columns.crow_indices(), columns.col_indices(), columns.values()
        self.fan_out = (self.crow[1:] - self.crow[:-1]).to(torch.int64)
        self.size = self.weights.shape[0]
        self.overflows = torch.zeros((), dtype=torch.int64, device=self.val.device)

    def set_budget(self, budget):
        self.budget = budget
        self.slots = torch.arange(budget, device=self.val.device)

    def recurrent(self, spikes):
        counts = self.fan_out * (spikes[0] > 0).to(torch.int64)  # edges each neuron contributes
        ends = torch.cumsum(counts, 0)                           # where each one's edges end
        starts = ends - counts
        total = ends[-1]
        # Counted on the GPU and read once a tick: a step never waits to find out
        self.overflows += (total > self.budget).to(torch.int64)
        owner = torch.searchsorted(ends, self.slots, right=True).clamp_(max=self.size - 1)
        edge = (self.crow[owner] + self.slots - starts[owner]).clamp_(max=self.col.numel() - 1)
        inside = (self.slots < total).to(self.val.dtype)  # slots past the last edge add nothing
        summed = torch.zeros_like(spikes)
        summed[0].index_add_(0, self.col[edge], self.val[edge] * inside)
        return summed

    def forward(self, rates, conductance, delay_buffer, spikes, v, refrac):
        voltage_stim = self.scale * self.poisson(rates)
        recurrent_input = self.scale * self.recurrent(spikes)
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
        # One brain on a GPU gathers spikes and replays a captured step; a batch of them (calibration) does not
        self.fast = batch == 1 and torch.device(self.device).type == 'cuda'
        build = GatherModel if self.fast else ConnectomeModel
        self.model = build(batch, self.num_neurons, DT, MODEL_PARAMS, weights,
                           exc_indices=list(stimulated), device=self.device)
        self.state = self.model.state_init()
        self.rates = torch.zeros(batch, self.num_neurons, device=self.device)
        self.groups = {name: torch.tensor(idx, dtype=torch.long, device=self.device)
                       for name, idx in groups.items()}
        self._graph = None
        self._counts = None
        self._overflows = 0
        if self.fast:
            self.model.prepare()
            self.model.set_budget(EDGE_BUDGET)

    def reset(self):
        """Start a new life: every neuron back at rest, with no input."""
        fresh = self.model.state_init()
        if self._graph is not None:
            for held, new in zip(self.state, fresh):  # the graph reads these tensors; keep them
                held.copy_(new)
        else:
            self.state = fresh
        self.rates.zero_()

    def set_input(self, rates_hz, trial=None):
        """Set Poisson input rates (Hz) per group; groups not named get no input."""
        target = self.rates if trial is None else self.rates[trial:trial + 1]
        target.zero_()
        for name, hz in rates_hz.items():
            target[:, self.groups[name]] = hz

    @torch.no_grad()
    def run(self, duration_ms):
        """Advance the brain; return spike counts per neuron, shape (batch, num_neurons)."""
        steps = int(round(duration_ms / DT))
        if not self.fast:
            counts = torch.zeros(self.batch, self.num_neurons, device=self.device)
            state = self.state
            for _ in range(steps):
                state = self.model(self.rates, *state)
                counts += state[2]
            self.state = state
            return counts

        graph = self._capture()
        rewind = [t.clone() for t in self.state]
        self._counts.zero_()
        for _ in range(steps):
            graph.replay()
        overflows = int(self.model.overflows)
        if overflows != self._overflows:
            # Some step had more spiking than the budget held, so run the tick again with room
            self._overflows = overflows
            budget = self.model.budget * 4
            log.warning('A brain step needed more than %d edges; widening to %d and running it again',
                        self.model.budget, budget)
            for held, saved in zip(self.state, rewind):
                held.copy_(saved)
            self._graph = None
            self.model.set_budget(budget)
            return self.run(duration_ms)
        return self._counts.clone()

    def _step(self):
        out = self.model(self.rates, *self.state)
        for held, new in zip(self.state, out):
            held.copy_(new)
        self._counts.add_(self.state[2])

    def _capture(self):
        """Record one 0.1 ms step as a CUDA graph, so replaying it costs one launch, not ~30."""
        if self._graph is not None:
            return self._graph
        self.state = [t.clone() for t in self.state]  # fixed buffers the graph reads and writes
        self._counts = torch.zeros_like(self.state[2])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._step()
        log.info('Brain step captured as a CUDA graph, gathering up to %d edges a step', self.model.budget)
        return self._graph

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
