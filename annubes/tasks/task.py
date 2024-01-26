import itertools
import random
import colorsys
import sys
import numpy as np
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TODO: Complete docstrings
# TODO: Evaluate https://docs.pydantic.dev/latest/ for input validation
class Task:
    """General class for defining a task."""
    def __init__(self,
                 name: str,
                 session_in: dict[str, float] = {'v': 0.5, 'a': 0.5}, # use the right order
                 ordered: bool = False, # True if you want to keep the order of the modalities
                 t_in: int = 5000, # time for each stimulus in ms
                 value_in: list[float | int] = [.8, .9, 1], # intensity values for each stimulus
                 scaling: bool = False, # True if you want to scale the input
                 t_fixation: int | None = 100, # time for fixation in ms
                 value_fixation: int | float | None = None, # intensity value for fixation
                 max_sequential: int | None = None, # maximum number of sequential trials of the same modality
                 catch_prob: float | int = 0, # probability of catch trials in the session, between 0 and 1
                 inter_trial: int = 0, # inter-trial interval in ms
                 dt: int = 20, # time step in ms
                 tau: int = 100, # time constant in ms
                 std_inp_noise: float | int =  0.01, # standard deviation for input noise
                 baseline_inp: float | int = 0.2, # baseline input for all neurons
                 n_out: int = 2, # number of outputs
                 value_out: list[float | int] = [0, 1]
                 ):
        """Constructor method for Task class."""
        # Attributes from the class constructor parameters
        self.name = name
        self.session_in = OrderedDict(session_in)
        self.ordered = ordered
        self.t_in = t_in
        self.value_in = value_in
        self.scaling = scaling
        self.t_fixation = t_fixation
        self.value_fixation = value_fixation
        self.max_sequential = max_sequential
        self.catch_prob = catch_prob
        self.inter_trial = inter_trial
        self.dt = dt
        self.tau = tau
        self.std_inp_noise = std_inp_noise
        self.baseline_inp = baseline_inp
        self.n_out = n_out
        self.value_out = value_out

        # Derived attributes
        self.trials = {}
        self.trials['name'] = self.name
        self.modalities = list(OrderedDict.fromkeys(char for string in session_in for char in string))
        self.modality_idx = {m: i for i, m in enumerate(self.modalities)}
        self.n_in = len(self.modalities) + 1 # +1 for start cue
        self.value_in.sort()
        self.imin = self.value_in[0]
        self.imax = self.value_in[-1]
        if (self.value_fixation is not None) and self.scaling:
            self.value_fixation = self._scale_input(self.value_fixation, self.imin, self.imax)
        self.T = self.inter_trial + self.t_fixation + self.t_in
        self.t = np.linspace(0, self.T, self.dt)
        self.t = np.linspace(0, self.T, int((self.T + self.dt)/self.dt))
        self.value_out.sort()
        self.low_out = self.value_out[0]
        self.high_out = self.value_out[1]

        # Checks
        tolerance = 1/50
        if not abs(sum(self.session_in.values()) - 1) < tolerance:
            raise ValueError('The sum of the probabilities of `session_in` must be 1.')

    def _scale_input(self, f, min, max):
        """Method for scaling input."""
        return 0.6*(f - min) / (max - min)

    def _build_trials_seq(self, batch_size, rng):

        # Extract keys and probabilities from the dictionary
        scenarios = list(self.session_in.keys())
        probabilities = np.array(list(self.session_in.values()))
        # Normalize probabilities to ensure they sum to 1
        probabilities /= probabilities.sum()
        # Generate random numbers of samples based on the probabilities
        prob_samples = np.random.multinomial(batch_size, probabilities)
        # Create a dictionary to store the results
        session_in_samples = {
            scenario: np.random.multinomial(prob_samples[i], [1-self.catch_prob, self.catch_prob])
                for i, scenario in enumerate(scenarios)}
        # Generate the sequence of modalities
        modality_seq = []
        for m in scenarios:
            temp_seq = session_in_samples[m][0] * [m] + session_in_samples[m][1] * ['catch']
            rng.shuffle(temp_seq)
            modality_seq += list(temp_seq)
        if not self.ordered:
            rng.shuffle(modality_seq)
            if self.max_sequential is not None:
                # Shuffle the list using Fisher-Yates algorithm with consecutive constraint
                i = len(modality_seq) - 1
                while i > 0:
                    # Picking j can't be fixed, otherwise the algorithm is not random
                    # We may want to change this in the future
                    j = random.randint(0, i)
                    modality_seq[i], modality_seq[j] = modality_seq[j], modality_seq[i]
                    i -= 1
                    # Check and fix the consecutive constraint
                    count = 1
                    while i > 0 and modality_seq[i] == modality_seq[i - 1] and count >= self.max_sequential:
                        i -= 1
        modality_seq = np.array(modality_seq)
        return modality_seq

    def _build_trials_inputs(self, batch_size, modality_seq, rng, phases):
        x = np.zeros((batch_size, len(self.t), self.n_in), dtype=np.float32)
        sel_value_in = np.full((batch_size, self.n_in - 1), self.value_in[0], dtype=np.float32)

        for n in range(batch_size):
            for m, idx in self.modality_idx.items():
                if (modality_seq[n] != 'catch') and (m in modality_seq[n]):
                    sel_value_in[n, idx] = rng.choice(self.value_in[1:], 1)
                if self.scaling:
                    sel_value_in[n, idx] = self._scale_input(sel_value_in[n, idx], self.imin, self.imax)
                x[n, phases['input'], idx] = sel_value_in[n, idx]
                x[n, phases['t_fixation'], idx] = self.value_fixation
            x[n, phases['input'], len(self.modality_idx)] = 1 # start cue

        # Store intensities in trials
        self.trials['sel_value_in'] = sel_value_in

        # Add noise to inputs
        alpha = self.dt/self.tau
        inp_noise = 1/alpha * np.sqrt(2 * alpha) * self.std_inp_noise * rng.normal(loc=0, scale=1, size=x.shape)

        return x + self.baseline_inp + inp_noise

    def _build_trials_outputs(self, batch_size, phases, choice):
        y = np.zeros((batch_size, len(self.t), self.n_out), dtype=np.float32)
        for i in range(batch_size):
            if self.inter_trial is not None:
                y[i, phases['inter_trial'], :] = self.low_out
            if self.t_fixation is not None:
                y[i, phases['t_fixation'], :] = self.low_out

            y[i, phases['input'], choice[i]] = self.high_out
            y[i, phases['input'], 1 - choice[i]] = self.low_out

        return y

    def generate_trials(self,
                        batch_size: int = 20,
                        numpy_seed: int = None):
        """Method for generating synthetic trials."""
        # Set the seed for reproducibility
        if numpy_seed is None:
            numpy_seed = random.randrange(2**32 - 1)
        self.trials['numpy_seed'] = numpy_seed
        rng = np.random.default_rng(numpy_seed)
        np.random.seed(numpy_seed)

        # Generate sequence of modalities
        modality_seq = self._build_trials_seq(batch_size, rng)

        # Setup phases of trial
        phases = {}
        phases['inter_trial'] = np.where(self.t <= self.inter_trial)[0]
        phases['t_fixation'] = np.where((self.t > self.inter_trial) & (self.t <= self.inter_trial + self.t_fixation))[0]
        phases['input'] = np.where(self.t > self.inter_trial + self.t_fixation)[0]
        choice = (modality_seq != 'catch').astype(np.int_)

        # Trial Info
        self.trials['modality_seq'] = modality_seq
        self.trials['choice'] = choice
        self.trials['phases'] = phases
        self.trials['t'] = self.t
        self.trials['value_fixation'] = self.value_fixation

        # Generate and store inputs
        self.trials['inputs'] = self._build_trials_inputs(batch_size, modality_seq, rng, phases)
        # Generate and store outputs
        self.trials['outputs'] = self._build_trials_outputs(batch_size, phases, choice)

    def plot_trials(self, n = 1):
        """Method for plotting generated trials.

        Args:
            n (int, optional): _description_. Defaults to 1.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if n > self.trials['inputs'].shape[0]:
            raise ValueError('n cannot be greater than the number of trials generated.')

        fig = make_subplots(rows=n, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.5/n,
                            subplot_titles=[
                                "Trial " + str(i + 1) + " - modality " + str(self.trials['modality_seq'][i])
                                for i in range(n)])
        showlegend = True
        colors = [
            '#%02x%02x%02x' % tuple(int(c * 255)
                                    for c in colorsys.hsv_to_rgb(i / len(self.modality_idx), 1.0, 1.0))
                                    for i in range(len(self.modality_idx))]
        for i in range(n):
            for m, idx in self.modality_idx.items():
                fig.add_trace(go.Scatter(
                    name=m,
                    mode="markers+lines", x=self.trials['t'], y=self.trials['inputs'][i][:,idx],
                    marker_symbol="star",
                    legendgroup=m,
                    showlegend=showlegend,
                    line_color = colors[idx]
                ), row=i+1, col=1)
            fig.add_trace(go.Scatter(
                name="START",
                mode="markers+lines", x=self.trials['t'], y=self.trials['inputs'][i][:,len(self.modality_idx)],
                marker_symbol="star",
                legendgroup="START",
                showlegend=showlegend,
                line_color = 'green'
            ), row=i+1, col=1)
            fig.add_trace(go.Scatter(
                name="Choice 1: NO STIMULUS",
                mode="lines", x=self.trials['t'], y=self.trials['outputs'][i][:,0],
                legendgroup="Choice 1",
                showlegend=showlegend,
                line_color = 'orange'
            ), row=i+1, col=1)
            fig.add_trace(go.Scatter(
                name="Choice 2: STIMULUS",
                mode="lines", x=self.trials['t'], y=self.trials['outputs'][i][:,1],
                legendgroup="Choice 2",
                showlegend=showlegend,
                line_color = 'purple'
            ), row=i+1, col=1)
            fig.add_vline(
                x=self.inter_trial + self.t_fixation, line_width=3, line_dash="dash", line_color="red")
            showlegend = False
        fig.update_layout(height=1300, width=900, title_text="Trials")
        return fig
