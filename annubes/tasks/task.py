import colorsys
import warnings
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots


@dataclass
class Task:
    """General class for defining a task.

    Args:
        name (str): name of the task.
        session_in (dict[str, float], optional): Dictionary representing the session. Keys are the modalities
            and values are the probabilities. If order matters, put the modalities in the right order.
            Defaults to {'v': 0.5, 'a': 0.5}.
        ordered (bool, optional): True if you want to keep the order of the modalities. Defaults to False.
        t_in (int, optional): time for each stimulus in ms. Defaults to 1000.
        value_in (list[float], optional): intensity values for each stimulus. Defaults to [.8, .9, 1].
        scaling (bool, optional): True if you want to scale the input. Defaults to False.
        t_fixation (int | None, optional): time for fixation in ms. Defaults to 100.
        value_fixation (float | None, optional): intensity value for fixation. Defaults to None.
        max_sequential (int | None, optional): maximum number of sequential trials of the same modality.
            Defaults to None.
        catch_prob (float, optional): probability of catch trials in the session, between 0 and 1.
            Defaults to 0.
        inter_trial (int, optional): inter-trial interval in ms. Defaults to 0.
        dt (int, optional): time step in ms. Defaults to 20.
        tau (int, optional): time constant in ms. Defaults to 100.
        std_inp_noise (float, optional): standard deviation for input noise. Defaults to 0.01.
        baseline_inp (float, optional): baseline input for all neurons. Defaults to 0.2.
        n_out (int, optional): number of outputs. Defaults to 2.
        value_out (list[float], optional): low and high intensity values for the output signals.
            Defaults to [0, 1].

    Raises:
        ValueError: if the sum of the probabilities of `session_in` is not 1.
        ValueError: if `catch_prob` is not between 0 (included) and 1 (excluded).
    """

    name: str
    session_in: dict[str, float] = field(default_factory=lambda: {"v": 0.5, "a": 0.5})
    t_in: int = 1000
    value_in: list[float] = field(default_factory=lambda: [0.8, 0.9, 1])
    coeff: float = 0
    t_fixation: int | None = 100
    value_fixation: float | None = None
    catch_prob: float = 0
    inter_trial: int = 0
    dt: int = 20
    value_out: list[float] = field(default_factory=lambda: [0, 1])

    def __post_init__(self):
        if not self.catch_prob >= 0 and self.catch_prob < 1:
            raise ValueError("`catch_prob` must be higher or equal to 0, or lower than 1.")

        sum_session_in = sum(self.session_in.values())
        for i in self.session_in:
            self.session_in[i] = self.session_in[i] / sum_session_in
        self.value_in.sort()
        self.value_out.sort()
        if self.value_fixation is not None:
            self.value_fixation = self._rescale(self.value_fixation)

        # Derived attributes
        self.modalities = list(dict.fromkeys(char for string in self.session_in for char in string))
        self.n_modalities = len(self.modalities)  # does not include start cue
        duration = self.inter_trial + self.t_fixation + self.t_in
        self.t = np.linspace(0, duration, int((duration + self.dt) / self.dt))  # TODO: rename attribute

    def _rescale(
        self,
        input_: float,
        min_intensity: float | None = None,
        max_intensity: float | None = None,
    ) -> float:
        """Rescale `input_` value if `self.coeff` is non-zero.

        Args:
            input_ (float): Value that will be rescaled.
            min_intensity (float): Minimum value of the input intensities. Defaults to `min(self.value_in)`
            max_intensity (float): Maximum value of the input intensities. Defaults to `max(self.value_in)`

        Returns:
            float: scaled input value.
        """
        if not self.coeff:
            return input_

        if min_intensity is None:
            min_intensity = min(self.value_in)
        if max_intensity is None:
            max_intensity = max(self.value_in)

        try:
            return self.coeff * (input_ - min_intensity) / (max_intensity - min_intensity)
        except ZeroDivisionError:
            warnings.warn("Identical max and min intensities while rescaling. Returning unmodified input value.")
            return input_

    def generate_trials(
        self,
        ntrials: int = 20,
        shuffle: bool = True,
        max_sequential: int = 0,
        input_baseline: float = 0.2,
        tau: int = 100,
        noise_std: float = 0.01,
        random_seed: int | None = None,
    ) -> None:
        """Method for generating trials. It populates the `trials` attribute.

        Args:
            ntrials: Number of trials to generate. Defaults to 20.
            shuffle: `False` will maintain the order of `self.session_in`. `True` will shuffle the order of the trials.
                Defaults to True.
            max_sequential: Maximum number of sequential trials of the same modality. Only used if shuffle is True.
                Defaults to 0 (no maximum).
            input_baseline (float, optional): Baseline input for all neurons.
                Defaults to 0.2.
            tau (int, optional): Time constant in ms.  # TODO: needs better clarification
                Defaults to 100.
            noise_std (float, optional): Standard deviation of input noise.
                Defaults to 0.01.
            random_seed: Seed for numpy random number generator (rng).
                Defaults to None (random seed).
        """
        self._ntrials = ntrials
        rng = np.random.default_rng(random_seed)

        # Generate sequence of modalities
        modality_seq = self._build_trials_seq(shuffle, max_sequential, rng)

        # Setup phases of trial
        phases = {}
        phases["inter_trial"] = np.where(self.t <= self.inter_trial)[0]
        phases["t_fixation"] = np.where((self.t > self.inter_trial) & (self.t <= self.inter_trial + self.t_fixation))[0]
        phases["input"] = np.where(self.t > self.inter_trial + self.t_fixation)[0]
        choice = (modality_seq != "catch").astype(np.int_)

        # Trial Info
        self.trials = {"name": self.name}
        self.trials["modality_seq"] = modality_seq
        self.trials["choice"] = choice
        self.trials["phases"] = phases
        self.trials["t"] = self.t
        self.trials["value_fixation"] = self.value_fixation

        # Generate and store inputs and outputs
        alpha = self.dt / tau
        noise_factor = noise_std * np.sqrt(2 * alpha) / alpha
        self.trials["inputs"] = self._build_trials_inputs(input_baseline, noise_factor, rng)
        self.trials["outputs"] = self._build_trials_outputs()

    def _build_trials_seq(
        self,
        shuffle: bool,
        max_sequential: int,
        rng: np.random.Generator,
    ) -> NDArray:
        """Generate a sequence of modalities.

        Returns:
            NDarray[str]: list of modalities.
        """
        # Extract keys and probabilities from the dictionary
        scenarios = list(self.session_in.keys())
        probabilities = np.array(list(self.session_in.values()))
        # Generate random numbers of samples based on the probabilities
        prob_samples = rng.multinomial(self._ntrials, probabilities)
        # Create a dictionary to store the results
        session_in_samples = {
            scenario: rng.multinomial(prob_samples[i], [1 - self.catch_prob, self.catch_prob])
            for i, scenario in enumerate(scenarios)
        }
        # Generate the sequence of modalities
        modality_seq = []
        for m in scenarios:
            temp_seq = session_in_samples[m][0] * [m] + session_in_samples[m][1] * ["catch"]
            rng.shuffle(temp_seq)
            modality_seq += list(temp_seq)
        if shuffle:
            rng.shuffle(modality_seq)
            if max_sequential:
                # Shuffle the list using Fisher-Yates algorithm with consecutive constraint
                i = len(modality_seq) - 1
                while i > 0:
                    # Picking j can't be fixed, otherwise the algorithm is not random
                    # We may want to change this in the future
                    j = rng.integers(0, i)
                    modality_seq[i], modality_seq[j] = modality_seq[j], modality_seq[i]
                    i -= 1
                    # Check and fix the consecutive constraint
                    count = 1
                    while i > 0 and modality_seq[i] == modality_seq[i - 1] and count >= max_sequential:
                        i -= 1

        return np.array(modality_seq)

    def _build_trials_inputs(
        self,
        input_baseline: float,
        noise_factor: float,
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        """Generate trial inputs.

        Returns:
            NDarray[np.float32]: array of inputs.
        """
        x = np.zeros(
            (self._ntrials, len(self.t), self.n_modalities + 1), dtype=np.float32
        )  # n_modalities+1 for start cue
        sel_value_in = np.full(
            (self._ntrials, self.n_modalities), min(self.value_in), dtype=np.float32
        )  # TODO: needs a better name

        modality_seq = self.trials["modality_seq"]
        phases = self.trials["phases"]
        for n in range(self._ntrials):
            for idx, m in enumerate(self.modalities):
                if (modality_seq[n] != "catch") and (m in modality_seq[n]):
                    sel_value_in[n, idx] = rng.choice(self.value_in[1:], 1)
                sel_value_in[n, idx] = self._rescale(sel_value_in[n, idx])
                x[n, phases["input"], idx] = sel_value_in[n, idx]
                x[n, phases["t_fixation"], idx] = self.value_fixation
            x[n, phases["input"], self.n_modalities] = 1  # start cue

        # Store intensities in trials
        self.trials["sel_value_in"] = sel_value_in

        noise = noise_factor * rng.normal(loc=0, scale=1, size=x.shape)
        return x + input_baseline + noise

    def _build_trials_outputs(self) -> NDArray:
        """Generate trial outputs.

        Returns:
            NDarray[np.float32]: array of outputs.
        """
        phases = self.trials["phases"]
        choice = self.trials["choice"]

        y = np.zeros((self._ntrials, len(self.t), len(self.value_out)), dtype=np.float32)
        for i in range(self._ntrials):
            if self.inter_trial is not None:
                y[i, phases["inter_trial"], :] = min(self.value_out)
            if self.t_fixation is not None:
                y[i, phases["t_fixation"], :] = min(self.value_out)

            y[i, phases["input"], choice[i]] = max(self.value_out)
            y[i, phases["input"], 1 - choice[i]] = min(self.value_out)

        return y

    def plot_trials(self, n: int = 1) -> go.Figure:
        """Method for plotting generated trials.

        Args:
            n (int, optional): number of trials to plot. Defaults to 1.

        Raises:
            ValueError: if `n` is greater than the number of trials generated.

        Returns:
            go.Figure: plotly figure.
        """
        if n > self.trials["inputs"].shape[0]:
            raise ValueError("n cannot be greater than the number of trials generated.")

        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.5 / n,
            subplot_titles=[
                "Trial " + str(i + 1) + " - modality " + str(self.trials["modality_seq"][i]) for i in range(n)
            ],
        )
        showlegend = True
        colors = [
            "#{:02x}{:02x}{:02x}".format(
                *tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / self.n_modalities, 1.0, 1.0))
            )
            for i in range(self.n_modalities)
        ]
        for i in range(n):
            for idx, m in enumerate(self.modalities):
                fig.add_trace(
                    go.Scatter(
                        name=m,
                        mode="markers+lines",
                        x=self.trials["t"],
                        y=self.trials["inputs"][i][:, idx],
                        marker_symbol="star",
                        legendgroup=m,
                        showlegend=showlegend,
                        line_color=colors[idx],
                    ),
                    row=i + 1,
                    col=1,
                )
            fig.add_trace(
                go.Scatter(
                    name="START",
                    mode="markers+lines",
                    x=self.trials["t"],
                    y=self.trials["inputs"][i][:, self.n_modalities],
                    marker_symbol="star",
                    legendgroup="START",
                    showlegend=showlegend,
                    line_color="green",
                ),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    name="Choice 1: NO STIMULUS",
                    mode="lines",
                    x=self.trials["t"],
                    y=self.trials["outputs"][i][:, 0],
                    legendgroup="Choice 1",
                    showlegend=showlegend,
                    line_color="orange",
                ),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    name="Choice 2: STIMULUS",
                    mode="lines",
                    x=self.trials["t"],
                    y=self.trials["outputs"][i][:, 1],
                    legendgroup="Choice 2",
                    showlegend=showlegend,
                    line_color="purple",
                ),
                row=i + 1,
                col=1,
            )
            fig.add_vline(x=self.inter_trial + self.t_fixation, line_width=3, line_dash="dash", line_color="red")
            showlegend = False
        fig.update_layout(height=1300, width=900, title_text="Trials")
        return fig
