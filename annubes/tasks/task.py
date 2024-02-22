import colorsys
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots


@dataclass
class TaskSettingsMixin:
    """Mixin data class for defining attributes related to extra Task settings.

    Args:
        fix_intensity: Intensity of input signal during fixation.
            Defaults to 0.
        fix_time: Fixation time in ms. Note that the duration of each input and output signal is increased by this time.
            Defaults to 100.
        iti: Inter-trial interval, or time window between sequential trials, in ms.
            Defaults to 0.
        dt: Sampling interval (inverted sampling frequency) in ms.
            Defaults to 20.
        tau: Time constant for the dynamics of each network node in ms.
            Defaults to 100.
        n_outputs: Number of output nodes in the network, signaling different behavioral choices.
            Defaults to 2.
        output_behavior: List of possible intensity values of the behavioral output. Currently only the smallest and
            largest value of this list are used.
            Defaults to [0, 1].
        input_baseline: Baseline input for all neurons.
            Defaults to 0.2.
        noise_std: Standard deviation of input noise.
            Defaults to 0.01.
        rescaling_coeff: Rescaling coefficient for `Task.stim_intensities` and `self.fix_intensity`. If set to non-zero
            value, these values are linearly rescaled along (0, rescaling_coeff).
            Defaults to 0 (i.e. no rescaling).
    """

    fix_intensity: float = 0
    fix_time: int = 100
    iti: int = 0
    dt: int = 20
    tau: int = 100
    n_outputs: int = 2
    output_behavior: list[float] = field(default_factory=lambda: [0, 1])
    input_baseline: float = 0.2
    noise_std: float = 0.01
    rescaling_coeff: float = 0


@dataclass
class Task(TaskSettingsMixin):
    """General data class for defining a task.

    A task is defined by a set of trials, each of which is characterized by a sequence of inputs and expected outputs.

    Args:
        name: Name of the task.
        session: Configuration of the trials that can appear during a session.
            It is given by a dictionary representing the ratio (values) of the different trials (keys) within the task.
            Trials with a single modality (e.g., a visual trial) must be represented by single characters, while trials
            with multiple modalities (e.g., an audiovisual trial) are represented by the character combination of those
            trials. Note that values are read relative to each other, such that e.g. `{"v": 0.25, "a": 0.75}` ==
            `{"v": 1, "a": 3}` is True.
            Defaults to {"v": 0.5, "a": 0.5}.
        stim_intensities: List of possible intensity values of each stimulus.
            Defaults to [0.8, 0.9, 1].
        stim_time: Duration of each stimulus in ms.
            Defaults to 1000.
        catch_prob: probability of catch trials in the session, between 0 and 1.
            Defaults to 0.5.
        shuffle_trials: If True (default), trial order will be randomized. If False, all trials corresponding to one
            modality (e.g. visual) are run before any trial of the next modality (e.g. auditory) starts, in the order
            defined in `session`, followed by catch trials.
        max_sequential: Maximum number of sequential trials of the same modality. Only used if shuffle is True.
            Defaults to None (no maximum).

    Raises:
        ValueError: if `catch_prob` is not between 0 and 1.
    """

    name: str
    session: dict[str, float] = field(default_factory=lambda: {"v": 0.5, "a": 0.5})
    stim_intensities: list[float] = field(default_factory=lambda: [0.8, 0.9, 1])
    stim_time: int = 1000
    catch_prob: float = 0.5
    shuffle_trials: bool = True
    max_sequential: int | None = None

    def __post_init__(self):
        if not self.catch_prob >= 0 and self.catch_prob < 1:
            msg = "`catch_prob` must be between 0 and 1."
            raise ValueError(msg)

        sum_session_vals = sum(self.session.values())
        for i in self.session:
            self.session[i] = self.session[i] / sum_session_vals
        if not self.shuffle_trials:
            self.session = OrderedDict(self.session)

        # Derived attributes
        self.modalities = list(dict.fromkeys(char for string in self.session for char in string))
        self.n_inputs = len(self.modalities) + 1  # includes start cue
        trial_duration = self.iti + self.fix_time + self.stim_time
        self.time = np.linspace(
            0,
            trial_duration,
            int((trial_duration + self.dt) / self.dt),
        )  # TODO: rename attribute

    def generate_trials(
        self,
        ntrials: int = 20,
        random_seed: int | dict | None = None,
    ) -> dict[str, Any]:
        """Method for generating trials.

        Args:
            ntrials: Number of trials to generate.
                Defaults to 20.
            random_seed: Seed for numpy's random number generator (rng). If an int is given, it will be used as the seed
                for `np.random.default_rng()`. If a dict is given, it must be in the form of a random state as given by
                `rng.__getstate__()` (previous runs will have stored this value as `self.trials["random_state"]`).
                Defaults to None (i.e. the initial state itself is random).

        Returns:
            dict containing all attributes of `Task`, including internal ones created in this method, as well as the
            inputs and outputs of the generated trials.
        """
        self._ntrials = ntrials

        # Set random state
        if isinstance(random_seed, dict):
            np.random.default_rng().__setstate__(random_seed)
        else:
            self._rng = np.random.default_rng(random_seed)
        self._random_state = self._rng.__getstate__()

        # Generate sequence of modalities
        self._modality_seq = self._build_trials_seq()

        # Setup phases of trial
        self._phases = {}
        self._phases["iti"] = np.where(self.time <= self.iti)[0]
        self._phases["fix_time"] = np.where(
            (self.time > self.iti) & (self.time <= self.iti + self.fix_time),
        )[0]
        self._phases["input"] = np.where(self.time > self.iti + self.fix_time)[0]
        self._choice = (self._modality_seq != "catch").astype(np.int_)

        # Generate inputs and outputs
        self._inputs = self._build_trials_inputs()
        self._outputs = self._build_trials_outputs()

        # Store and return trial data
        trials = vars(self)
        trials["inputs"] = self._inputs
        trials["outputs"] = self._outputs
        return trials

    def _rescale(
        self,
        input_: float,
        coeff: float,
        min_intensity: float | None = None,
        max_intensity: float | None = None,
    ) -> float:
        """Rescale `input_` value along (0,`coeff`) if `coeff` is non-zero.

        Rescaling happens as follows:
            coeff * (input_ - min_intensity) / (max_intensity - min_intensity)

        Args:
            input_: Value that will be rescaled.
            coeff: Maximum value after rescaling. If set to 0, `input_` is returned unmodified.
            min_intensity: Minimum value of the input intensities. Defaults to `min(self.stim_intensities)`
            max_intensity: Maximum value of the input intensities. Defaults to `max(self.stim_intensities)`

        Returns:
            float: Rescaled input value.
        """
        if not coeff:
            return input_
        if min_intensity is None:
            min_intensity = min(self.stim_intensities)
        if max_intensity is None:
            max_intensity = max(self.stim_intensities)

        try:
            return coeff * (input_ - min_intensity) / (max_intensity - min_intensity)
        except ZeroDivisionError:
            warnings.warn(
                "Identical max and min intensities while rescaling. Returning unmodified input value.",
                stacklevel=2,
            )
            return input_

    def _build_trials_seq(self) -> NDArray:
        """Generate a sequence of modalities."""
        # Extract keys and probabilities from the dictionary
        scenarios = list(self.session.keys())
        probabilities = np.array(list(self.session.values()))
        # Generate random numbers of samples based on the probabilities
        prob_samples = self._rng.multinomial(self._ntrials, probabilities)
        # Create a dictionary to store the results
        session_in_samples = {
            scenario: self._rng.multinomial(prob_samples[i], [1 - self.catch_prob, self.catch_prob])
            for i, scenario in enumerate(scenarios)
        }
        # Generate the sequence of modalities
        modality_seq = []
        for m in scenarios:
            temp_seq = session_in_samples[m][0] * [m] + session_in_samples[m][1] * ["catch"]
            self._rng.shuffle(temp_seq)
            modality_seq += list(temp_seq)
        if self.shuffle_trials:
            self._rng.shuffle(modality_seq)
            if self.max_sequential:
                # Shuffle the list using Fisher-Yates algorithm with consecutive constraint
                i = len(modality_seq) - 1
                while i > 0:
                    # Picking j can't be fixed, otherwise the algorithm is not random
                    # We may want to change this in the future
                    j = self._rng.integers(0, i)
                    modality_seq[i], modality_seq[j] = modality_seq[j], modality_seq[i]
                    i -= 1
                    # Check and fix the consecutive constraint
                    count = 1
                    while i > 0 and modality_seq[i] == modality_seq[i - 1] and count >= self.max_sequential:
                        i -= 1
        return np.array(modality_seq)

    def _build_trials_inputs(self) -> NDArray[np.float32]:
        """Generate trial inputs."""
        x = np.zeros(
            (self._ntrials, len(self.time), self.n_inputs),
            dtype=np.float32,
        )
        sel_value_in = np.full(  # TODO: needs a better name
            (self._ntrials, self.n_inputs - 1),  # should not include start cue
            min(self.stim_intensities),
            dtype=np.float32,
        )

        for n in range(self._ntrials):
            for idx, m in enumerate(self.modalities):
                if (self._modality_seq[n] != "catch") and (m in self._modality_seq[n]):
                    sel_value_in[n, idx] = self._rng.choice(self.stim_intensities[1:], 1)
                sel_value_in[n, idx] = self._rescale(sel_value_in[n, idx], self.rescaling_coeff)
                x[n, self._phases["input"], idx] = sel_value_in[n, idx]
                x[n, self._phases["fix_time"], idx] = self._rescale(
                    self.fix_intensity,
                    self.rescaling_coeff,
                )
            x[n, self._phases["input"], self.n_inputs - 1] = 1  # start cue

        # generate noise
        alpha = self.dt / self.tau
        noise_factor = self.noise_std * np.sqrt(2 * alpha) / alpha
        noise = noise_factor * self._rng.normal(loc=0, scale=1, size=x.shape)

        return x + self.input_baseline + noise

    def _build_trials_outputs(self) -> NDArray[np.float32]:
        """Generate trial outputs."""
        y = np.zeros((self._ntrials, len(self.time), self.n_outputs), dtype=np.float32)
        for i in range(self._ntrials):
            if self.iti > 0:
                y[i, self._phases["iti"], :] = min(self.output_behavior)
            if self.fix_time > 0:
                y[i, self._phases["fix_time"], :] = min(self.output_behavior)

            y[i, self._phases["input"], self._choice[i]] = max(self.output_behavior)
            y[i, self._phases["input"], 1 - self._choice[i]] = min(self.output_behavior)

        return y

    def plot_trials(self, n_plots: int = 1) -> go.Figure:
        """Method for plotting generated trials.

        Args:
            n_plots: number of trials to plot (capped by number of trials generated). Defaults to 1.

        Returns:
            go.Figure: Plotly figure of trial results.
        """
        if (p := n_plots) > (t := self._ntrials):
            msg = f"Number of plots requested ({p}) exceeds number of trials ({t}). Will plot all trials."
            warnings.warn(msg, stacklevel=2)
            n_plots = self._ntrials

        fig = make_subplots(
            rows=n_plots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.5 / n_plots,
            subplot_titles=[f"Trial {i + 1}  - modality {self._modality_seq[i]}" for i in range(n_plots)],
        )
        showlegend = True
        colors = [
            "#{:02x}{:02x}{:02x}".format(
                *tuple(
                    int(c * 255) for c in colorsys.hsv_to_rgb(i / self.n_inputs - 1, 1.0, 1.0)
                ),  # should not include start cue
            )
            for i in range(self.n_inputs - 1)
        ]
        for i in range(n_plots):
            for idx, m in enumerate(self.modalities):
                fig.add_trace(
                    go.Scatter(
                        name=m,
                        mode="markers+lines",
                        x=self.time,
                        y=self._inputs[i][:, idx],
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
                    x=self.time,
                    y=self._inputs[i][:, self.n_inputs],  # not for start cue
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
                    x=self.time,
                    y=self._outputs[i][:, 0],
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
                    x=self.time,
                    y=self._outputs[i][:, 1],
                    legendgroup="Choice 2",
                    showlegend=showlegend,
                    line_color="purple",
                ),
                row=i + 1,
                col=1,
            )
            fig.add_vline(
                x=self.iti + self.fix_time,
                line_width=3,
                line_dash="dash",
                line_color="red",
            )
            showlegend = False
        fig.update_layout(height=1300, width=900, title_text="Trials")
        return fig
