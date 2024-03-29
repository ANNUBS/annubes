import colorsys
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots


@dataclass(kw_only=True)
class TaskSettingsMixin:
    """Mixin data class defining detailed parameters for `Task`.

    These settings are expected to be maintained throughout most experiments,
    whereas the attributes of `Task` itself are expected to be more commonly
    adjusted between individual experiments.

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
        noise_std: Standard deviation of input noise.
            Defaults to 0.01.
        scaling: If True, input and output signals are rescaled between 0 and 1. A MinMaxScaler logic is used
            for this purpose.
            Defaults to True.
    """

    fix_intensity: float = 0
    fix_time: int = 100
    iti: int = 0
    dt: int = 20
    tau: int = 100
    n_outputs: int = 2
    output_behavior: list[float] = field(default_factory=lambda: [0, 1])
    noise_std: float = 0.01
    scaling: bool = True


@dataclass()
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
        catch_prob: probability of catch trials in the session, between 0 and 1 (extremes included).
            Defaults to 0.5.
        shuffle_trials: If True (default), trial order will be randomized. If False, all trials corresponding to one
            modality (e.g. visual) are run before any trial of the next modality (e.g. auditory) starts, in the order
            defined in `session` (with randomly interspersed catch trials).
        max_sequential: Maximum number of sequential trials of the same modality. Only used if shuffle is True.
            Defaults to None (no maximum).

    Raises:
        ValueError: if `catch_prob` is not between 0 and 1.
        TypeError: if `catch_prob` is not a float.
    """

    name: str
    session: dict[str, float] = field(default_factory=lambda: {"v": 0.5, "a": 0.5})
    stim_intensities: list[float] = field(default_factory=lambda: [0.8, 0.9, 1])
    stim_time: int = 1000
    catch_prob: float = 0.5
    shuffle_trials: bool = True
    max_sequential: int | None = None

    def __post_init__(self):
        self._task_settings = vars(self).copy()

        if not 0 <= self.catch_prob <= 1:
            msg = "`catch_prob` must be between 0 and 1."
            raise ValueError(msg)

        sum_session_vals = sum(self.session.values())
        self._session = {}
        for i in self.session:
            self._session[i] = self.session[i] / sum_session_vals
        if not self.shuffle_trials:
            self._session = OrderedDict(self._session)

        if not self.dt > 0:
            msg = "`dt` must be greater than 0."
            raise ValueError(msg)
        if not self.tau > 0:
            msg = "`tau` must be greater than 0."
            raise ValueError(msg)

        # Derived and other attributes
        self._modalities = set(dict.fromkeys(char for string in self._session for char in string))
        self._n_inputs = len(self._modalities) + 1  # includes start cue
        trial_duration = self.iti + self.fix_time + self.stim_time
        self._time = np.linspace(
            0,
            trial_duration,
            int((trial_duration + self.dt) / self.dt),
        )  # TODO: rename attribute

    def generate_trials(
        self,
        ntrials: int = 20,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        """Method for generating trials.

        Args:
            ntrials: Number of trials to generate.
                Defaults to 20.
            random_seed: Seed for numpy's random number generator (rng). If an int is given, it will be used as the seed
                for `np.random.default_rng()`.
                Defaults to None (i.e. the initial state itself is random).

        Returns:
            dict containing all input parameters of `Task` ("task_settings"), the input parameters for the current
            `generate_trials()` method's call ("ntrials", "random_state"), and the generated data ("modality_seq",
            "time", "phases", "inputs", "outputs").
        """
        self._ntrials = ntrials

        # Set random state
        if random_seed is None:
            rng = np.random.default_rng(random_seed)
            random_seed = rng.integers(2**32)
        self._rng = np.random.default_rng(random_seed)
        self._random_seed = random_seed

        # Generate sequence of modalities
        self._modality_seq = self._build_trials_seq()

        # Setup phases of trial
        self._phases = self._setup_trial_phases()

        # Generate inputs and outputs
        self._inputs = self._build_trials_inputs()
        self._outputs = self._build_trials_outputs()

        # Store trials settings and data
        return {
            "task_settings": self._task_settings,
            "ntrials": self._ntrials,
            "random_seed": self._random_seed,
            "modality_seq": self._modality_seq,
            "time": self._time,
            "phases": self._phases,
            "inputs": self._inputs,
            "outputs": self._outputs,
        }

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
                *tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / self._n_inputs, 1.0, 1.0)),
            )
            for i in range(self._n_inputs)
        ]
        for i in range(n_plots):
            for idx, m in enumerate(self._modalities):
                fig.add_trace(
                    go.Scatter(
                        name=m,
                        mode="markers+lines",
                        x=self._time,
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
                    x=self._time,
                    y=self._inputs[i][:, self._n_inputs - 1],
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
                    x=self._time,
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
                    x=self._time,
                    y=self._outputs[i][:, 1],
                    legendgroup="Choice 2",
                    showlegend=showlegend,
                    line_color="purple",
                ),
                row=i + 1,
                col=1,
            )
            fig.add_vline(
                x=self.iti + self.fix_time + self.dt,
                line_width=3,
                line_dash="dash",
                line_color="red",
            )
            showlegend = False
        fig.update_layout(height=1300, width=900, title_text="Trials")
        return fig

    def _build_trials_seq(self) -> NDArray:
        """Generate a sequence of modalities."""
        # Extract keys and probabilities from the dictionary
        scenarios = list(self._session.keys())
        probabilities = np.array(list(self._session.values()))
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

    def _setup_trial_phases(self) -> dict[str, NDArray]:
        """Setup phases of trial, time-wise."""
        phases = {}
        phases["iti"] = np.where(self._time <= self.iti)[0]
        phases["fix_time"] = np.where(
            (self._time > self.iti) & (self._time <= self.iti + self.fix_time),
        )[0]
        phases["input"] = np.where(self._time > self.iti + self.fix_time)[0]
        return phases

    def _minmaxscaler(
        self,
        input_: NDArray[np.float32],
        rescale_range: tuple[float, float] = (0, 1),
    ) -> NDArray[np.float32]:
        """Rescale `input_` array to a given range.

        Rescaling happens as follows:

            `X_std = (input_ - input_.min()) / (input_.max() - input_.min())`
            `X_scaled = X_std * (max - min) + min`
            where min, max = range.
        The logic is the same as that of `sklearn.preprocessing.MinMaxScaler` estimator. Each array is rescaled to the
        given range, for each trial contained in `input_`.


        Args:
            input_: Input array of shape (self._ntrials, len(self._time), self._n_inputs).
            rescale_range: Desired range of transformed data. Defaults to (0, 1).

        Returns:
            Rescaled input array.
        """
        input_std = (input_ - input_.min()) / (input_.max() - input_.min())

        return input_std * (max(rescale_range) - min(rescale_range)) + min(rescale_range)

    def _build_trials_inputs(self) -> NDArray[np.float32]:
        """Generate trial inputs."""
        x = np.zeros(
            (self._ntrials, len(self._time), self._n_inputs),
            dtype=np.float32,
        )

        for n in range(self._ntrials):
            for idx, _ in enumerate(self._modalities):
                value = self._rng.choice(self.stim_intensities, 1) if self._modality_seq[n] != "catch" else 0
                x[n, self._phases["input"], idx] = value
                x[n, self._phases["fix_time"], idx] = self.fix_intensity
            x[n, self._phases["input"], self._n_inputs - 1] = 1  # start cue

        # generate and add noise
        alpha = self.dt / self.tau
        noise_factor = self.noise_std * np.sqrt(2 * alpha) / alpha
        x += noise_factor * self._rng.normal(loc=0, scale=1, size=x.shape)

        if self.scaling:
            x = self._minmaxscaler(x)

        return x

    def _build_trials_outputs(self) -> NDArray[np.float32]:
        """Generate trial outputs."""
        y = np.zeros((self._ntrials, len(self._time), self.n_outputs), dtype=np.float32)
        choice = (self._modality_seq != "catch").astype(np.int_)
        for i in range(self._ntrials):
            if self.iti > 0:
                y[i, self._phases["iti"], :] = min(self.output_behavior)
            if self.fix_time > 0:
                y[i, self._phases["fix_time"], :] = min(self.output_behavior)

            y[i, self._phases["input"], choice[i]] = max(self.output_behavior)
            y[i, self._phases["input"], 1 - choice[i]] = min(self.output_behavior)

        if self.scaling:
            y = self._minmaxscaler(y)

        return y
