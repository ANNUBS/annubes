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
        iti: Inter-trial interval, or time window between sequential trials, in ms. If a tuple is given, it is
            interpreted as an interval of possible values, and for each trial the value will be randomly picked from it.
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
    iti: int | tuple[int, int] = 0
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
        # Check input parameters
        ## Check str
        self._check_str("name", self.name)
        ## Check dict
        self._check_session("session", self.session)
        ## Check float
        for intensity in self.stim_intensities:
            self._check_float_positive("stim_intensities", intensity)
        self._check_float_positive("catch_prob", self.catch_prob, prob=True)
        self._check_float_positive("fix_intensity", self.fix_intensity)
        for intensity in self.output_behavior:
            self._check_float_positive("output_behavior", intensity)
        self._check_float_positive("noise_std", self.noise_std)
        ## Check int
        self._check_time_vars()
        if self.max_sequential is not None:
            self._check_int_positive("max_sequential", self.max_sequential, strict=True)
        self._check_int_positive("n_outputs", self.n_outputs, strict=True)
        ## Check bool
        self._check_bool("shuffle_trials", self.shuffle_trials)
        self._check_bool("scaling", self.scaling)

        self._task_settings = vars(self).copy()

        sum_session_vals = sum(self.session.values())
        self._session = {}
        for i in self.session:
            self._session[i] = self.session[i] / sum_session_vals
        if not self.shuffle_trials:
            self._session = OrderedDict(self._session)

        # Derived and other attributes
        self._modalities = set(dict.fromkeys(char for string in self._session for char in string))
        self._n_inputs = len(self._modalities) + 1  # includes start cue

    def generate_trials(
        self,
        ntrials: int | tuple[int, int] = 20,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        """Method for generating trials.

        Args:
            ntrials: Number of trials to generate. If a tuple is given, it is interpreted as an interval of
                possible values, and a value will be randomly picked from it.
                Defaults to 20.
            random_seed: Seed for numpy's random number generator (rng). If an int is given, it will be used as the seed
                for `np.random.default_rng()`.
                Defaults to None (i.e. the initial state itself is random).

        Returns:
            dict containing all input parameters of `Task` ("task_settings"), the input parameters for the current
            `generate_trials()` method's call ("ntrials", "random_state"), and the generated data ("modality_seq",
            "time", "phases", "inputs", "outputs").
        """
        # Check input parameters
        if isinstance(ntrials, tuple):
            if len(ntrials) != 2:  # noqa: PLR2004
                msg = "`ntrials` must be an integer or a tuple of two integers."
                raise ValueError(msg)
            self._check_int_positive("ntrials", ntrials[0], strict=True)
            self._check_int_positive("ntrials", ntrials[1], strict=True)
        else:
            self._check_int_positive("ntrials", ntrials, strict=True)
        if random_seed is not None:
            self._check_int_positive("random_seed", random_seed, strict=False)

        # Set random state
        if random_seed is None:
            rng = np.random.default_rng(random_seed)
            random_seed = rng.integers(2**32)
        self._rng = np.random.default_rng(random_seed)
        self._random_seed = random_seed

        self._ntrials = self._rng.integers(min(ntrials), max(ntrials)) if isinstance(ntrials, tuple) else ntrials

        # Generate sequence of modalities
        self._modality_seq = self._build_trials_seq()

        # Setup phases of trial
        self._iti, self._time, self._phases = self._setup_trial_phases()

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
        # Check input parameters
        self._check_int_positive("n_plots", n_plots, strict=True)

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
                        x=self._time[i],
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
                    x=self._time[i],
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
                    x=self._time[i],
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
                    x=self._time[i],
                    y=self._outputs[i][:, 1],
                    legendgroup="Choice 2",
                    showlegend=showlegend,
                    line_color="purple",
                ),
                row=i + 1,
                col=1,
            )
            fig.add_vline(
                x=self._iti[i] + self.fix_time + self.dt,
                line_width=3,
                line_dash="dash",
                line_color="red",
                row=i + 1,
                col=1,
            )
            showlegend = False
        fig.update_layout(height=1300, width=900, title_text="Trials")
        return fig

    def _check_str(self, name: str, value: Any) -> None:  # noqa: ANN401
        if not isinstance(value, str):
            msg = f"`{name}` must be a string"
            raise TypeError(msg)

    def _check_session(self, name: str, value: Any) -> None:  # noqa: ANN401
        if not isinstance(value, dict):
            msg = f"`{name}` must be a dictionary."
            raise TypeError(msg)
        if not all(isinstance(k, str) for k in value):
            msg = f"Keys of `{name}` must be strings."
            raise TypeError(msg)
        if not all(isinstance(v, (float | int)) for v in value.values()):
            msg = f"Values of `{name}` must be floats or integers."
            raise TypeError(msg)

    def _check_float_positive(self, name: str, value: Any, prob: bool = False) -> None:  # noqa: ANN401
        if not isinstance(value, float | int):
            msg = f"`{name}` must be a float or integer."
            raise TypeError(msg)
        if not value >= 0:
            msg = f"`{name}` must be greater than or equal to 0."
            raise ValueError(msg)
        if prob and not 0 <= value <= 1:
            msg = f"`{name}` must be between 0 and 1."
            raise ValueError(msg)

    def _check_int_positive(self, name: str, value: Any, strict: bool) -> None:  # noqa: ANN401
        if not isinstance(value, int | np.int_):
            msg = f"`{name}` must be an integer."
            raise TypeError(msg)
        if strict:
            if not value > 0:
                msg = f"`{name}` must be greater than 0."
                raise ValueError(msg)
        elif not value >= 0:
            msg = f"`{name}` must be greater than or equal to 0."
            raise ValueError(msg)

    def _check_bool(self, name: str, value: Any) -> None:  # noqa: ANN401
        if not isinstance(value, bool):
            msg = f"`{name}` must be a boolean."
            raise TypeError(msg)

    def _check_time_vars(self) -> None:
        strictly_positive = {
            "stim_time": (self.stim_time, True),
            "dt": (self.dt, True),
            "tau": (self.tau, True),
            "fix_time": (self.fix_time, False),
        }
        for name, value in strictly_positive.items():
            self._check_int_positive(name, value[0], strict=value[1])
        if isinstance(self.iti, tuple):
            for iti in self.iti:
                self._check_int_positive("iti", iti, strict=False)
        else:
            self._check_int_positive("iti", self.iti, strict=False)

    def _build_trials_seq(self) -> NDArray[np.str_]:
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
        joined_seq = "".join(modality_seq)
        for mod in self._session:
            violations = joined_seq.count(mod * (self.max_sequential + 1))
            if violations > 0:
                warnings.warn(
                    f"`max_sequential` limit of {self.max_sequential} was violated {violations} times for {mod}.\n"
                    "Please check the current trials sequence, and if it is not acceptable redefine your task or try"
                    f" to change the random seed, now set to {self._random_seed}.\n",
                    stacklevel=2,
                )
        return np.array(modality_seq)

    def _setup_trial_phases(
        self,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[Any],
    ]:
        """Setup phases of trial, time-wise."""
        # Generate inter-trial duration sequence
        if type(self.iti) is tuple:
            iti = self._rng.integers(min(self.iti), max(self.iti), self._ntrials)
            iti = np.array([round(i / 100) * 100 for i in iti])  # round to the nearest hundred
        else:
            iti = np.full(self._ntrials, self.iti)
        # Generate time sequence for each trial
        time = np.empty(self._ntrials, dtype=object)
        phases = np.empty(self._ntrials, dtype=object)
        for n in range(self._ntrials):
            trial_duration = iti[n] + self.fix_time + self.stim_time
            time[n] = np.linspace(0, trial_duration, int((trial_duration + self.dt) / self.dt))
            phases[n] = {}
            phases[n]["iti"] = np.where(time[n] <= iti[n])[0]
            phases[n]["fix_time"] = np.where(
                (time[n] > iti[n]) & (time[n] <= iti[n] + self.fix_time),
            )[0]
            phases[n]["input"] = np.where(time[n] > iti[n] + self.fix_time)[0]
        return iti, time, phases

    def _minmaxscaler(
        self,
        input_: NDArray[np.float64],
        rescale_range: tuple[float, float] = (0, 1),
    ) -> NDArray[np.float64]:
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

        return np.array(input_std * (max(rescale_range) - min(rescale_range)) + min(rescale_range))

    def _build_trials_inputs(self) -> NDArray[np.float64]:
        """Generate trials time and inputs ndarrays."""
        x = np.empty(self._ntrials, dtype=object)
        for n in range(self._ntrials):
            x[n] = np.zeros(
                (len(self._time[n]), self._n_inputs),
                dtype=np.float32,
            )
            for idx, _ in enumerate(self._modalities):
                value = self._rng.choice(self.stim_intensities, 1) if self._modality_seq[n] != "catch" else 0
                x[n][self._phases[n]["fix_time"], idx] = self.fix_intensity
                x[n][self._phases[n]["input"], idx] = value
            x[n][self._phases[n]["input"], self._n_inputs - 1] = 1  # start cue
            # generate and add noise
            alpha = self.dt / self.tau
            noise_factor = self.noise_std * np.sqrt(2 * alpha) / alpha
            x[n] += noise_factor * self._rng.normal(loc=0, scale=1, size=x[n].shape)

            if self.scaling:
                x[n] = self._minmaxscaler(x[n])

        return x

    def _build_trials_outputs(self) -> NDArray[np.float64]:
        """Generate trials outputs ndarray."""
        y = np.empty(self._ntrials, dtype=object)
        choice = (self._modality_seq != "catch").astype(np.int_)
        for n in range(self._ntrials):
            y[n] = np.full((len(self._time[n]), self.n_outputs), min(self.output_behavior), dtype=np.float32)
            y[n][self._phases[n]["input"], choice[n]] = max(self.output_behavior)
            y[n][self._phases[n]["input"], 1 - choice[n]] = min(self.output_behavior)

            if self.scaling:
                y[n] = self._minmaxscaler(y[n])

        return y
