import colorsys
import random
from collections import OrderedDict

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots


class Task:
    """General class for defining a task."""

    def __init__(
        self,
        name: str,
        session_in: dict[str, float] | None = None,
        ordered: bool = False,
        t_in: int = 1000,
        value_in: list[float] | None = None,
        scaling: bool = False,
        t_fixation: int | None = 100,
        value_fixation: float | None = None,
        max_sequential: int | None = None,
        catch_prob: float = 0,
        inter_trial: int = 0,
        dt: int = 20,
        tau: int = 100,
        std_inp_noise: float = 0.01,
        baseline_inp: float = 0.2,
        n_out: int = 2,
        value_out: list[float] | None = None,
    ):
        """Constructor method for the Task class.

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

        Attributes:
            name (str): name of the task.
            session_in (OrderedDict[str, float]): Ordered Dictionary representing the session.
                Keys are the modalities and values are the probabilities. If order matters,
                put the modalities in the right order. Defaults to {'v': 0.5, 'a': 0.5}.
            ordered (bool): True if you want to keep the order of the modalities. Defaults to False.
            t_in (int): time for each stimulus in ms. Defaults to 1000.
            value_in (list[float]): intensity values for each stimulus. Defaults to [.8, .9, 1].
            scaling (bool): True if you want to scale the input. Defaults to False.
            t_fixation (int | None): time for fixation in ms. Defaults to 100.
            value_fixation (float | None): intensity value for fixation. Defaults to None.
            max_sequential (int | None): maximum number of sequential trials of the same modality.
                Defaults to None.
            catch_prob (float): probability of catch trials in the session, between 0 and 1.
                Defaults to 0.
            inter_trial (int): inter-trial interval in ms. Defaults to 0.
            dt (int): time step in ms. Defaults to 20.
            tau (int): time constant in ms. Defaults to 100.
            std_inp_noise (float): standard deviation for input noise. Defaults to 0.01.
            baseline_inp (float): baseline input for all neurons. Defaults to 0.2.
            n_out (int): number of outputs. Defaults to 2.
            value_out (list[float]): low and high intensity values for the output signals.
                Defaults to [0, 1].
            trials (dict): dictionary containing all the trials generated and the task's relevant information.
            modalities (list[str]): list of individual modalities.
            modality_idx (dict[str, int]): dictionary of modalities and their index.
            n_in (int): number of inputs.
            imin (float): minimum value of the input intensities.
            imax (float): maximum value of the input intensities.
            coeff (float): coefficient for scaling the input.
            T (int): total time of each trial in ms.
            t (np.ndarray): time vector of each trial in ms.
            low_out (float): low intensity value for the output signals.
            high_out (float): high intensity value for the output signals.

        Raises:
            ValueError: if the sum of the probabilities of `session_in` is not 1.
            ValueError: if `catch_prob` is not between 0 (included) and 1 (excluded).
        """
        # Attributes from the class constructor parameters
        self.name = name
        self.session_in = OrderedDict(session_in) if session_in else OrderedDict({"v": 0.5, "a": 0.5})
        self.ordered = ordered
        self.t_in = t_in
        self.value_in = value_in if value_in else [0.8, 0.9, 1]
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
        self.value_out = value_out if value_out else [0, 1]

        # Derived attributes
        self.trials = {}
        self.trials["name"] = self.name
        self.modalities = list(OrderedDict.fromkeys(char for string in session_in for char in string))
        self.modality_idx = {m: i for i, m in enumerate(self.modalities)}
        self.n_in = len(self.modalities) + 1  # +1 for start cue
        self.value_in.sort()
        self.imin = self.value_in[0]
        self.imax = self.value_in[-1]
        self.coeff = 0.6
        if (self.value_fixation is not None) and self.scaling:
            self.value_fixation = self._scale_input(self.value_fixation, self.coeff, self.imin, self.imax)
        self.T = self.inter_trial + self.t_fixation + self.t_in
        self.t = np.linspace(0, self.T, self.dt)
        self.t = np.linspace(0, self.T, int((self.T + self.dt) / self.dt))
        self.value_out.sort()
        self.low_out = self.value_out[0]
        self.high_out = self.value_out[1]

        # Checks
        # session_in
        tolerance = 1 / 50
        if not abs(sum(self.session_in.values()) - 1) < tolerance:
            raise ValueError("The sum of the probabilities of `session_in` must be 1.")
        # catch_prob
        if not (catch_prob >= 0 and catch_prob < 1):
            raise ValueError("`catch_prob` must be higher or equal to 0, or lower than 1.")

    def _scale_input(
        self,
        f: float,
        coeff: float,
        min_intensity: float,
        max_intensity: float,
    ) -> float:
        """Internal method for scaling input.

        Args:
            f (float): input value.
            coeff (float): coefficient for scaling the input.
            min_intensity (float) : minimum value of the input intensities.
            max_intensity (float): maximum value of the input intensities.

        Returns:
            float: scaled input value.
        """
        return coeff * (f - min_intensity) / (max_intensity - min_intensity)

    def _build_trials_seq(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> NDArray:
        """Internal method for generating a sequence of modalities.

        Args:
            batch_size (int): number of trials to generate.
            rng (np.random.Generator): random number generator.

        Returns:
            NDarray[str]: list of modalities.
        """
        # Extract keys and probabilities from the dictionary
        scenarios = list(self.session_in.keys())
        probabilities = np.array(list(self.session_in.values()))
        # Normalize probabilities to ensure they sum to 1
        probabilities /= probabilities.sum()
        # Generate random numbers of samples based on the probabilities
        prob_samples = np.random.Generator.multinomial(batch_size, probabilities)
        # Create a dictionary to store the results
        session_in_samples = {
            scenario: np.random.Generator.multinomial(prob_samples[i], [1 - self.catch_prob, self.catch_prob])
            for i, scenario in enumerate(scenarios)
        }
        # Generate the sequence of modalities
        modality_seq = []
        for m in scenarios:
            temp_seq = session_in_samples[m][0] * [m] + session_in_samples[m][1] * ["catch"]
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
        return np.array(modality_seq)

    def _build_trials_inputs(
        self,
        batch_size: int,
        modality_seq: list[str],
        rng: np.random.Generator,
        phases: dict[str, np.ndarray],
    ) -> NDArray:
        """Internal method for generating inputs.

        Args:
            batch_size (int): number of trials to generate.
            modality_seq (list[str]): list of modalities.
            rng (np.random.Generator): random number generator.
            phases (dict[str, np.ndarray]): dictionary of phases of the trials.

        Returns:
            np.ndarray: array of inputs.
        """
        x = np.zeros((batch_size, len(self.t), self.n_in), dtype=np.float32)
        sel_value_in = np.full((batch_size, self.n_in - 1), self.value_in[0], dtype=np.float32)

        for n in range(batch_size):
            for m, idx in self.modality_idx.items():
                if (modality_seq[n] != "catch") and (m in modality_seq[n]):
                    sel_value_in[n, idx] = rng.choice(self.value_in[1:], 1)
                if self.scaling:
                    sel_value_in[n, idx] = self._scale_input(sel_value_in[n, idx], self.coeff, self.imin, self.imax)
                x[n, phases["input"], idx] = sel_value_in[n, idx]
                x[n, phases["t_fixation"], idx] = self.value_fixation
            x[n, phases["input"], len(self.modality_idx)] = 1  # start cue

        # Store intensities in trials
        self.trials["sel_value_in"] = sel_value_in

        # Add noise to inputs
        alpha = self.dt / self.tau
        inp_noise = 1 / alpha * np.sqrt(2 * alpha) * self.std_inp_noise * rng.normal(loc=0, scale=1, size=x.shape)

        return x + self.baseline_inp + inp_noise

    def _build_trials_outputs(
        self,
        batch_size: int,
        phases: dict[str, np.ndarray],
        choice: np.ndarray,
    ) -> NDArray:
        """Internal method for generating outputs.

        Args:
            batch_size (int): number of trials to generate.
            phases (dict[str, np.ndarray]): dictionary of phases of the trials.
            choice (np.ndarray): array of choices.

        Returns:
            np.ndarray: array of outputs.
        """
        y = np.zeros((batch_size, len(self.t), self.n_out), dtype=np.float32)
        for i in range(batch_size):
            if self.inter_trial is not None:
                y[i, phases["inter_trial"], :] = self.low_out
            if self.t_fixation is not None:
                y[i, phases["t_fixation"], :] = self.low_out

            y[i, phases["input"], choice[i]] = self.high_out
            y[i, phases["input"], 1 - choice[i]] = self.low_out

        return y

    def generate_trials(
        self,
        batch_size: int = 20,
        numpy_seed: int | None = None,
    ) -> None:
        """Method for generating trials. It populates the `trials` attribute.

        Args:
            batch_size (int, optional): number of trials to generate. Defaults to 20.
            numpy_seed (int, optional): seed for numpy random number generator. Defaults to None.
        """
        # Set the seed for reproducibility
        if numpy_seed is None:
            numpy_seed = random.randrange(2**32 - 1)
        self.trials["numpy_seed"] = numpy_seed
        rng = np.random.default_rng(numpy_seed)
        np.random.seed(numpy_seed)  # noqa: NPY002

        # Generate sequence of modalities
        modality_seq = self._build_trials_seq(batch_size, rng)

        # Setup phases of trial
        phases = {}
        phases["inter_trial"] = np.where(self.t <= self.inter_trial)[0]
        phases["t_fixation"] = np.where((self.t > self.inter_trial) & (self.t <= self.inter_trial + self.t_fixation))[0]
        phases["input"] = np.where(self.t > self.inter_trial + self.t_fixation)[0]
        choice = (modality_seq != "catch").astype(np.int_)

        # Trial Info
        self.trials["modality_seq"] = modality_seq
        self.trials["choice"] = choice
        self.trials["phases"] = phases
        self.trials["t"] = self.t
        self.trials["value_fixation"] = self.value_fixation

        # Generate and store inputs
        self.trials["inputs"] = self._build_trials_inputs(batch_size, modality_seq, rng, phases)
        # Generate and store outputs
        self.trials["outputs"] = self._build_trials_outputs(batch_size, phases, choice)

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
                *tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / len(self.modality_idx), 1.0, 1.0))
            )
            for i in range(len(self.modality_idx))
        ]
        for i in range(n):
            for m, idx in self.modality_idx.items():
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
                    y=self.trials["inputs"][i][:, len(self.modality_idx)],
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
