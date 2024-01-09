import itertools
import random
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Task:
    """General class for defining a task."""
    def __init__(self,
                 name: str,
                 modalities: list[str] = ['v', 'a', 'catch'],
                 intensity: list[float | int] = [.8, .9, 1],
                 stimulus: int = 5000,
                 fixation: int | None = 100,
                 tau: int = 100,
                 std_inp_noise: float | int =  0.01, # standard deviation for input noise
                 baseline_inp: float | int = 0.2, # baseline input for all neurons
                 high_output: float | int = 1.5, # desired network outputs
                 low_output: float | int = 0.2
                 ) -> None:
        """Constructor method for Task class.

        Args:
            name (str): _description_
            modalities (list[str], optional): _description_. Defaults to ['v', 'a', 'catch'].
                'catch' refers to giving only gaussian noise to the animal, no stimulus at all.
            intensity (list[float | int], optional): _description_. Defaults to [.8, .9, 1].
            stimulus (int, optional): _description_. Defaults to 5000.
            fixation (int | None, optional): _description_. Defaults to 100.
            tau (int, optional): _description_. Defaults to 100.
            std_inp_noise (float | int, optional): _description_. Defaults to 0.01.
            baseline_inp (float | int, optional): _description_. Defaults to 0.2.
            high_output (float | int, optional): _description_. Defaults to 1.5.
            low_output (float | int, optional): _description_. Defaults to 0.2.

        Attributes:
            name (str): _description_
            modalities (list[str]): _description_
            intensity (list[float | int]): _description_
            stimulus (int): _description_
            fixation (int | None): _description_
            tau (int): _description_
            std_inp_noise (float | int): _description_
            baseline_inp (float | int): _description_
            high_output (float | int): _description_
            low_output (float | int): _description_
            imin (float | int): _description_
            imax (float | int): _description_
            T (float | int): _description_
            Nin (int): _description_
            Nout (int): _description_
            trials (dict): _description_
        """
        # Attributes from the class constructor parameters
        self.name = name
        self.modalities = modalities
        self.intensity = intensity
        self.stimulus = stimulus
        self.fixation = fixation
        self.tau = tau
        self.std_inp_noise = std_inp_noise
        self.baseline_inp = baseline_inp
        self.high_output = high_output
        self.low_output = low_output
        # Other attributes
        self.intensity.sort()
        self.imin = self.intensity[0]
        self.imax = self.intensity[-1]
        self.T = self.fixation + self.stimulus
        self.Nin = 3  # number of inputs
        self.Nout = 2  # number of outputs
        self.trials = {}
        self.trials['name'] = self.name

    def scale_input(self, f):
        """Method for scaling input.

        Args:
            f (_type_): _description_

        Returns:
            _type_: _description_
        """
        return .6*(f - self.imin) / (self.imax - self.imin)

    def generate_trials(self,
                        batch_size: int = 20,
                        scaling: bool = False,
                        dt: int = 5,
                        catch_prob: float = None,
                        numpy_seed: int = None):
        """Method for generating synthetic trials.

        Args:
            batch_size (int, optional): _description_. Defaults to 20.
            scaling (bool, optional): _description_. Defaults to False.
            dt (int, optional): _description_. Defaults to 5.
            catch_prob (float, optional): _description_. Defaults to None.
            numpy_seed (int, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        self.dt = self.tau / dt
        t = np.linspace(self.dt, self.T, int(self.T / self.dt))

        if numpy_seed is None:
            numpy_seed = random.randrange(sys.maxsize)
        self.trials['numpy_seed'] = numpy_seed
        rng = np.random.default_rng(numpy_seed)

        # -------------------------------------------------------------------------------------
        # Select task condition
        # -------------------------------------------------------------------------------------

        if catch_prob is not None:
            if 'catch' in self.modalities:
                other_mod = len(self.modalities) - 1  # number of other modalities (excluding 'catch')
            else:
                raise ValueError('Cannot have "catch" in modalities when using catch_prob')
            non_catch_prob = (1 - catch_prob) / other_mod  # prob for generating trials in each of the other modalities
            n_other_trials = int(non_catch_prob * batch_size)  # number of trials in other modalities
            non_catch_trials = list(itertools.chain.from_iterable([[m] * n_other_trials for m in self.modalities[:-1]]))
            modality = np.array(non_catch_trials + (batch_size - len(non_catch_trials)) * ['catch'])
            rng.shuffle(modality)
        else:
            modality = rng.choice(self.modalities, batch_size)
        # note that 'batch_size' intensity values have been sampled but they may not be used (see below)
        intensity = rng.choice(self.intensity, (batch_size, 2))

        # -------------------------------------------------------------------------------------
        # Setup phases of trial
        # -------------------------------------------------------------------------------------

        phases = {}
        phases['fixation'] = np.where(t <= self.fixation)[0]
        phases['stimulus'] = np.where(t > self.fixation)[0]

        # -------------------------------------------------------------------------------------
        # Trial Info
        # -------------------------------------------------------------------------------------

        choice = (modality != 'catch').astype(np.int_)

        self.trials['modality'] = modality
        self.trials['choice'] = choice
        self.trials['phases'] = phases
        self.trials['t'] = t

        # -------------------------------------------------------------------------------------
        # Inputs
        # -------------------------------------------------------------------------------------

        x = np.zeros((batch_size, len(t), self.Nin), dtype=np.float32)
        for i in range(batch_size):
            # set visual to minimum intensity in auditory and catch trials
            if (modality[i] == 'catch') or (modality[i] == 'a'):
                intensity[i, 0] = self.imin

            # set auditory to minimum intensity in visual and catch trials
            if (modality[i] == 'catch') or (modality[i] == 'v'):
                intensity[i, 1] = self.imin

            # intensity shouldn't be imin for non-catch trials
            if modality[i] != 'catch':
                if ('v' in modality[i]) and (intensity[i, 0] == self.imin):
                    intensity[i, 0] = rng.choice(self.intensity[1:], 1)

                if ('a' in modality[i]) and (intensity[i, 1] == self.imin):
                    intensity[i, 1] = rng.choice(self.intensity[1:], 1)

            if scaling:
                # scale input
                intensity[i, 0] = self.scale_input(intensity[i, 0])
                intensity[i, 1] = self.scale_input(intensity[i, 1])

            # input for all trials
            x[i, phases['stimulus'], 0] = intensity[i, 0] # visual
            x[i, phases['stimulus'], 1] = intensity[i, 1] # auditory
            x[i, phases['stimulus'], 2] = 1 # start cue

        # store intensities in trial
        self.trials['intensity'] = intensity

        # add noise to inputs
        alpha = dt/self.tau

        inp_noise = 1/alpha * np.sqrt(2 * alpha) * self.std_inp_noise * rng.normal(loc=0, scale=1, size=x.shape)

        self.trials['inputs'] = x + self.baseline_inp + inp_noise

        # -------------------------------------------------------------------------------------
        # target output
        # -------------------------------------------------------------------------------------

        y = np.zeros((batch_size, len(t), self.Nout), dtype=np.float32)
        for i in range(batch_size):

            if self.fixation is not None:
                y[i, phases['fixation'], :] = self.low_output

            y[i, phases['stimulus'], choice[i]] = self.high_output
            y[i, phases['stimulus'], 1 - choice[i]] = self.low_output

        self.trials['outputs'] = y

        return self.trials


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
                                "Trial " + str(i + 1) + " - modality " + str(self.trials['modality'][i])
                                for i in range(n)])
        showlegend = True
        for i in range(n):
            fig.add_trace(go.Scatter(
                name="VISUAL",
                mode="markers+lines", x=self.trials['t'], y=self.trials['inputs'][i][:,0],
                marker_symbol="star",
                legendgroup="VISUAL",
                showlegend=showlegend,
                line_color = 'blue'
            ), row=i+1, col=1)
            fig.add_trace(go.Scatter(
                name="AUDITORY",
                mode="markers+lines", x=self.trials['t'], y=self.trials['inputs'][i][:,1],
                marker_symbol="star",
                legendgroup="AUDITORY",
                showlegend=showlegend,
                line_color = 'black'
            ), row=i+1, col=1)
            fig.add_trace(go.Scatter(
                name="START",
                mode="markers+lines", x=self.trials['t'], y=self.trials['inputs'][i][:,2],
                marker_symbol="star",
                legendgroup="START",
                showlegend=showlegend,
                line_color = 'green'
            ), row=i+1, col=1)
            fig.add_trace(go.Scatter(
                name="NO STIMULUS",
                mode="lines", x=self.trials['t'], y=self.trials['outputs'][i][:,0],
                legendgroup="OUTPUT 1",
                showlegend=showlegend,
                line_color = 'orange'
            ), row=i+1, col=1)
            fig.add_trace(go.Scatter(
                name="STIMULUS/STIMULI",
                mode="lines", x=self.trials['t'], y=self.trials['outputs'][i][:,1],
                legendgroup="OUTPUT 2",
                showlegend=showlegend,
                line_color = 'purple'
            ), row=i+1, col=1)
            fig.add_vline(
                x=self.fixation + self.dt, line_width=3, line_dash="dash", line_color="red")
            showlegend = False
            fig.update_yaxes(range=[0, 2], row=i+1, col=1)
        fig.update_layout(height=1300, width=900, title_text="Trials")
        return fig
