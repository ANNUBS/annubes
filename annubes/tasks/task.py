import itertools
import numpy as np


class Task:
    def __init__(self,
                 modalities: list[str] = ['v', 'a', 'catch'],
                 intensity: list[float | int] = [.8, .9, 1],
                 stimulus: int = 5000,
                 fixation: int | None = 100,
                 tau: int = 100,
                 std_inp_noise: float | int =  0.01, # standard deviation for input noise
                 baseline_inp: float | int = 0.2, # baseline input for all neurons
                 high_output: float | int = 1.5, # desired network outputs
                 low_output: float | int = 0.2
                 ):
        """General class for defining a task.

        Args:
            modalities (list[str], optional): _description_. Defaults to ['v', 'a', 'catch'].
            intensity (list[float | int], optional): _description_. Defaults to [.8, .9, 1].
            stimulus (int, optional): _description_. Defaults to 5000.
            fixation (int | None, optional): _description_. Defaults to 100.
            tau (int, optional): _description_. Defaults to 100.
            std_inp_noise (float | int, optional): _description_. Defaults to 0.01.
            baseline_inp (float | int, optional): _description_. Defaults to 0.2.
            high_output (float | int, optional): _description_. Defaults to 1.5.
            low_output (float | int, optional): _description_. Defaults to 0.2.
        """
        # 'catch' refers to giving only gaussian noise to the animal, no stimulus at all

        # Attributes from the class constructor parameters
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
        self.imin = self.intensity[0]
        self.imax = self.intensity[-1]
        self.T = self.fixation + self.stimulus
        self.Nin = 3  # number of inputs
        self.Nout = 2  # number of outputs
        # Input labels
        self.VISUAL = 0  # visual input greater than boundary frequency
        self.AUDITORY = 1  # auditory input greater than boundary frequency
        self.START = 2  # start cue


    def generate_trials(self,
                        rng,
                        dt,
                        minibatch_size,
                        catch_prob=None,
                        numpy_seed=None):
        # -------------------------------------------------------------------------------------
        # Select task condition
        # -------------------------------------------------------------------------------------

        if catch_prob is not None:
            non_catch_prob = (1 - catch_prob) / 1  # probability for generating trials in each of the other modalities
            n_other_trials = int(non_catch_prob * minibatch_size)  # number of trials in other modalities
            non_catch_trials = list(itertools.chain.from_iterable([[m] * n_other_trials for m in self.modalities[:-1]]))
            modality = np.array(non_catch_trials + (minibatch_size - len(non_catch_trials)) * ['catch'])
            rng.shuffle(modality)
        else:
            modality = rng.choice(self.modalities, minibatch_size)
        # note that 'minibatch_size' intensity values have been sampled but they may not be used (see below)
        intensity = rng.choice(self.intensity, (minibatch_size, 2))

        # -------------------------------------------------------------------------------------
        # Setup phases of trial
        # -------------------------------------------------------------------------------------

        t = np.linspace(dt, self.T, int(self.T / dt))
        phases = {}
        phases['fixation'] = np.where(t <= self.fixation)[0]
        phases['stimulus'] = np.where(t > self.fixation)[0]

        # -------------------------------------------------------------------------------------
        # Trial Info
        # -------------------------------------------------------------------------------------

        choice = (modality != 'catch').astype(np.int_)

        trials = {}
        trials['modality'] = modality
        trials['choice'] = choice
        trials['phases'] = phases

        # -------------------------------------------------------------------------------------
        # Inputs
        # -------------------------------------------------------------------------------------

        x = np.zeros((minibatch_size, len(t), self.Nin), dtype=np.float32)
        for i in range(minibatch_size):
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

            # input for all trials
            x[i, phases['stimulus'], self.VISUAL] = intensity[i, 0]
            x[i, phases['stimulus'], self.AUDITORY] = intensity[i, 1]
            x[i, phases['stimulus'], self.START] = 1

        # store intensities in trial
        trials['intensity'] = intensity

        # add noise to inputs
        alpha = dt/self.tau

        inp_noise = 1/alpha * np.sqrt(2 * alpha) * self.std_inp_noise * rng.normal(loc=0, scale=1, size=x.shape)

        trials['inputs'] = x + self.baseline_inp + inp_noise

        # -------------------------------------------------------------------------------------
        # target output
        # -------------------------------------------------------------------------------------

        y = np.zeros((minibatch_size, len(t), self.Nout), dtype=np.float32)
        for i in range(minibatch_size):

            if self.fixation is not None:
                y[i, phases['fixation'], :] = self.low_output

            y[i, phases['stimulus'], choice[i]] = self.high_output
            y[i, phases['stimulus'], 1 - choice[i]] = self.low_output

        trials['outputs'] = y

        return trials
