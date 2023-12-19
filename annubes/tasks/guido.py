import itertools
import numpy as np


class GuidoTask:
    def __init__(self, task_param):
        self.intensity = [9, 9.25, 9.5, 9.75, 10, 10.25, 10.5, 11, 12, 13]
        # 'catch' refers to giving only gaussian noise to the animal, no stimulus at all
        self.modalities = ['v', 'a', 'va', 'catch']

        self.imin = self.intensity[0]
        self.imax = self.intensity[-1]

        self.fixation = 100
        self.stimulus = 1000  # stimulus is present for model second
        self.T = self.fixation + self.stimulus

        self.Nin = 3  # number of inputs
        self.Nout = 2  # number of outputs
        self.baseline_inp = 0.2  # baseline input for all neurons
        self.use_fixation = task_param['fixation']
        self.tau = task_param['tau']
        self.std_inp_noise = task_param['std_inp_noise']  # standard deviation for input noise

        # desired network outputs
        self.high_output = 1.5
        self.low_output = 0.2

        # Input labels
        self.VISUAL = 0  # visual input greater than boundary frequency
        self.AUDITORY = 1  # auditory input greater than boundary frequency
        self.START = 2  # start cue

    def scale_input(self, f):
            return .6*(f - self.imin) / (self.imax - self.imin)

    def generate_trials(self, rng, dt, minibatch_size, catch_prob=None):
        # -------------------------------------------------------------------------------------
        # Select task condition
        # -------------------------------------------------------------------------------------

        if catch_prob is not None:
            # non_catch_prob = (1 - catch_prob) / 3  # probability for generating 'v', 'a' and 'va' trials
            # p = [non_catch_prob] * 3 + [catch_prob]
            # modality = rng.choice(self.modalities, minibatch_size, p)

            # numpy's choice method takes probabilities of generating each item but doesn't seem to be very accurate.
            # Below, I create a vector containing trials of different modalities in proportion to probabilities by
            # shuffling. Possible downside of this approach is that the number of trials in different modalities are
            # always same.
            non_catch_prob = (1 - catch_prob) / 3  # probability for generating trials in each of the other modalities
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
            x[i, phases['stimulus'], self.VISUAL] = self.scale_input(intensity[i, 0])
            x[i, phases['stimulus'], self.AUDITORY] = self.scale_input(intensity[i, 1])
            x[i, phases['stimulus'], self.START] = 1

        # store intensities in trial
        trials['intensity'] = intensity

        # add noise to inputs
        alpha = dt/self.tau

        inp_noise = 1/alpha * np.sqrt(2 * alpha) * self.std_inp_noise * rng.normal(loc=0, scale=1, size=x.shape) #! altered
       # inp_noise = rng.normal(loc=0, scale=self.std_inp_noise, size=x.shape)
        trials['inputs'] = x + self.baseline_inp + inp_noise

        # -------------------------------------------------------------------------------------
        # target output
        # -------------------------------------------------------------------------------------

        y = np.zeros((minibatch_size, len(t), self.Nout), dtype=np.float32)
        for i in range(minibatch_size):

            if(self.use_fixation):
                y[i, phases['fixation'], :] = self.low_output

            y[i, phases['stimulus'], choice[i]] = self.high_output
            y[i, phases['stimulus'], 1 - choice[i]] = self.low_output

        trials['outputs'] = y

        return trials

