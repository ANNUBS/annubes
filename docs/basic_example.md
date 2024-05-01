# Basic example

## Generate synthetic data

The `Task` data class can be used for defining a behavioral task, and many parameters can be set. The configuration of the trials that can appear during a session is given by a dictionary representing the ratio of the different trials within the task (`session`). Trials with a single modality (e.g., a visual trial) must be represented by single characters, while trials with multiple modalities (e.g., an audiovisual trial) are represented by the character combination of those trials. The probability of catch trials (denoted by X) in the session can be set using the `catch_prob` parameter.

```python
from annubes.task import Task

task = Task(name='example_task',
                session={"v":0.5, "a":0.5},
                stim_intensities=[0.7, 0.9],
                stim_time=2000,
                catch_prob=0.3)
```

For more details about the `Task` class parameters, see the [API Documentation](https://annubs.github.io/annubes/latest/api/task/#annubes.task.Task).

Then, trials can be generated:

```python

NTRIALS = 10
trials = task.generate_trials(NTRIALS)
```

And plotted:

```python
task.plot_trials(NTRIALS)
```

<p align="center">
  <img src="https://github.com/ANNUBS/annubes/blob/ead1437b7ee6ad6998ce2b3653fd0b3b3d875e25/docs/example_trials_plot.png?raw=true" width="700">
</p>

## Train neural networks

This functionality is still under development.
