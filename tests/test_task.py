from collections import OrderedDict
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from annubes.task import Task

NAME = "test_task"
NTRIALS = 100
RND_SEED = 100


@pytest.fixture()
def task():
    return Task(
        name=NAME,
    )


@pytest.mark.parametrize(
    "name",
    [
        NAME,
        pytest.param(5, marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_post_init_check_str(name: Any):
    Task(name=name)


@pytest.mark.parametrize(
    "session",
    [
        {"v": 0.5, "a": 0.5},
        pytest.param(5, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param({1: 0.5, "a": 0.5}, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param({"v": "a", "a": 0.5}, marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_post_init_check_session(session: Any):
    Task(name=NAME, session=session)


@pytest.mark.parametrize(
    ("stim_intensities", "catch_prob", "fix_intensity", "output_behavior", "noise_std"),
    [
        ([0.8, 0.9, 1], 0.5, 0, [0, 1], 0.01),
        pytest.param([0.8, "a"], 0.5, 0, [0, 1], 0.01, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param([0.8, -1], 0.5, 0, [0, 1], 0.01, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param([0.8, 0.9, 1], "a", 0, [0, 1], 0.01, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param([0.8, 0.9, 1], 5, 0, [0, 1], 0.01, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param([0.8, 0.9, 1], 0.5, "a", [0, 1], 0.01, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param([0.8, 0.9, 1], 0.5, -1, [0, 1], 0.01, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param([0.8, 0.9, 1], 0.5, 0, ["0", 1], 0.01, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param([0.8, 0.9, 1], 0.5, 0, [-1, 1], 0.01, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param([0.8, 0.9, 1], 0.5, 0, [0, 1], "a", marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param([0.8, 0.9, 1], 0.5, 0, [0, 1], -1, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_post_init_check_float_positive(
    stim_intensities: Any,
    catch_prob: Any,
    fix_intensity: Any,
    output_behavior: Any,
    noise_std: Any,
):
    Task(
        name=NAME,
        stim_intensities=stim_intensities,
        catch_prob=catch_prob,
        fix_intensity=fix_intensity,
        output_behavior=output_behavior,
        noise_std=noise_std,
    )


@pytest.mark.parametrize(
    ("stim_time", "dt", "tau", "fix_time", "iti"),
    [
        (1000, 20, 100, 100, 0),
        pytest.param("a", 20, 100, 100, 0, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(0, 20, 100, 100, 0, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(1000, "a", 100, 100, 0, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(1000, 0, 100, 100, 0, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(1000, 20, "a", 100, 0, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(1000, 20, 0, 100, 0, marks=pytest.mark.xfail(raises=ValueError)),
        (1000, 20, 100, 0, 0),
        pytest.param(1000, 20, 100, "a", 0, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(1000, 20, 100, -1, 0, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(1000, 20, 100, 100, "a", marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(1000, 20, 100, 100, -1, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(1000, 20, 100, 100, ("a", 0), marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(1000, 20, 100, 100, (-1, 0), marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_post_init_check_time_vars(
    stim_time: Any,
    dt: Any,
    tau: Any,
    fix_time: Any,
    iti: Any,
):
    Task(
        name=NAME,
        stim_time=stim_time,
        dt=dt,
        tau=tau,
        fix_time=fix_time,
        iti=iti,
    )


@pytest.mark.parametrize(
    ("max_sequential", "n_outputs"),
    [
        (None, 2),
        pytest.param("a", 2, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(0, 2, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(None, "a", marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(None, 0, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_post_init_check_other_int_positive(
    max_sequential: Any,
    n_outputs: Any,
):
    Task(
        name=NAME,
        max_sequential=max_sequential,
        n_outputs=n_outputs,
    )


@pytest.mark.parametrize(
    ("shuffle_trials", "scaling"),
    [
        (True, True),
        pytest.param("a", True, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(False, "a", marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_post_init_check_bool(
    shuffle_trials: Any,
    scaling: Any,
):
    Task(
        name=NAME,
        shuffle_trials=shuffle_trials,
        scaling=scaling,
    )


@pytest.mark.parametrize(
    ("session", "shuffle_trials", "expected_dict", "expected_type"),
    [
        ({"v": 0.5, "a": 0.5}, True, {"v": 0.5, "a": 0.5}, dict),
        ({"v": 1, "a": 3}, True, {"v": 0.25, "a": 0.75}, dict),
        ({"v": 0, "a": 100}, True, {"v": 0, "a": 1}, dict),
        ({"v": 1, "va": 3, "a": 6}, True, {"v": 0.1, "va": 0.3, "a": 0.6}, dict),
        ({"v": 1, "va": 3, "a": 6}, False, {"v": 0.1, "va": 0.3, "a": 0.6}, OrderedDict),
    ],
)
def test_post_init_session(
    session: dict,
    shuffle_trials: bool,
    expected_dict: dict | OrderedDict,
    expected_type: dict | OrderedDict,
):
    task = Task(NAME, session=session, shuffle_trials=shuffle_trials)
    assert task.name == NAME
    assert task._session == expected_dict
    assert isinstance(task._session, expected_type)  # type: ignore # noqa: PGH003


@pytest.mark.parametrize(
    ("catch_prob", "expected"),
    [
        pytest.param(-1, -1, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(2, 2, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(None, None, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param("0.6", "0.6", marks=pytest.mark.xfail(raises=TypeError)),
        (0.6, 0.6),
    ],
)
def test_post_init_catch_prob(catch_prob: float | None, expected: float | None):
    task = Task(NAME, catch_prob=catch_prob)  # type: ignore # noqa: PGH003
    assert task.name == NAME
    assert task.catch_prob == expected


@pytest.mark.parametrize(
    ("dt", "tau"),
    [
        pytest.param(20, 30),
        pytest.param(0, 0, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(-1, -1, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_post_init_noise_related(
    dt: int,
    tau: int,
):
    task = Task(NAME, dt=dt, tau=tau)
    assert task.dt == dt
    assert task.tau == tau


@pytest.mark.parametrize(
    ("session", "expected_modalities"),
    [
        ({"v": 0.5, "a": 0.5}, {"v", "a"}),
        ({"v": 0.5, "a": 0.5}, {"a", "v"}),
        ({"v": 0.4, "av": 0.1, "a": 0.5}, {"v", "a"}),
        ({"v": 0.4, "av": 0.1, "m": 0.1, "a": 0.4}, {"v", "a", "m"}),
    ],
)
def test_post_init_derived_attributes(
    session: dict,
    expected_modalities: set[str],
):
    task = Task(NAME, session=session)
    assert task._modalities == expected_modalities
    assert task._n_inputs == len(expected_modalities) + 1  # add the start signal


@pytest.mark.parametrize(
    ("session", "catch_prob"),
    [
        ({"v": 0.5, "a": 0.5}, 0.5),
        ({"v": 0.8, "a": 0.2}, 0.1),
        ({"v": 0.5, "a": 0.5}, 0),
        pytest.param({"v": 0.5, "a": 0.5}, -1, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_build_trials_seq_distributions(session: dict, catch_prob: float):
    task = Task(NAME, session=session, catch_prob=catch_prob)
    _ = task.generate_trials(ntrials=NTRIALS)
    assert isinstance(task._modality_seq, np.ndarray)

    # Assert that the counts match the expected distribution within a certain tolerance
    ratios = {
        modality: np.sum(task._modality_seq == modality) / len(task._modality_seq)
        for modality in [*task._modalities, "catch"]
    }
    expected_ratios = {
        "v": task.session["v"] - task.catch_prob * task.session["v"],
        "a": task.session["a"] - task.catch_prob * task.session["a"],
        "catch": task.catch_prob,
    }
    tolerance = 0.2

    for modality, actual_ratio in ratios.items():
        assert np.isclose(
            actual_ratio,
            expected_ratios[modality],
            atol=tolerance,
        ), f"Actual difference for {modality}: {round(abs(actual_ratio - expected_ratios[modality]), 3)}"

    # Assert that total counts match the expected number exactly
    assert np.isclose(ratios["a"] + ratios["v"] + ratios["catch"], 1.0)  # avoid floating point errors
    assert len(task._modality_seq) == task._ntrials


def test_build_trials_seq_shuffling():
    task_shuffled = Task(NAME, shuffle_trials=True)
    task_not_shuffled = Task(NAME, shuffle_trials=False)

    _ = task_shuffled.generate_trials(ntrials=NTRIALS)
    _ = task_not_shuffled.generate_trials(ntrials=NTRIALS)

    assert task_shuffled._modality_seq.shape == task_not_shuffled._modality_seq.shape

    # check that shuffled and unshuffled are different
    assert not np.array_equal(task_shuffled._modality_seq, task_not_shuffled._modality_seq)

    # check that there is no "v" after any "a", which would mean that the list is shuffled
    keys = list(task_shuffled._session.keys())
    first_occurrence = list(task_not_shuffled._modality_seq).index(keys[1])
    assert keys[0] not in task_not_shuffled._modality_seq[first_occurrence:]


def test_build_trials_seq_maximum_sequential_trials():
    # Create a Task instance with shuffling enabled and a maximum sequential trial constraint
    task = Task(name=NAME, max_sequential=4)
    _ = task.generate_trials(ntrials=NTRIALS)

    # Ensure that no more than the specified maximum number of consecutive trials of the same modality occur
    sequence_string = "".join(task._modality_seq).replace("catch", "X")
    too_many = task.max_sequential + 1
    for mod in set(sequence_string):
        assert (
            mod * (too_many) not in sequence_string
        ), f'{mod.replace("X", "catch")} was detected too many times (seed: {task._random_seed})'


@pytest.mark.parametrize(
    ("stim_time", "fix_time", "iti"),
    [(1000, 100, 0), (1000, 100, (300, 500))],
)
def test_setup_trial_phases(stim_time: int, fix_time: int, iti: int | tuple[int, int]):
    task = Task(NAME, stim_time=stim_time, fix_time=fix_time, iti=iti)
    _ = task.generate_trials(ntrials=NTRIALS)
    trial_indices = range(NTRIALS)
    # iti
    assert task._iti.shape == (NTRIALS,)
    if type(iti) is tuple:
        assert min(task._iti) >= min(iti)
        assert max(task._iti) >= max(iti)
    else:
        assert all(task._iti == iti)
    # time
    assert task._time.shape == (NTRIALS,)
    assert all(
        len(task._time[n_trial]) == int(stim_time + fix_time + task._iti[n_trial] + task.dt) / task.dt
        for n_trial in trial_indices
    )
    assert all(max(task._time[n_trial]) == stim_time + fix_time + task._iti[n_trial] for n_trial in trial_indices)
    # phases
    assert task._phases.shape == (NTRIALS,)
    assert all(
        len(task._phases[n_trial]["iti"]) == len(np.where(task._time[n_trial] <= task._iti[n_trial])[0])
        for n_trial in trial_indices
    )
    assert all(
        len(task._phases[n_trial]["fix_time"])
        == len(
            np.where(
                (task._time[n_trial] > task._iti[n_trial]) & (task._time[n_trial] <= task._iti[n_trial] + fix_time),
            )[0],
        )
        for n_trial in trial_indices
    )
    assert all(
        len(task._phases[n_trial]["input"]) == len(np.where(task._time[n_trial] > task._iti[n_trial] + fix_time)[0])
        for n_trial in trial_indices
    )


def test_minmaxscaler():
    task = Task(name=NAME, scaling=True)
    _ = task.generate_trials(ntrials=NTRIALS)
    trial_indices = range(NTRIALS)
    # Check that the signals are scaled between 0 and 1, and that min is 0 and max is 1
    ## Inputs
    assert all((task._inputs[n_trial] >= 0).all() and (task._inputs[n_trial] <= 1).all() for n_trial in trial_indices)
    assert all(task._inputs[n_trial].min() == 0 and task._inputs[n_trial].max() == 1 for n_trial in trial_indices)
    # Outputs
    assert all((task._outputs[n_trial] >= 0).all() and (task._outputs[n_trial] <= 1).all() for n_trial in trial_indices)
    assert all(task._outputs[n_trial].min() == 0 and task._outputs[n_trial].max() == 1 for n_trial in trial_indices)


@pytest.mark.parametrize(
    ("ntrials", "random_seed"),
    [
        (20, None),
        pytest.param("a", None, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(0, None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(0.5, None, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param((30, 40, 50), None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(("40", 50), None, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param((40, "50"), None, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(20, "a", marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(20, -1, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(20, 0.5, marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_generate_trials_check(
    task: Task,
    ntrials: Any,
    random_seed: Any,
):
    _ = task.generate_trials(ntrials=ntrials, random_seed=random_seed)


@pytest.mark.parametrize(
    "ntrials",
    [NTRIALS, (100, 200)],
)
def test_generate_trials(task: Task, ntrials: int | tuple[int, int]):
    trials = task.generate_trials(ntrials=ntrials)
    if type(ntrials) is tuple:
        assert task._ntrials >= ntrials[0]
        assert task._ntrials <= ntrials[1]
    trial_indices = range(task._ntrials)
    assert all(trials[key].shape == (task._ntrials,) for key in ["modality_seq", "time", "phases", "inputs", "outputs"])
    assert all(
        trials["inputs"][n_trial].shape == (len(task._time[n_trial]), task._n_inputs) for n_trial in trial_indices
    )
    assert all(
        trials["outputs"][n_trial].shape == (len(task._time[n_trial]), task.n_outputs) for n_trial in trial_indices
    )


@pytest.mark.parametrize(
    ("session", "ntrials", "random_seed"),
    [
        ({"v": 0.5, "a": 0.5}, 20, None),
        ({"v": 1, "a": 3}, NTRIALS, 100),
    ],
)
def test_reproduce_experiment(
    session: dict,
    ntrials: int,
    random_seed: int | None,
):
    task = Task(name=NAME, session=session)
    trials = task.generate_trials(ntrials=ntrials, random_seed=random_seed)
    task_repro = Task(**trials["task_settings"])
    trials_repro = task_repro.generate_trials(trials["ntrials"], trials["random_seed"])
    trial_indices = range(trials["ntrials"])

    tested_outputs = []
    # Check that the output is the same
    assert task == task_repro
    for x, y in trials.items():
        if not isinstance(y, np.ndarray):
            # task_settings, ntrials, random_seed
            assert y == trials_repro[x]
            tested_outputs.append(x)
        elif type(y[0]) is np.ndarray:
            # time, inputs, outputs
            assert all(np.array_equal(trials[x][n_trial], trials_repro[x][n_trial]) for n_trial in trial_indices)
            tested_outputs.append(x)
    assert all(
        all(
            np.array_equal(trials["phases"][n_trial][key], trials_repro["phases"][n_trial][key])
            for key in trials["phases"][n_trial]
        )
        for n_trial in trial_indices
    )
    tested_outputs.append("phases")
    assert np.array_equal(trials["modality_seq"], trials_repro["modality_seq"])
    tested_outputs.append("modality_seq")
    assert set(tested_outputs) == set(trials.keys())


def test_plot_trials(task: Task):
    # Generate trial data
    ntrials = 3
    _ = task.generate_trials(ntrials=ntrials)
    # Call plot_trials
    n_plots = 2
    fig = task.plot_trials(n_plots=n_plots)
    # Assert basic properties of the plot
    assert isinstance(fig, go.Figure)
    assert len(fig["data"]) / (task._n_inputs + task.n_outputs) == n_plots  # Number of plots should match n_plots
    assert fig.layout.title.text == "Trials"  # Check title
    # Test with n_plots > ntrials
    n_plots = 5
    with pytest.warns(UserWarning) as w:
        fig = task.plot_trials(n_plots=n_plots)
    # Check if the warning was raised
    assert len(w) == 1
    warning = w[0]
    assert (
        str(warning.message)
        == f"Number of plots requested ({n_plots}) exceeds number of trials ({ntrials}). Will plot all trials."
    )
    assert warning.category == UserWarning
