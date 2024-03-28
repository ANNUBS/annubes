from collections import OrderedDict

import numpy as np
import plotly.graph_objects as go
import pytest

from annubes.task import Task

NAME = "test_task"
NTRIALS = 100


@pytest.fixture()
def task():
    return Task(
        name=NAME,
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
def test_init_session(
    session: dict,
    shuffle_trials: bool,
    expected_dict: dict | OrderedDict,
    expected_type: dict | OrderedDict,
):
    task = Task(NAME, session=session, shuffle_trials=shuffle_trials)
    assert task.name == NAME
    assert task._session == expected_dict
    assert isinstance(task._session, expected_type)


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
def test_init_catch_prob(catch_prob: float | None, expected: float | None):
    task = Task(NAME, catch_prob=catch_prob)
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
    ("session", "stim_time", "fix_time", "iti", "expected_modalities"),
    [
        ({"v": 0.5, "a": 0.5}, 0, 100, 1000, {"v", "a"}),
        ({"v": 0.5, "a": 0.5}, 0, 100, 1000, {"a", "v"}),
        ({"v": 0.4, "av": 0.1, "a": 0.5}, 30, 200, 500, {"v", "a"}),
        ({"v": 0.4, "av": 0.1, "m": 0.1, "a": 0.4}, 20, 100, 500, {"v", "a", "m"}),
    ],
)
def test_post_init_derived_attributes(
    session: dict,
    stim_time: int,
    fix_time: int,
    iti: int,
    expected_modalities: set[str],
):
    task = Task(NAME, session=session, stim_time=stim_time, fix_time=fix_time, iti=iti)
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
    task._ntrials = NTRIALS
    task._rng = np.random.default_rng(NTRIALS)
    modality_seq = task._build_trials_seq()
    assert isinstance(modality_seq, np.ndarray)
    assert len(modality_seq) == task._ntrials
    task._modalities.add("catch")
    counts = {modality: np.sum(modality_seq == modality) for modality in task._modalities}
    # Assert that the counts match the expected distribution within a certain tolerance
    assert np.isclose(counts["catch"] / len(modality_seq), task.catch_prob, atol=0.1)  # within 10% tolerance
    assert np.isclose(
        counts["v"] / len(modality_seq),
        task.session["v"] - task.catch_prob * task.session["v"],
        atol=0.1,
    )  # within 10% tolerance
    assert np.isclose(
        counts["a"] / len(modality_seq),
        task.session["a"] - task.catch_prob * task.session["a"],
        atol=0.1,
    )
    assert np.isclose(
        (counts["a"] + counts["v"] + counts["catch"]) / len(modality_seq),
        1,
        atol=0.05,
    )


def test_build_trials_seq_shuffling():
    task_shuffled = Task(NAME, shuffle_trials=True)
    task_not_shuffled = Task(NAME, shuffle_trials=False)

    task_shuffled._ntrials = NTRIALS
    task_not_shuffled._ntrials = NTRIALS

    task_shuffled._rng = np.random.default_rng(NTRIALS)
    task_not_shuffled._rng = np.random.default_rng(NTRIALS)

    sequence_shuffled = task_shuffled._build_trials_seq()
    sequence_not_shuffled = task_not_shuffled._build_trials_seq()

    # Verify that the generated sequences are shuffled or not shuffled accordingly
    assert sequence_shuffled.shape == sequence_not_shuffled.shape
    assert not np.array_equal(sequence_shuffled, sequence_not_shuffled)


def test_build_trials_seq_maximum_sequential_trials():
    # Create a Task instance with shuffling enabled and a maximum sequential trial constraint
    task = Task(name=NAME, max_sequential=4)
    task._ntrials = NTRIALS
    task._rng = np.random.default_rng(NTRIALS)
    modality_seq = task._build_trials_seq()
    # Ensure that no more than the specified maximum number of consecutive trials of the same modality occur
    for modality in task._modalities:
        for i in range(len(modality_seq) - task.max_sequential):
            assert np.sum(modality_seq[i : i + task.max_sequential] == modality) <= task.max_sequential


def test_generate_trials(task: Task):
    trials = task.generate_trials(ntrials=NTRIALS)
    trial_indices = range(NTRIALS)
    assert trials["inputs"].shape == (task._ntrials,)
    assert all(
        trials["inputs"][n_trial].shape == (len(task._time[n_trial]), task._n_inputs) for n_trial in trial_indices
    )
    assert trials["outputs"].shape == (task._ntrials,)
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

    # Check that the output is the same
    assert task == task_repro
    for x, y in trials.items():
        if type(y) is not np.ndarray:
            # task_settings, ntrials, random_seed
            assert y == trials_repro[x]
        elif type(y[0]) is np.ndarray:
            # time, inputs, outputs
            assert all(np.array_equal(trials[x][n_trial], trials_repro[x][n_trial]) for n_trial in trial_indices)
    assert all(
        all(
            np.array_equal(trials["phases"][n_trial][key], trials_repro["phases"][n_trial][key])
            for key in trials["phases"][n_trial]
        )
        for n_trial in trial_indices
    )
    assert np.array_equal(trials["modality_seq"], trials_repro["modality_seq"])


def test_plot_trials(task: Task):
    # Generate trial data
    ntrials = 3
    task.generate_trials(ntrials=ntrials)
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
