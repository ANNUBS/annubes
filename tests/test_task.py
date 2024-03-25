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
def test_post_init_session(
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
def test_post_init_catch_prob(catch_prob: float | None, expected: float | None):
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
    assert len(task._modality_seq) == task._ntrials
    task._modalities.add("catch")
    counts = {modality: np.sum(task._modality_seq == modality) for modality in task._modalities}
    # Assert that the counts match the expected distribution within a certain tolerance
    assert np.isclose(counts["catch"] / len(task._modality_seq), task.catch_prob, atol=0.1)  # within 10% tolerance
    assert np.isclose(
        counts["v"] / len(task._modality_seq),
        task.session["v"] - task.catch_prob * task.session["v"],
        atol=0.1,
    )  # within 10% tolerance
    assert np.isclose(
        counts["a"] / len(task._modality_seq),
        task.session["a"] - task.catch_prob * task.session["a"],
        atol=0.1,
    )
    assert np.isclose(
        (counts["a"] + counts["v"] + counts["catch"]) / len(task._modality_seq),
        1,
        atol=0.05,
    )


def test_build_trials_seq_shuffling():
    task_shuffled = Task(NAME, shuffle_trials=True)
    task_not_shuffled = Task(NAME, shuffle_trials=False)

    _ = task_shuffled.generate_trials(ntrials=NTRIALS)
    _ = task_not_shuffled.generate_trials(ntrials=NTRIALS)

    # Verify that the generated sequences are shuffled or not shuffled accordingly
    assert task_shuffled._modality_seq.shape == task_not_shuffled._modality_seq.shape
    assert not np.array_equal(task_shuffled._modality_seq, task_not_shuffled._modality_seq)


def test_build_trials_seq_maximum_sequential_trials():
    # Create a Task instance with shuffling enabled and a maximum sequential trial constraint
    task = Task(name=NAME, max_sequential=4)
    _ = task.generate_trials(ntrials=NTRIALS)
    # Ensure that no more than the specified maximum number of consecutive trials of the same modality occur
    for modality in task._modalities:
        for i in range(len(task._modality_seq) - task.max_sequential):
            assert np.sum(task._modality_seq[i : i + task.max_sequential] == modality) <= task.max_sequential


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
        assert min(task._iti) == min(iti)
        assert max(task._iti) == max(iti)
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
    # Check that the inputs are scaled between 0 and 1
    assert all(task._inputs[n_trial].min() >= 0 and task._inputs[n_trial].max() <= 1 for n_trial in trial_indices)
    # Check that the outputs are scaled between 0 and 1
    assert all(task._outputs[n_trial].min() >= 0 and task._outputs[n_trial].max() <= 1 for n_trial in trial_indices)


def test_generate_trials(task: Task):
    trials = task.generate_trials(ntrials=NTRIALS)
    trial_indices = range(NTRIALS)
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
