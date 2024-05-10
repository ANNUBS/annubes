from collections import OrderedDict
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from annubes.task import Task

NAME = "test_task"
# Task's default parameters
SESSION = {"v": 0.5, "a": 0.5}
STIM_INTENSITIES = [0.8, 0.9, 1]
STIM_TIME = 1000
CATCH_PROB = 0.5
SHUFFLE_TRIALS = True
MAX_SEQUENTIAL = None
# TaskSettingsMixin's default parameters
FIX_INTENSITY = 0
FIX_TIME = 100
ITI = 0
DT = 20
TAU = 100
N_OUTPUTS = 2
OUTPUT_BEHAVIOR = [0.0, 1.0]
NOISE_STD = 0.01
SCALING = True
NTRIALS = 100
RND_SEED = None


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
        SESSION,
        pytest.param(5, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param({1: 0.5, "a": 0.5}, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param({"v": "a", "a": 0.5}, marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_post_init_check_session(session: Any):
    Task(name=NAME, session=session)


@pytest.mark.parametrize("stim_intensities", [STIM_INTENSITIES, [0.9]])
@pytest.mark.parametrize("catch_prob", [CATCH_PROB, 0])
@pytest.mark.parametrize("fix_intensity", [FIX_INTENSITY, 100])
def test_post_init_check_float_positive_valid(stim_intensities: list[float], catch_prob: float, fix_intensity: float):
    Task(
        name=NAME,
        stim_intensities=stim_intensities,
        catch_prob=catch_prob,
        fix_intensity=fix_intensity,
        output_behavior=OUTPUT_BEHAVIOR,
        noise_std=NOISE_STD,
    )


@pytest.mark.parametrize(
    ("stim_intensities", "catch_prob", "fix_intensity", "output_behavior", "noise_std", "expected_exception"),
    [
        ([0.8, "a"], CATCH_PROB, FIX_INTENSITY, OUTPUT_BEHAVIOR, NOISE_STD, TypeError),
        ([0.8, -1], CATCH_PROB, FIX_INTENSITY, OUTPUT_BEHAVIOR, NOISE_STD, ValueError),
        (STIM_INTENSITIES, "a", FIX_INTENSITY, OUTPUT_BEHAVIOR, NOISE_STD, TypeError),
        (STIM_INTENSITIES, 5, FIX_INTENSITY, OUTPUT_BEHAVIOR, NOISE_STD, ValueError),
        (STIM_INTENSITIES, CATCH_PROB, "a", OUTPUT_BEHAVIOR, NOISE_STD, TypeError),
        (STIM_INTENSITIES, CATCH_PROB, -1, OUTPUT_BEHAVIOR, NOISE_STD, ValueError),
        (STIM_INTENSITIES, CATCH_PROB, FIX_INTENSITY, ["0", 1], NOISE_STD, TypeError),
        (STIM_INTENSITIES, CATCH_PROB, FIX_INTENSITY, [-1, 1], NOISE_STD, ValueError),
        (STIM_INTENSITIES, CATCH_PROB, FIX_INTENSITY, OUTPUT_BEHAVIOR, "a", TypeError),
        (STIM_INTENSITIES, CATCH_PROB, FIX_INTENSITY, OUTPUT_BEHAVIOR, -1, ValueError),
    ],
)
def test_post_init_check_float_positive_invalid(  # noqa: PLR0913
    stim_intensities: Any,
    catch_prob: Any,
    fix_intensity: Any,
    output_behavior: Any,
    noise_std: Any,
    expected_exception: Any,
):
    with pytest.raises(expected_exception):
        Task(
            name=NAME,
            stim_intensities=stim_intensities,
            catch_prob=catch_prob,
            fix_intensity=fix_intensity,
            output_behavior=output_behavior,
            noise_std=noise_std,
        )


@pytest.mark.parametrize("fix_time", [FIX_TIME, 0, (3000, 5000)])
@pytest.mark.parametrize("iti", [ITI, 0, (3000, 5000)])
def test_post_init_check_time_vars_valid(
    fix_time: int | tuple[int, int],
    iti: int | tuple[int, int],
):
    Task(
        name=NAME,
        stim_time=STIM_TIME,
        dt=DT,
        tau=TAU,
        fix_time=fix_time,
        iti=iti,
    )


@pytest.mark.parametrize(
    ("stim_time", "dt", "tau", "fix_time", "iti", "expected_exception"),
    [
        ("a", DT, TAU, FIX_TIME, ITI, TypeError),
        (0, DT, TAU, FIX_TIME, ITI, ValueError),
        (STIM_TIME, "a", TAU, FIX_TIME, ITI, TypeError),
        (STIM_TIME, 0, TAU, FIX_TIME, ITI, ValueError),
        (STIM_TIME, DT, "a", FIX_TIME, ITI, TypeError),
        (STIM_TIME, DT, 0, FIX_TIME, ITI, ValueError),
        (STIM_TIME, DT, TAU, "a", ITI, TypeError),
        (STIM_TIME, DT, TAU, -1, ITI, ValueError),
        (STIM_TIME, DT, TAU, ("a", 0), ITI, TypeError),
        (STIM_TIME, DT, TAU, (-1, 0), ITI, ValueError),
        (STIM_TIME, DT, TAU, FIX_TIME, "a", TypeError),
        (STIM_TIME, DT, TAU, FIX_TIME, -1, ValueError),
        (STIM_TIME, DT, TAU, FIX_TIME, ("a", 0), TypeError),
        (STIM_TIME, DT, TAU, FIX_TIME, (-1, 0), ValueError),
    ],
)
def test_post_init_check_time_vars_invalid(  # noqa: PLR0913
    stim_time: Any,
    dt: Any,
    tau: Any,
    fix_time: Any,
    iti: Any,
    expected_exception: Any,
):
    with pytest.raises(expected_exception):
        Task(
            name=NAME,
            stim_time=stim_time,
            dt=dt,
            tau=tau,
            fix_time=fix_time,
            iti=iti,
        )


@pytest.mark.parametrize("max_sequential", [MAX_SEQUENTIAL, 4])
def test_post_init_check_other_int_positive_valid(
    max_sequential: int | None,
):
    Task(
        name=NAME,
        max_sequential=max_sequential,
        n_outputs=N_OUTPUTS,
    )


@pytest.mark.parametrize(
    ("max_sequential", "n_outputs", "expected_exception"),
    [
        ("a", N_OUTPUTS, TypeError),
        (0, N_OUTPUTS, ValueError),
        (MAX_SEQUENTIAL, "a", TypeError),
        (MAX_SEQUENTIAL, 0, ValueError),
    ],
)
def test_post_init_check_other_int_positive_invalid(
    max_sequential: Any,
    n_outputs: Any,
    expected_exception: Any,
):
    with pytest.raises(expected_exception):
        Task(
            name=NAME,
            max_sequential=max_sequential,
            n_outputs=n_outputs,
        )


@pytest.mark.parametrize("shuffle_trials", [True, False])
@pytest.mark.parametrize("scaling", [True, False])
def test_post_init_check_bool_valid(
    shuffle_trials: bool,
    scaling: bool,
):
    Task(
        name=NAME,
        shuffle_trials=shuffle_trials,
        scaling=scaling,
    )


@pytest.mark.parametrize(
    ("shuffle_trials", "scaling", "expected_exception"),
    [
        ("a", SCALING, TypeError),
        (SHUFFLE_TRIALS, "a", TypeError),
    ],
)
def test_post_init_check_bool_invalid(
    shuffle_trials: Any,
    scaling: Any,
    expected_exception: Any,
):
    with pytest.raises(expected_exception):
        Task(
            name=NAME,
            shuffle_trials=shuffle_trials,
            scaling=scaling,
        )


@pytest.mark.parametrize(
    ("session", "shuffle_trials", "expected_dict", "expected_type"),
    [
        (SESSION, True, {"v": 0.5, "a": 0.5}, dict),
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
    assert isinstance(task._session, expected_type)  # type: ignore[arg-type]


@pytest.mark.parametrize("catch_prob", [CATCH_PROB, 0])
def test_post_init_catch_prob_valid(catch_prob: float):
    task = Task(NAME, catch_prob=catch_prob)
    assert task.name == NAME
    assert task.catch_prob == catch_prob


@pytest.mark.parametrize(
    ("catch_prob", "expected_exception"),
    [(-1, ValueError), (2, ValueError), (None, TypeError), ("0.6", TypeError)],
)
def test_post_init_catch_prob_invalid(catch_prob: Any, expected_exception: Any):
    with pytest.raises(expected_exception):
        Task(NAME, catch_prob=catch_prob)


def test_post_init_noise_related_valid():
    task = Task(NAME, dt=DT, tau=TAU)
    assert task.dt == DT
    assert task.tau == TAU


@pytest.mark.parametrize(
    ("dt", "tau", "expected_exception"),
    [
        (0, DT, ValueError),
        (-1, DT, ValueError),
        (TAU, 0, ValueError),
        (TAU, -1, ValueError),
    ],
)
def test_post_init_noise_related_invalid(
    dt: int,
    tau: int,
    expected_exception: Any,
):
    with pytest.raises(expected_exception):
        Task(NAME, dt=dt, tau=tau)


@pytest.mark.parametrize(
    ("session", "expected_modalities"),
    [
        (SESSION, {"v", "a"}),
        (SESSION, {"a", "v"}),
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


@pytest.mark.parametrize("session", [SESSION, {"v": 0.8, "a": 0.2}])
@pytest.mark.parametrize("catch_prob", [CATCH_PROB, 0, 0.1])
def test_build_trials_seq_distributions(session: dict, catch_prob: float):
    task = Task(NAME, session=session, catch_prob=catch_prob)
    _ = task.generate_trials(ntrials=NTRIALS)
    assert isinstance(task._modality_seq, np.ndarray)

    # Assert that the counts match the expected distribution within a certain tolerance
    ratios = {
        modality: np.sum(task._modality_seq == modality) / len(task._modality_seq)
        for modality in [*task._modalities, "X"]
    }
    expected_ratios = {
        "v": task.session["v"] - task.catch_prob * task.session["v"],
        "a": task.session["a"] - task.catch_prob * task.session["a"],
        "X": task.catch_prob,
    }
    tolerance = 0.2

    for modality, actual_ratio in ratios.items():
        assert np.isclose(
            actual_ratio,
            expected_ratios[modality],
            atol=tolerance,
        ), f"Actual difference for {modality}: {round(abs(actual_ratio - expected_ratios[modality]), 3)}"

    # Assert that total counts match the expected number exactly
    assert np.isclose(ratios["a"] + ratios["v"] + ratios["X"], 1.0)  # avoid floating point errors
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


@pytest.mark.parametrize(
    ("session", "catch_prob", "max_sequential"),
    [
        ({"v": 0.5, "a": 0.5}, 0.5, 4),
        ({"v": 0.5, "a": 0.5}, 0.5, 1),
        ({"v": 0.8, "a": 0.2}, 0.1, 4),
        ({"v": 0.5, "a": 0.5}, 0.9, 4),
        ({"v": 0.5, "a": 0.5}, 0.9, 1),
        pytest.param({"v": 0.5, "a": 0.5}, 0.5, 0.5, marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_build_trials_seq_maximum_sequential_trials(session: dict[str, float], catch_prob: float, max_sequential: int):
    # Create a Task instance with shuffling enabled and a maximum sequential trial constraint
    task = Task(name=NAME, session=session, catch_prob=catch_prob, max_sequential=max_sequential)
    _ = task.generate_trials(ntrials=NTRIALS)

    # Ensure that no more than the specified maximum number of consecutive trials of the same modality occur
    sequence_string = "".join(task._modality_seq)
    too_many = task.max_sequential + 1  # type: ignore[operator]
    for mod in set(sequence_string):
        assert mod * (too_many) not in sequence_string, f"{mod} was detected too many times (seed: {task._random_seed})"


@pytest.mark.parametrize("fix_time", [100, (3000, 5000)])
@pytest.mark.parametrize("iti", [0, (300, 500)])
def test_setup_trial_phases(fix_time: int | tuple[int, int], iti: int | tuple[int, int]):
    task = Task(NAME, stim_time=STIM_TIME, fix_time=fix_time, iti=iti)
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
        len(task._time[n_trial]) == int(STIM_TIME + task._fix_time[n_trial] + task._iti[n_trial] + task.dt) / task.dt
        for n_trial in trial_indices
    )
    assert all(
        max(task._time[n_trial]) == STIM_TIME + task._fix_time[n_trial] + task._iti[n_trial]
        for n_trial in trial_indices
    )
    # phases
    assert task._phases.shape == (NTRIALS,)
    assert all(
        len(task._phases[n_trial]["fix_time"]) == len(np.where(task._time[n_trial] <= task._fix_time[n_trial])[0])
        for n_trial in trial_indices
    )
    assert all(
        len(task._phases[n_trial]["input"])
        == len(
            np.where(
                (task._time[n_trial] > task._fix_time[n_trial])
                & (task._time[n_trial] <= task._fix_time[n_trial] + STIM_TIME),
            )[0],
        )
        for n_trial in trial_indices
    )
    assert all(
        len(task._phases[n_trial]["iti"])
        == len(np.where(task._time[n_trial] >= task._fix_time[n_trial] + STIM_TIME)[0])
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


@pytest.mark.parametrize("random_seed", [RND_SEED, 100])
def test_generate_trials_check_valid(
    task: Task,
    random_seed: int | None,
):
    task.generate_trials(ntrials=NTRIALS, random_seed=random_seed)


@pytest.mark.parametrize(
    ("ntrials", "random_seed", "expected_exception"),
    [
        ("a", RND_SEED, TypeError),
        (0, RND_SEED, ValueError),
        (0.5, RND_SEED, TypeError),
        ((30, 40, 50), RND_SEED, ValueError),
        (("40", 50), RND_SEED, TypeError),
        ((40, "50"), RND_SEED, TypeError),
        (NTRIALS, "a", TypeError),
        (NTRIALS, -1, ValueError),
        (NTRIALS, 0.5, TypeError),
    ],
)
def test_generate_trials_check_invalid(
    task: Task,
    ntrials: Any,
    random_seed: Any,
    expected_exception: Any,
):
    with pytest.raises(expected_exception):
        task.generate_trials(ntrials=ntrials, random_seed=random_seed)


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


@pytest.mark.parametrize("session", [SESSION, {"v": 1, "a": 3}])
@pytest.mark.parametrize("random_seed", [RND_SEED, 100])
def test_reproduce_experiment(
    session: dict,
    random_seed: int | None,
):
    task = Task(name=NAME, session=session)
    trials = task.generate_trials(ntrials=NTRIALS, random_seed=random_seed)
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


def test_intensity_trials():
    task = Task(name=NAME, session=SESSION, stim_intensities=STIM_INTENSITIES, scaling=SCALING)
    trials = task.generate_trials(ntrials=NTRIALS)
    high_val = 0.6  # for a signal to be considered high
    low_val = 0.3  # for a signal to be considered low
    for n in range(NTRIALS):
        for idx, mod in enumerate(task._modalities):
            assert (
                (trials["inputs"][n][task._phases[n]["input"], idx] > high_val).all()
                if trials["modality_seq"][n] == mod  # check if the signal is high if the modality is the current one
                else (
                    trials["inputs"][n][task._phases[n]["input"], idx] < low_val
                ).all()  # check if the signal is low otherwise
            )
