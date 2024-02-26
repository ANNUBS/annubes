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
    assert task.session == expected_dict
    assert isinstance(task.session, expected_type)


@pytest.mark.parametrize(
    ("catch_prob", "expected"),
    [
        pytest.param(-1, -1, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(1, 1, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(2, 2, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(None, None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("0.6", "0.6", marks=pytest.mark.xfail(raises=ValueError)),
        (0.6, 0.6),
    ],
)
def test_init_catch_prob(catch_prob: float | None, expected: float | None):
    task = Task(NAME, catch_prob=catch_prob)
    assert task.name == NAME
    assert task.catch_prob == expected


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
    assert task.modalities == expected_modalities
    assert task.n_inputs == len(expected_modalities) + 1  # add the start signal
    assert max(task.time) == stim_time + fix_time + iti
    assert len(task.time) == int((stim_time + fix_time + iti + task.dt) / task.dt)
