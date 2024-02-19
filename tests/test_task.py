import pytest

from annubes.task import Task


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
def test_task_catch_prob(catch_prob: float | None, expected: float | None):
    task = Task("task1", catch_prob=catch_prob)
    assert task.name == "task1"
    assert task.catch_prob == expected
