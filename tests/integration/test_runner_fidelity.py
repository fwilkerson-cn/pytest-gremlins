"""Per-mutant verdicts with the lightweight runner on and off.

Each case pairs a target with a test the runner cannot execute faithfully:
one taking a fixture, one parametrized, one needing a conftest fixture, one
``async def``. With the runner on, the test is imported and called with no
arguments: a fixture-taking test raises and a parametrized node id never
resolves. An async test reaches one of those two paths depending on its
plugin (anyio parametrizes the node id; an unparametrized plugin leaves a
coroutine that is never awaited), so none of them is judged.
With the runner off, every mutant runs through pytest and receives the
verdict pytest gives.

Both columns are pinned so a change to either shows up.
"""

from __future__ import annotations

import re

import pytest

_TARGET = """
def classify(n):
    if n > 10:
        return 'big'
    return 'small'
"""

_BASE_ARGS = (
    '--gremlins',
    '--gremlin-targets=sample.py',
    '--gremlin-operators=comparison',
    '--gremlin-no-coverage-filter',
    '-v',
)
_RUNNER_OFF = ('--gremlin-no-lightweight-runner',)

_SAYS_NOTHING = {
    'fixture': """
        def test_says_nothing(tmp_path):
            assert tmp_path.is_dir()
        """,
    'parametrized': """
        import pytest

        @pytest.mark.parametrize('n', [1, 2, 3])
        def test_says_nothing(n):
            assert n > 0
        """,
    'conftest_fixture': """
        def test_says_nothing(greeting):
            assert greeting == 'hello'
        """,
    'async': """
        import pytest

        @pytest.mark.anyio
        async def test_says_nothing():
            assert True
        """,
}

_CATCHES = {
    'fixture': """
        from sample import classify

        def test_covers(tmp_path):
            assert tmp_path.is_dir()
            assert classify(11) == 'big'
            assert classify(10) == 'small'
            assert classify(1) == 'small'
        """,
    'async': """
        import pytest
        from sample import classify

        @pytest.mark.anyio
        async def test_covers():
            assert classify(11) == 'big'
            assert classify(10) == 'small'
            assert classify(1) == 'small'
        """,
}

_CONFTEST = """
import pytest

@pytest.fixture
def greeting():
    return 'hello'
"""


def _verdicts(output: str) -> dict[str, int]:
    """Return the per-status gremlin counts from a console report."""

    def count(label: str) -> int:
        match = re.search(rf'{label}: (\d+) gremlins', output)
        return int(match.group(1)) if match else 0

    return {label: count(label) for label in ('Zapped', 'Survived', 'Timeout', 'Error')}


def _run(pytester: pytest.Pytester, test_source: str, *extra_args: str) -> dict[str, int]:
    pytester.makepyfile(sample=_TARGET)
    pytester.makepyfile(test_sample=test_source)
    result = pytester.runpytest_subprocess(*_BASE_ARGS, *extra_args)
    return _verdicts(result.stdout.str())


@pytest.fixture
def pytester_with_conftest_fixture(pytester_with_markers: pytest.Pytester) -> pytest.Pytester:
    existing = pytester_with_markers.path.joinpath('conftest.py').read_text()
    pytester_with_markers.makeconftest(existing + _CONFTEST)
    return pytester_with_markers


@pytest.mark.medium
class DescribeRunnerOff:
    """Every mutant runs through pytest, so unrelated tests cannot zap and covering tests do."""

    @pytest.mark.parametrize('case', ['fixture', 'parametrized', 'async'])
    def it_reports_survived_for_a_test_that_says_nothing(
        self, pytester_with_markers: pytest.Pytester, case: str
    ) -> None:
        verdicts = _run(pytester_with_markers, _SAYS_NOTHING[case], *_RUNNER_OFF)

        assert verdicts['Zapped'] == 0
        assert verdicts['Survived'] > 0
        assert verdicts['Timeout'] == 0
        assert verdicts['Error'] == 0

    def it_reports_survived_for_a_test_needing_a_conftest_fixture(
        self,
        pytester_with_conftest_fixture: pytest.Pytester,
    ) -> None:
        verdicts = _run(pytester_with_conftest_fixture, _SAYS_NOTHING['conftest_fixture'], *_RUNNER_OFF)

        assert verdicts['Zapped'] == 0
        assert verdicts['Survived'] > 0
        assert verdicts['Error'] == 0

    @pytest.mark.parametrize('case', ['fixture', 'async'])
    def it_reports_zapped_for_a_test_that_catches_the_mutant(
        self,
        pytester_with_markers: pytest.Pytester,
        case: str,
    ) -> None:
        verdicts = _run(pytester_with_markers, _CATCHES[case], *_RUNNER_OFF)

        assert verdicts['Zapped'] > 0
        assert verdicts['Error'] == 0

    def it_holds_under_parallel_execution(self, pytester_with_markers: pytest.Pytester) -> None:
        verdicts = _run(
            pytester_with_markers,
            _SAYS_NOTHING['fixture'],
            *_RUNNER_OFF,
            '--gremlin-parallel',
            '--gremlin-workers=2',
        )

        assert verdicts['Zapped'] == 0
        assert verdicts['Survived'] > 0

    def it_holds_under_batch_execution(self, pytester_with_markers: pytest.Pytester) -> None:
        verdicts = _run(pytester_with_markers, _SAYS_NOTHING['fixture'], *_RUNNER_OFF, '--gremlin-batch')

        assert verdicts['Zapped'] == 0
        assert verdicts['Survived'] > 0

    def it_can_be_turned_off_from_pyproject(self, pytester_with_markers: pytest.Pytester) -> None:
        pytester_with_markers.makepyprojecttoml('[tool.pytest-gremlins]\nlightweight_runner = false\n')

        verdicts = _run(pytester_with_markers, _SAYS_NOTHING['fixture'])

        assert verdicts['Zapped'] == 0
        assert verdicts['Survived'] > 0

    def it_lets_the_command_line_beat_pyproject(self, pytester_with_markers: pytest.Pytester) -> None:
        pytester_with_markers.makepyprojecttoml('[tool.pytest-gremlins]\nlightweight_runner = true\n')

        verdicts = _run(pytester_with_markers, _SAYS_NOTHING['fixture'], *_RUNNER_OFF)

        assert verdicts['Zapped'] == 0
        assert verdicts['Survived'] > 0


@pytest.mark.medium
class DescribeRunnerOn:
    """The default path is unchanged: the runner's verdicts on the same fixtures are pinned as they stand."""

    @pytest.mark.parametrize('case', ['fixture', 'parametrized', 'async'])
    def it_scores_an_unexecutable_test_as_a_kill(self, pytester_with_markers: pytest.Pytester, case: str) -> None:
        verdicts = _run(pytester_with_markers, _SAYS_NOTHING[case])

        assert verdicts['Survived'] == 0
        assert verdicts['Zapped'] > 0

    def it_scores_a_test_needing_a_conftest_fixture_as_a_kill(
        self,
        pytester_with_conftest_fixture: pytest.Pytester,
    ) -> None:
        verdicts = _run(pytester_with_conftest_fixture, _SAYS_NOTHING['conftest_fixture'])

        assert verdicts['Survived'] == 0
        assert verdicts['Zapped'] > 0


_TARGET_READING_FILE = """
from pathlib import Path

HERE = Path(__file__).resolve().parent


def classify(n):
    if n > 10:
        return 'big'
    return 'small'


def untested(n):
    if n > 3:
        return 'x'
    return 'y'
"""

_COVERS_CLASSIFY_ONLY = """
from sample import classify

def test_covers():
    assert classify(11) == 'big'
    assert classify(10) == 'small'
    assert classify(1) == 'small'
"""


@pytest.mark.medium
class DescribeInstrumentedModuleAttributes:
    """An instrumented module keeps the import-time attributes its code reads, under either runner."""

    @pytest.mark.parametrize('runner_args', [(), _RUNNER_OFF], ids=['runner_on', 'runner_off'])
    def it_gives_an_instrumented_module_a_usable_dunder_file(
        self,
        pytester_with_markers: pytest.Pytester,
        runner_args: tuple[str, ...],
    ) -> None:
        pytester_with_markers.makepyfile(sample=_TARGET_READING_FILE)
        pytester_with_markers.makepyfile(test_sample=_COVERS_CLASSIFY_ONLY)

        result = pytester_with_markers.runpytest_subprocess(*_BASE_ARGS, *runner_args)

        verdicts = _verdicts(result.stdout.str())
        assert verdicts['Error'] == 0
        assert verdicts['Survived'] > 0
        assert verdicts['Zapped'] > 0


_SLOW_COVERING_TEST = """
import time
from sample import classify

def test_covers_slowly():
    time.sleep(3)
    assert classify(11) == 'big'
    assert classify(1) == 'small'
"""


@pytest.mark.medium
class DescribeWarmCacheAcrossConfigChanges:
    """A cached verdict is only reused for the timeout and runner mode that produced it."""

    def it_rejudges_a_timeout_when_the_timeout_is_raised(self, pytester_with_markers: pytest.Pytester) -> None:
        pytester_with_markers.makepyfile(sample=_TARGET)
        pytester_with_markers.makepyfile(test_sample=_SLOW_COVERING_TEST)
        cache_args = (*_BASE_ARGS, *_RUNNER_OFF, '--gremlin-cache')

        first = _verdicts(pytester_with_markers.runpytest_subprocess(*cache_args, '--gremlin-timeout=1').stdout.str())
        second_run = pytester_with_markers.runpytest_subprocess(*cache_args, '--gremlin-timeout=120')
        second = _verdicts(second_run.stdout.str())

        assert first['Timeout'] > 0
        assert second['Timeout'] == 0
        assert second['Zapped'] > 0
        assert 'cache hit' not in second_run.stdout.str()

    def it_rejudges_a_fabricated_kill_when_the_runner_is_turned_off(
        self,
        pytester_with_markers: pytest.Pytester,
    ) -> None:
        pytester_with_markers.makepyfile(sample=_TARGET)
        pytester_with_markers.makepyfile(test_sample=_SAYS_NOTHING['fixture'])
        cache_args = (*_BASE_ARGS, '--gremlin-cache')

        first = _verdicts(pytester_with_markers.runpytest_subprocess(*cache_args).stdout.str())
        second = _verdicts(pytester_with_markers.runpytest_subprocess(*cache_args, *_RUNNER_OFF).stdout.str())

        assert first['Zapped'] > 0
        assert second['Zapped'] == 0
        assert second['Survived'] > 0
