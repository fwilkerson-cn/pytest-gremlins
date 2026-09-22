"""Tests for the configurable per-mutant timeout: CLI option, session field, and threading."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
from unittest.mock import (
    MagicMock,
    patch,
)

from _pytest.config.argparsing import (
    OptionGroup,
    Parser,
)
import pytest

from pytest_gremlins.instrumentation.gremlin import Gremlin
from pytest_gremlins.plugin import (
    DEFAULT_GREMLIN_TIMEOUT,
    GremlinSession,
    _exit_on_unusable_numeric_options,
    _run_tests_with_coverage,
    _test_gremlin,
    pytest_addoption,
)
from pytest_gremlins.reporting.results import GremlinResultStatus


def _make_gremlin(gremlin_id: str = 'g001', file_path: str = 'src/mod.py') -> Gremlin:
    return Gremlin(
        gremlin_id=gremlin_id,
        file_path=file_path,
        line_number=1,
        original_node=ast.Constant(value=True),
        mutated_node=ast.Constant(value=False),
        operator_name='BooleanNegate',
        description='negate boolean',
    )


@pytest.mark.small
class DescribeGremlinTimeoutOption:
    def it_registers_the_gremlin_timeout_option(self) -> None:
        parser = MagicMock(spec=Parser)
        group = MagicMock(spec=OptionGroup)
        parser.getgroup.return_value = group

        pytest_addoption(parser)

        added_option_names = [call.args[0] for call in group.addoption.call_args_list if call.args]
        assert '--gremlin-timeout' in added_option_names

    def it_defaults_the_option_to_none_so_toml_can_win(self) -> None:
        parser = MagicMock(spec=Parser)
        group = MagicMock(spec=OptionGroup)
        parser.getgroup.return_value = group

        pytest_addoption(parser)

        added_options = {c.args[0]: c.kwargs for c in group.addoption.call_args_list if c.args}
        assert added_options['--gremlin-timeout']['default'] is None

    def it_parses_the_option_as_an_int(self) -> None:
        parser = MagicMock(spec=Parser)
        group = MagicMock(spec=OptionGroup)
        parser.getgroup.return_value = group

        pytest_addoption(parser)

        added_options = {c.args[0]: c.kwargs for c in group.addoption.call_args_list if c.args}
        assert added_options['--gremlin-timeout']['type'] is int


@pytest.mark.small
class DescribeGremlinSessionTimeout:
    def it_defaults_the_session_timeout_to_the_module_default(self) -> None:
        session = GremlinSession()

        assert session.timeout == DEFAULT_GREMLIN_TIMEOUT

    def it_keeps_the_default_at_thirty_seconds(self) -> None:
        assert DEFAULT_GREMLIN_TIMEOUT == 30

    def it_accepts_an_explicit_timeout(self) -> None:
        session = GremlinSession(timeout=900)

        assert session.timeout == 900


@pytest.mark.small
class DescribeTestGremlinTimeout:
    def it_passes_the_given_timeout_to_the_subprocess(self) -> None:
        completed = subprocess.CompletedProcess(args=['pytest'], returncode=0, stdout=b'', stderr=b'')

        with patch('pytest_gremlins.plugin.subprocess.run', return_value=completed) as run:
            _test_gremlin(_make_gremlin(), ['pytest'], Path('/project'), instrumented_dir=None, timeout=600)

        assert run.call_args.kwargs['timeout'] == 600

    def it_defaults_the_subprocess_timeout_to_the_module_default(self) -> None:
        completed = subprocess.CompletedProcess(args=['pytest'], returncode=0, stdout=b'', stderr=b'')

        with patch('pytest_gremlins.plugin.subprocess.run', return_value=completed) as run:
            _test_gremlin(_make_gremlin(), ['pytest'], Path('/project'), instrumented_dir=None)

        assert run.call_args.kwargs['timeout'] == DEFAULT_GREMLIN_TIMEOUT

    def it_reports_timeout_when_the_subprocess_exceeds_the_limit(self) -> None:
        expired = subprocess.TimeoutExpired(cmd=['pytest'], timeout=1)

        with patch('pytest_gremlins.plugin.subprocess.run', side_effect=expired):
            result = _test_gremlin(_make_gremlin(), ['pytest'], Path('/project'), instrumented_dir=None, timeout=1)

        assert result.status is GremlinResultStatus.TIMEOUT

    def it_reports_survived_when_tests_pass_inside_the_limit(self) -> None:
        completed = subprocess.CompletedProcess(args=['pytest'], returncode=0, stdout=b'', stderr=b'')

        with patch('pytest_gremlins.plugin.subprocess.run', return_value=completed):
            result = _test_gremlin(_make_gremlin(), ['pytest'], Path('/project'), instrumented_dir=None, timeout=600)

        assert result.status is GremlinResultStatus.SURVIVED


@pytest.mark.small
class DescribeGremlinTimeoutRejection:
    @pytest.mark.parametrize('value', [0, -5])
    def it_exits_naming_the_flag_and_the_value(self, value: int) -> None:
        with patch('pytest_gremlins.plugin.pytest') as mock_pytest:
            _exit_on_unusable_numeric_options(None, None, value)

        mock_pytest.exit.assert_called_once()
        message = mock_pytest.exit.call_args.args[0]
        assert '--gremlin-timeout' in message
        assert str(value) in message
        assert mock_pytest.exit.call_args.kwargs['returncode'] == 4

    def it_accepts_a_positive_value(self) -> None:
        with patch('pytest_gremlins.plugin.pytest') as mock_pytest:
            _exit_on_unusable_numeric_options(None, None, 1)

        mock_pytest.exit.assert_not_called()

    @pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.0, 100.5])
    def it_rejects_a_max_pardons_pct_outside_zero_to_one_hundred(self, value: float) -> None:
        with patch('pytest_gremlins.plugin.pytest') as mock_pytest:
            _exit_on_unusable_numeric_options(value, None, None)

        mock_pytest.exit.assert_called_once()
        assert '--gremlin-max-pardons-pct' in mock_pytest.exit.call_args.args[0]


@pytest.mark.medium
class DescribeCoveragePrescanTimeout:
    def it_lets_the_configured_timeout_raise_the_prescan_cap(self, tmp_path: Path) -> None:
        with patch('pytest_gremlins.plugin.subprocess.run') as run:
            _run_tests_with_coverage(['tests/test_example.py::test_one'], tmp_path, timeout=600)

        assert run.call_args.kwargs['timeout'] == 600

    def it_keeps_the_default_cap_when_the_configured_timeout_is_shorter(self, tmp_path: Path) -> None:
        with patch('pytest_gremlins.plugin.subprocess.run') as run:
            _run_tests_with_coverage(['tests/test_example.py::test_one'], tmp_path, timeout=30)

        assert run.call_args.kwargs['timeout'] == 120


_SLOW_TARGET = """
def classify(n):
    if n > 10:
        return 'big'
    return 'small'
"""

_SLOW_COVERING_TEST = """
import time
from sample import classify

def test_covers_slowly():
    time.sleep(3)
    assert classify(11) == 'big'
    assert classify(1) == 'small'
"""


@pytest.mark.medium
class DescribeConfiguredTimeoutReachesEveryPath:
    """--gremlin-timeout bounds the subprocess, batch and parallel executions alike."""

    @pytest.mark.parametrize(
        'mode_args',
        [(), ('--gremlin-batch',), ('--gremlin-parallel', '--gremlin-workers=2')],
        ids=['subprocess', 'batch', 'parallel'],
    )
    def it_reports_timeout_when_the_covering_test_outlasts_the_limit(
        self,
        pytester_with_markers: pytest.Pytester,
        mode_args: tuple[str, ...],
    ) -> None:
        pytester_with_markers.makepyfile(sample=_SLOW_TARGET)
        pytester_with_markers.makepyfile(test_sample=_SLOW_COVERING_TEST)

        result = pytester_with_markers.runpytest_subprocess(
            '--gremlins',
            '--gremlin-targets=sample.py',
            '--gremlin-operators=comparison',
            '--gremlin-no-coverage-filter',
            '--gremlin-no-lightweight-runner',
            '--gremlin-timeout=1',
            *mode_args,
        )

        assert re.search(r'Timeout: ([1-9]\d*) gremlins', result.stdout.str()), result.stdout.str()[-800:]
        assert 'Zapped: 0 gremlins' in result.stdout.str()
