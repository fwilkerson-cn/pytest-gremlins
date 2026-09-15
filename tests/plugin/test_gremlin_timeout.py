"""Tests for the configurable per-mutant timeout: CLI option, session field, and threading."""

from __future__ import annotations

import ast
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

    def it_defaults_to_a_value_that_clears_a_one_minute_suite(self) -> None:
        assert DEFAULT_GREMLIN_TIMEOUT >= 300

    def it_accepts_an_explicit_timeout(self) -> None:
        session = GremlinSession(timeout=900)

        assert session.timeout == 900


@pytest.mark.small
class DescribeTestGremlinTimeout:
    def it_passes_the_given_timeout_to_the_subprocess(self) -> None:
        completed = subprocess.CompletedProcess(args=['pytest'], returncode=0, stdout=b'', stderr=b'')

        with patch('pytest_gremlins.plugin.subprocess.run', return_value=completed) as run:
            _test_gremlin(_make_gremlin(), ['pytest'], MagicMock(), instrumented_dir=None, timeout=600)

        assert run.call_args.kwargs['timeout'] == 600

    def it_defaults_the_subprocess_timeout_to_the_module_default(self) -> None:
        completed = subprocess.CompletedProcess(args=['pytest'], returncode=0, stdout=b'', stderr=b'')

        with patch('pytest_gremlins.plugin.subprocess.run', return_value=completed) as run:
            _test_gremlin(_make_gremlin(), ['pytest'], MagicMock(), instrumented_dir=None)

        assert run.call_args.kwargs['timeout'] == DEFAULT_GREMLIN_TIMEOUT

    def it_reports_timeout_when_the_subprocess_exceeds_the_limit(self) -> None:
        expired = subprocess.TimeoutExpired(cmd=['pytest'], timeout=1)

        with patch('pytest_gremlins.plugin.subprocess.run', side_effect=expired):
            result = _test_gremlin(_make_gremlin(), ['pytest'], MagicMock(), instrumented_dir=None, timeout=1)

        assert result.status is GremlinResultStatus.TIMEOUT

    def it_reports_survived_when_tests_pass_inside_the_limit(self) -> None:
        completed = subprocess.CompletedProcess(args=['pytest'], returncode=0, stdout=b'', stderr=b'')

        with patch('pytest_gremlins.plugin.subprocess.run', return_value=completed):
            result = _test_gremlin(_make_gremlin(), ['pytest'], MagicMock(), instrumented_dir=None, timeout=600)

        assert result.status is GremlinResultStatus.SURVIVED
