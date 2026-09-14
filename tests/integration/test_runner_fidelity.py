"""Per-mutant runs must reproduce the verdicts a real pytest run would give.

Each test in ``DescribeVerdictFidelity`` pairs a target function with a test
that says nothing about it. The only honest verdict is ``Survived``: the test
cannot distinguish the original from any mutant. A ``Zapped`` verdict means the
runner counted its own inability to execute the test as the test catching the
mutation.
"""

from __future__ import annotations

import re

import pytest

_TARGET = """
def classify(n):
    if n > 10:
        return 'big'
    if n > 0:
        return 'small'
    return 'zero'
"""

_GREMLIN_ARGS = (
    '--gremlins',
    '--gremlin-targets=sample.py',
    '--gremlin-no-coverage-filter',
    '-v',
)


def _verdicts(output: str) -> tuple[int, int, int]:
    """Return (zapped, survived, error) counts from a console report."""

    def count(label: str) -> int:
        match = re.search(rf'{label}: (\d+) gremlins', output)
        return int(match.group(1)) if match else 0

    return count('Zapped'), count('Survived'), count('Error')


@pytest.mark.medium
class DescribeVerdictFidelity:
    """Verdicts for tests that do not exercise the target."""

    def it_does_not_credit_a_fixture_taking_test(self, pytester_with_markers: pytest.Pytester) -> None:
        pytester_with_markers.makepyfile(sample=_TARGET)
        pytester_with_markers.makepyfile(
            test_sample="""
            def test_says_nothing(tmp_path):
                assert tmp_path.is_dir()
            """,
        )

        result = pytester_with_markers.runpytest_subprocess(*_GREMLIN_ARGS)

        zapped, survived, _ = _verdicts(result.stdout.str())
        assert (zapped, survived) == (0, 11)

    def it_does_not_credit_a_parametrized_test(self, pytester_with_markers: pytest.Pytester) -> None:
        pytester_with_markers.makepyfile(sample=_TARGET)
        pytester_with_markers.makepyfile(
            test_sample="""
            import pytest

            @pytest.mark.parametrize('n', [1, 2, 3])
            def test_says_nothing(n):
                assert n > 0
            """,
        )

        result = pytester_with_markers.runpytest_subprocess(*_GREMLIN_ARGS)

        zapped, survived, _ = _verdicts(result.stdout.str())
        assert (zapped, survived) == (0, 11)

    def it_does_not_credit_a_test_needing_a_conftest_fixture(
        self,
        pytester_with_markers: pytest.Pytester,
    ) -> None:
        pytester_with_markers.makepyfile(sample=_TARGET)
        pytester_with_markers.makepyfile(
            conftest_extra="""
            # placeholder module; the fixture lives in conftest.py below
            """,
        )
        pytester_with_markers.makepyfile(
            test_sample="""
            import pytest

            @pytest.fixture
            def greeting():
                return 'hello'

            def test_says_nothing(greeting):
                assert greeting == 'hello'
            """,
        )

        result = pytester_with_markers.runpytest_subprocess(*_GREMLIN_ARGS)

        zapped, survived, _ = _verdicts(result.stdout.str())
        assert (zapped, survived) == (0, 11)

    def it_credits_a_fixture_taking_test_that_covers_the_target(
        self,
        pytester_with_markers: pytest.Pytester,
    ) -> None:
        pytester_with_markers.makepyfile(sample=_TARGET)
        pytester_with_markers.makepyfile(
            test_sample="""
            from sample import classify

            def test_covers(tmp_path):
                assert tmp_path.is_dir()
                assert classify(11) == 'big'
                assert classify(10) == 'small'
                assert classify(1) == 'small'
                assert classify(0) == 'zero'
                assert classify(-3) == 'zero'
            """,
        )

        result = pytester_with_markers.runpytest_subprocess(*_GREMLIN_ARGS)

        zapped, survived, _ = _verdicts(result.stdout.str())
        assert (zapped, survived) == (11, 0)
