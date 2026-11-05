"""Basic package tests for ragbench."""

import ragbench


def test_import_ragbench():
    """Verify that ragbench can be imported."""
    assert ragbench is not None


def test_version():
    """Verify that ragbench exposes the expected version."""
    assert ragbench.__version__ == "0.1.0"
