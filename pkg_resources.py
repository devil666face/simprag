"""Minimal `pkg_resources` compatibility layer.

This is provided because the version of `setuptools` installed in this
environment does not expose the legacy top-level `pkg_resources` module
that some third-party libraries (like `milvus_lite`) still import.

Only the pieces needed by those libraries are implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # Python 3.8+
    import importlib.metadata as _im
except ImportError:  # pragma: no cover - for very old Pythons
    import importlib_metadata as _im  # type: ignore


class DistributionNotFound(Exception):
    """Exception raised when a distribution cannot be located."""


@dataclass
class Distribution:
    version: str


def get_distribution(name: str) -> Distribution:
    """Return a minimal distribution object for *name*.

    Only the ``version`` attribute is provided, which is all that
    ``milvus_lite`` (and most callers) rely on.
    """

    try:
        version = _im.version(name)
    except _im.PackageNotFoundError as exc:  # type: ignore[attr-defined]
        raise DistributionNotFound(str(exc)) from exc
    return Distribution(version=version)


__all__ = ["DistributionNotFound", "get_distribution", "Distribution"]
