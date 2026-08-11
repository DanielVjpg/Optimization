"""The contract every optimization module implements.

Adding a module means: create ``modules/<name>/``, subclass
:class:`OptimizationModule`, register it in ``modules/__init__.py``. Nothing in
``core`` or in any existing module changes. That is the whole extension story.

A module must be able to answer four things:

1. Who am I?              -> ``key``, ``display_name``, ``description``
2. What data do I need?   -> ``required_inputs`` / ``optional_inputs``
3. What do I compute?     -> ``run(dataset) -> ModuleResult``
4. How do I look?         -> ``render(result)`` using Streamlit

``run`` must be free of Streamlit calls so the analysis can be driven from a
notebook, a script or a test without a browser.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from core.dataset import Dataset


@dataclass
class ModuleResult:
    """What a module hands back after running.

    ``tables`` and ``summary`` are deliberately generic so the app shell can
    display, export or diff any module's output without knowing what it means.
    Module-specific rich objects go in ``payload``.
    """

    module_key: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    headline: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    payload: Any = None


class ModuleInputError(ValueError):
    """Raised when a dataset does not satisfy a module's declared inputs."""


class OptimizationModule(ABC):
    """Base class for an analysis module."""

    key: str = ""
    display_name: str = ""
    description: str = ""
    required_inputs: tuple[str, ...] = ("sales",)
    optional_inputs: tuple[str, ...] = ()

    def validate(self, dataset: Dataset) -> None:
        """Check the dataset satisfies ``required_inputs``.

        Override to add module-specific checks, calling ``super().validate()``.
        """
        available = dataset.available_inputs()
        # A purchases-only dataset still yields a demand signal, so treat
        # 'sales' as satisfied when the loader promoted purchases to proxy.
        if dataset.demand_source == "purchases_proxy" and "purchases" in available:
            available = available | {"sales"}
        missing = [name for name in self.required_inputs if name not in available]
        if missing:
            raise ModuleInputError(
                f"{self.display_name or self.key} requires "
                f"{', '.join(self.required_inputs)}; missing: {', '.join(missing)}."
            )

    @abstractmethod
    def run(self, dataset: Dataset, **options: Any) -> ModuleResult:
        """Run the analysis. Must not call Streamlit."""

    @abstractmethod
    def render(self, result: ModuleResult) -> None:
        """Draw the module's Streamlit UI for ``result``."""

    def render_options(self) -> dict[str, Any]:
        """Draw module-specific sidebar controls and return them as options.

        Called before :meth:`run`; the returned dict is passed through as
        keyword arguments. Default: no options.
        """
        return {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} key={self.key!r}>"
