"""Module registry — the list the app shell renders from.

The app never imports a module directly. It asks the registry what exists,
which means shipping a new module is a registration call, not an edit to the
UI. Modules that are planned but unbuilt are registered too, so the roadmap is
visible in the product instead of only in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.module_base import OptimizationModule

_ACTIVE: dict[str, OptimizationModule] = {}
_PLANNED: dict[str, "PlannedModule"] = {}


@dataclass(frozen=True)
class PlannedModule:
    """A module on the roadmap that has no implementation yet."""

    key: str
    display_name: str
    description: str
    status: str = "planned"


def register(module: OptimizationModule) -> OptimizationModule:
    """Register a built module. Returns it, so it can be used as a decorator."""
    if not module.key:
        raise ValueError(f"{type(module).__name__} must define a non-empty key")
    if module.key in _ACTIVE:
        raise ValueError(f"module key {module.key!r} is already registered")
    _ACTIVE[module.key] = module
    _PLANNED.pop(module.key, None)
    return module


def register_planned(key: str, display_name: str, description: str, status: str = "planned") -> None:
    """Declare a roadmap module so the UI can show it as coming soon."""
    if key in _ACTIVE:
        return
    _PLANNED[key] = PlannedModule(key, display_name, description, status)


def get_module(key: str) -> OptimizationModule:
    try:
        return _ACTIVE[key]
    except KeyError:
        raise KeyError(
            f"No module registered under {key!r}. Available: {sorted(_ACTIVE)}"
        ) from None


def active_modules() -> list[OptimizationModule]:
    return [_ACTIVE[key] for key in sorted(_ACTIVE)]


def planned_modules() -> list[PlannedModule]:
    """Roadmap modules in the order they were declared.

    Insertion order, not alphabetical: the sequence in ``modules/__init__.py``
    is the intended build order and is worth showing as such.
    """
    return list(_PLANNED.values())


def clear() -> None:
    """Reset the registry. Used by tests."""
    _ACTIVE.clear()
    _PLANNED.clear()
