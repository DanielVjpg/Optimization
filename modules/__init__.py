"""Optimization modules.

Importing this package registers every module with :mod:`core.registry`. The
app shell renders whatever it finds there, so shipping a new module is a
one-line change here plus a new folder — no edits to `core`, to the app, or to
any existing module.

To add a module:

1. Create ``modules/<name>/`` with a ``module.py`` defining a subclass of
   :class:`core.module_base.OptimizationModule` and a ``ui.py`` for its
   Streamlit rendering.
2. Import it below and call ``registry.register(...)``.
3. Drop its ``register_planned`` line.

Modules on the roadmap are registered as *planned* so the roadmap is visible
in the product itself rather than only in the README.
"""

from core import registry
from modules.inventory.module import InventoryModule

registry.register(InventoryModule())

# --- Roadmap ----------------------------------------------------------------
# Both of these have a stub package (modules/staffing/, modules/layout/) with a
# README describing the intended approach and the inputs they will need. They
# read the same Dataset the inventory module does, which is the point of
# keeping ingestion in core.

registry.register_planned(
    key="staffing",
    display_name="Staffing & Scheduling",
    description=(
        "Convert forecast demand into required labour by half-hour, then build "
        "shift schedules that meet a target service level at minimum cost. "
        "Queueing theory for coverage, integer programming for the roster."
    ),
)

registry.register_planned(
    key="layout",
    display_name="Layout & Workflow",
    description=(
        "Analyse station placement and barista travel using from-to charts and "
        "spaghetti diagrams; propose a layout that cuts non-value-added motion "
        "and shortens drink cycle time."
    ),
)

__all__ = ["InventoryModule"]
