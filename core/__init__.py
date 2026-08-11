"""Shared foundation for every Optimization module.

`core` knows about *businesses and their data*. It knows nothing about
inventory, staffing, or layout specifically. Anything that is only meaningful
to a single module belongs in that module's package, not here.

The contract a module implements lives in :mod:`core.module_base`.
"""

from core.business_profile import BusinessProfile
from core.dataset import Dataset, ItemAttributes
from core.module_base import ModuleResult, OptimizationModule
from core.schemas import ITEMS_SCHEMA, PURCHASES_SCHEMA, SALES_SCHEMA, SchemaError

__all__ = [
    "BusinessProfile",
    "Dataset",
    "ItemAttributes",
    "ModuleResult",
    "OptimizationModule",
    "SALES_SCHEMA",
    "PURCHASES_SCHEMA",
    "ITEMS_SCHEMA",
    "SchemaError",
]
