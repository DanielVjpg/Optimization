"""Staffing & scheduling — PLANNED, not implemented.

See ``README.md`` in this folder for the intended inputs and approach.

When this is built, the work is:

1. Add ``module.py`` with::

       class StaffingModule(OptimizationModule):
           key = "staffing"
           display_name = "Staffing & Scheduling"
           required_inputs = ("sales",)
           def run(self, dataset, **options) -> ModuleResult: ...
           def render(self, result) -> None: ...

2. In ``modules/__init__.py``, replace the ``register_planned("staffing", ...)``
   call with ``registry.register(StaffingModule())``.

Nothing in ``core`` or in ``modules/inventory`` changes. That constraint is the
reason the registry and the shared ``Dataset`` exist.
"""
