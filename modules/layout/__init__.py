"""Layout & workflow — PLANNED, not implemented.

See ``README.md`` in this folder for the intended inputs and approach.

When this is built, the work is:

1. Add ``module.py`` with::

       class LayoutModule(OptimizationModule):
           key = "layout"
           display_name = "Layout & Workflow"
           required_inputs = ("sales",)
           def run(self, dataset, **options) -> ModuleResult: ...
           def render(self, result) -> None: ...

2. In ``modules/__init__.py``, replace the ``register_planned("layout", ...)``
   call with ``registry.register(LayoutModule())``.

Nothing in ``core`` or in ``modules/inventory`` changes.
"""
