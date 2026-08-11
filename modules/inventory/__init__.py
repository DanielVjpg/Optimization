"""Inventory and demand-forecasting module.

Layout:

``forecasting.py``  demand models, backtesting, per-item model selection
``policy.py``       safety stock, reorder point, EOQ and its constraints
``diagnostics.py``  over/under-ordering detection and cost estimation
``module.py``       orchestration; implements the OptimizationModule contract
``ui.py``           Streamlit rendering
"""

from modules.inventory.module import InventoryAnalysis, InventoryModule

__all__ = ["InventoryModule", "InventoryAnalysis"]
