# src/fidelity/__init__.py
"""Fidelity personal-portfolio integration: data (SnapTrade / CSV), performance, rebalancing."""
from src.fidelity.schema import POSITION_COLUMNS, ACTIVITY_COLUMNS
from src.fidelity.csv_source import load_positions_csv, load_activity_csv
from src.fidelity.rebalance import RebalanceSettings, build_rebalance_plan
