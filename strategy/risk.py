import pandas as pd
import numpy as np
import yaml
import pathlib

def load_config():
    path = pathlib.Path(__file__).parent.parent / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)

def calculate_position_size(account_balance, risk_pct, stop_distance_pct, signals_agreed=3):
    """
    Calculates position size such that hitting the stop loss costs exactly `risk_pct` of balance.
    Base risk is 2% (0.02). Scales down to 1% (0.01) if only 2/3 signals agree.
    Max position is capped at 95% of account balance (no leverage).
    """
    if stop_distance_pct <= 0:
        return 0
        
    # Scale risk based on signal confirmation
    actual_risk = risk_pct if signals_agreed == 3 else (risk_pct / 2.0)
    
    # Position size = (Risk $) / (Stop Loss %)
    risk_amount = account_balance * actual_risk
    ideal_position = risk_amount / stop_distance_pct
    max_position = account_balance * 0.95
    
    return min(ideal_position, max_position)

def calculate_stop_loss(entry_price, atr, signal_direction, regime):
    """ATR-based stop loss. Wider in high vol, tighter in low vol."""
    atr_multiplier = {'LOW': 1.5, 'MEDIUM': 2.0, 'HIGH': 2.5}.get(regime, 2.0)
    stop_distance = atr * atr_multiplier
    if 'BUY' in signal_direction:
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance

def calculate_take_profit(entry_price, stop_loss, signal_strength):
    """
    Risk/reward ratio scales with signal strength.
    Minimum 1.5:1 reward/risk. Strong signals target 3:1.
    """
    risk = abs(entry_price - stop_loss)
    # Norm signal strength to ~ 0-1 for multiplier (assume max bounded safely around 1.0)
    ss = min(max(signal_strength, 0.0), 1.0)
    rr_ratio = 1.5 + (ss * 1.5)
    
    if entry_price > stop_loss:
        return entry_price + (risk * rr_ratio)
    else:
        return entry_price - (risk * rr_ratio)

def check_max_drawdown(equity_curve):
    """Returns True if current drawdown exceeds max_drawdown_pct — halt trading."""
    if len(equity_curve) == 0: return False
    config = load_config()
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    current_dd = drawdown.iloc[-1]
    max_allowed = config['risk']['max_drawdown_pct']
    
    if abs(current_dd) > max_allowed:
        print(f"MAX DRAWDOWN EXCEEDED: {current_dd:.1%} > {max_allowed:.1%} — HALT")
        return True
    return False
