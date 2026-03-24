#!/usr/bin/env python3
"""
MNQ 5m V5 AGG Dual Mode Trend Continuation Strategy
Converted from TradingView Pine Script to Python

Note: This is a signal generation module. Trailing stops and intrabar
order execution are approximated for bar-close signal generation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, time

name = "mnq-5m-dual-mode-trend"
timeframe = "5m"
leverage = 1.0

def _ema(series: np.ndarray, length: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    result = np.full_like(series, np.nan, dtype=np.float64)
    alpha = 2.0 / (length + 1)
    result[0] = series[0]
    for i in range(1, len(series)):
        if not np.isnan(series[i]):
            if np.isnan(result[i-1]):
                result[i] = series[i]
            else:
                result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    return result

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Calculate Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr = np.full(n, np.nan)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length
    return atr

def _lowest(series: np.ndarray, lookback: int) -> np.ndarray:
    """Calculate lowest value over lookback period (excluding current bar)."""
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(lookback, n):
        result[i] = np.nanmin(series[i-lookback:i])
    return result

def _in_rth(open_time: pd.Series, use_rth: bool, session_start: time, session_end: time) -> np.ndarray:
    """Check if bar is within Regular Trading Hours."""
    if not use_rth:
        return np.ones(len(open_time), dtype=bool)
    
    result = np.zeros(len(open_time), dtype=bool)
    for i, ot in enumerate(open_time):
        if pd.isna(ot):
            continue
        bar_time = ot.time() if hasattr(ot, 'time') else datetime.fromtimestamp(ot).time()
        if session_start <= bar_time <= session_end:
            result[i] = True
    return result

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Generate trading signals for the MNQ 5m Dual Mode Trend Strategy.
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume]
        
    Returns:
        numpy.ndarray of position intent: +1 (long), -1 (short), 0 (flat)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    # Extract price data
    open_time = prices['open_time'].values
    open_p = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Strategy parameters (from Pine inputs)
    fast_len = 21
    slow_len = 50
    slope_len = 5
    min_slope_pt = 2.0
    atr_len = 14
    atr_rise_lb = 6
    atr_rise_min = 0.0
    relax_atr_for_shorts = True
    cooldown_bars = 3
    stop_atr = 2.0
    scalp_tp_atr = 3.0
    scalp_trail_atr = 1.6
    runner_tp_atr = 5.0
    runner_trail_atr = 2.4
    runner_trail_after = 1.6
    mode = "Scalp"  # Default mode
    use_rth = True
    session_start = time(9, 30)
    session_end = time(15, 55)
    flat_at_end = True
    use_breakdown_short = True
    break_lookback = 10
    require_bear_body = True
    use_break_only_if_no_pb = False
    
    # Calculate indicators
    fast_ema = _ema(close, fast_len)
    slow_ema = _ema(close, slow_len)
    atr_val = _atr(high, low, close, atr_len)
    lowest_low = _lowest(low, break_lookback)
    
    # Session filtering
    in_rth = _in_rth(pd.Series(open_time), use_rth, session_start, session_end)
    session_just_ended = np.zeros(n, dtype=bool)
    if n > 1:
        for i in range(1, n):
            session_just_ended[i] = in_rth[i-1] and not in_rth[i]
    
    # Trend detection
    trend_up = fast_ema > slow_ema
    trend_down = fast_ema < slow_ema
    
    # Slope filter
    fast_slope = np.full(n, np.nan)
    for i in range(slope_len, n):
        fast_slope[i] = fast_ema[i] - fast_ema[i - slope_len]
    
    slope_ok_long = fast_slope >= min_slope_pt
    slope_ok_short = fast_slope <= -min_slope_pt
    
    # ATR filter
    atr_delta = np.full(n, np.nan)
    for i in range(atr_rise_lb, n):
        atr_delta[i] = atr_val[i] - atr_val[i - atr_rise_lb]
    atr_rising_ok = atr_delta > atr_rise_min
    
    filters_ok_long = slope_ok_long & atr_rising_ok
    filters_ok_short = slope_ok_short & (np.ones(n, dtype=bool) if relax_atr_for_shorts else atr_rising_ok)
    
    # Entry logic (pullback)
    raw_long = trend_up & (low <= fast_ema) & (close > fast_ema)
    raw_short = trend_down & (high >= fast_ema) & (close < fast_ema)
    
    long_pullback_signal = in_rth & raw_long & filters_ok_long
    short_pullback_signal = in_rth & raw_short & filters_ok_short
    
    # Breakdown shorts
    bear_body_ok = ~require_bear_body | (close < open_p)
    breakdown_short = np.zeros(n, dtype=bool)
    for i in range(n):
        if use_breakdown_short and trend_down[i] and slope_ok_short[i] and bear_body_ok[i]:
            if i >= break_lookback and close[i] < lowest_low[i]:
                breakdown_short[i] = True
    
    short_signal_base = in_rth & (short_pullback_signal | (breakdown_short & (~use_break_only_if_no_pb | ~short_pullback_signal)))
    long_signal_base = in_rth & long_pullback_signal
    
    # State tracking
    position = 0  # 0=flat, 1=long, -1=short
    last_exit_bar = -cooldown_bars - 1
    entry_price = np.nan
    entry_atr = np.nan
    entry_mode = mode
    highest_since_entry = np.nan
    lowest_since_entry = np.nan
    
    for i in range(n):
        # Cooldown check
        cooldown_ok = (i - last_exit_bar) > cooldown_bars
        
        # Entry signals
        long_signal = cooldown_ok and long_signal_base[i] and position <= 0
        short_signal = cooldown_ok and short_signal_base[i] and position >= 0
        
        # Check exits first
        exit_signal = False
        if position != 0 and not np.isnan(entry_price) and not np.isnan(entry_atr):
            is_long = position > 0
            tp_mult = scalp_tp_atr if entry_mode == "Scalp" else runner_tp_atr
            trail_atr = scalp_trail_atr if entry_mode == "Scalp" else runner_trail_atr
            
            # Update extremes
            if is_long:
                highest_since_entry = max(highest_since_entry, high[i]) if not np.isnan(highest_since_entry) else high[i]
            else:
                lowest_since_entry = min(lowest_since_entry, low[i]) if not np.isnan(lowest_since_entry) else low[i]
            
            # Stop loss
            stop_price = entry_price - stop_atr * entry_atr if is_long else entry_price + stop_atr * entry_atr
            # Take profit
            tp_price = entry_price + tp_mult * entry_atr if is_long else entry_price - tp_mult * entry_atr
            
            # Trailing stop (approximated)
            trail_active = (entry_mode == "Scalp") or (
                (is_long and (highest_since_entry - entry_price) >= runner_trail_after * entry_atr) or
                (not is_long and (entry_price - lowest_since_entry) >= runner_trail_after * entry_atr)
            )
            
            trail_price = np.nan
            if trail_active:
                if is_long:
                    trail_price = highest_since_entry - trail_atr * entry_atr
                else:
                    trail_price = lowest_since_entry + trail_atr * entry_atr
            
            # Check if price hit stop or target (using next bar open approximation)
            if is_long:
                if low[i] <= stop_price or high[i] >= tp_price:
                    exit_signal = True
                elif not np.isnan(trail_price) and low[i] <= trail_price:
                    exit_signal = True
            else:
                if high[i] >= stop_price or low[i] <= tp_price:
                    exit_signal = True
                elif not np.isnan(trail_price) and high[i] >= trail_price:
                    exit_signal = True
            
            # Session end flat
            if flat_at_end and session_just_ended[i]:
                exit_signal = True
        
        # Process exit
        if exit_signal and position != 0:
            position = 0
            last_exit_bar = i
            entry_price = np.nan
            entry_atr = np.nan
            highest_since_entry = np.nan
            lowest_since_entry = np.nan
            signals[i] = 0
            continue
        
        # Process entry
        if long_signal:
            position = 1
            entry_price = close[i]
            entry_atr = atr_val[i] if not np.isnan(atr_val[i]) else 1.0
            entry_mode = mode
            highest_since_entry = high[i]
            signals[i] = 1
        elif short_signal:
            position = -1
            entry_price = close[i]
            entry_atr = atr_val[i] if not np.isnan(atr_val[i]) else 1.0
            entry_mode = mode
            lowest_since_entry = low[i]
            signals[i] = -1
        else:
            signals[i] = position
    
    return signals

if __name__ == "__main__":
    # Example usage
    print(f"Strategy: {name}")
    print(f"Timeframe: {timeframe}")
    print(f"Leverage: {leverage}")
