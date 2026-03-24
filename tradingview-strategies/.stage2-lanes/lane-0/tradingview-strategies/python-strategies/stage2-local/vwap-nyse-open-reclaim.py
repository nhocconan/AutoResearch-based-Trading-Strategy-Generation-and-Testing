"""
VWAP NYSE Open Reclaim + Pullback Hold Strategy
Converted from TradingView Pine Script

Limitations vs original:
- No multi-symbol panel (ES/NQ/YM) - single symbol only
- Session filtering approximated via datetime indexing
- Tick-based parameters require mintick input
- No intrabar calculation (calc_on_every_tick)
"""

import numpy as np
import pandas as pd
from typing import Optional

name = "VWAP NYSE Open Reclaim"
timeframe = "5m"
leverage = 1

def _compute_vwap_session(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    session_mask: np.ndarray,
    session_start_mask: np.ndarray
) -> np.ndarray:
    """Compute session-anchored VWAP with reset on session start."""
    n = len(close)
    vwap = np.full(n, np.nan)
    cum_pv = 0.0
    cum_v = 0.0
    
    for i in range(n):
        if session_start_mask[i]:
            cum_pv = ((high[i] + low[i] + close[i]) / 3.0) * volume[i]
            cum_v = volume[i]
        elif session_mask[i]:
            cum_pv += ((high[i] + low[i] + close[i]) / 3.0) * volume[i]
            cum_v += volume[i]
        
        if cum_v > 0 and session_mask[i]:
            vwap[i] = cum_pv / cum_v
    
    return vwap

def _compute_ema(series: np.ndarray, length: int) -> np.ndarray:
    """Compute exponential moving average."""
    n = len(series)
    ema = np.full(n, np.nan)
    multiplier = 2.0 / (length + 1)
    
    first_valid = 0
    for i in range(n):
        if not np.isnan(series[i]):
            first_valid = i
            break
    
    if first_valid < n:
        ema[first_valid] = series[first_valid]
        for i in range(first_valid + 1, n):
            if not np.isnan(series[i]):
                ema[i] = (series[i] - ema[i-1]) * multiplier + ema[i-1]
            else:
                ema[i] = ema[i-1]
    
    return ema

def _detect_session(
    open_time: pd.Series,
    start_hour: int,
    start_min: int,
    end_hour: int,
    end_min: int
) -> tuple:
    """Detect session bars and session start bars."""
    n = len(open_time)
    in_session = np.zeros(n, dtype=bool)
    session_start = np.zeros(n, dtype=bool)
    
    for i in range(n):
        dt = open_time.iloc[i]
        hour, minute = dt.hour, dt.minute
        
        # Handle sessions that cross midnight
        if start_hour < end_hour:
            in_sess = (hour > start_hour or (hour == start_hour and minute >= start_min)) and \
                      (hour < end_hour or (hour == end_hour and minute <= end_min))
        else:
            in_sess = (hour >= start_hour or hour <= end_hour)
        
        in_session[i] = in_sess
        
        if i > 0:
            session_start[i] = in_sess and not in_session[i-1]
        else:
            session_start[i] = in_sess
    
    return in_session, session_start

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Generate position signals for VWAP NYSE Open Reclaim strategy.
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume]
    
    Returns:
        numpy.ndarray of position signals: 1 (long), 0 (flat), -1 (short)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    # Extract columns
    open_time = prices['open_time']
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Parameters (from Pine Script defaults)
    ema_fast_len = 5
    ema_mid_len = 8
    ema_slow_len = 13
    touch_buffer_ticks = 2
    stop_buffer_ticks = 2
    max_stop_ticks = 40
    take_profit_ticks = 40
    max_vwap_crosses = 4
    reclaim_mode = "1-close+hold"  # or "2-closes"
    use_vol_confirm = True
    
    # NOTE: mintick must be provided per symbol (ES=0.25, NQ=0.25, YM=1.0, etc.)
    mintick = 0.25  # Default for ES futures
    tick = mintick
    buf = touch_buffer_ticks * tick
    stop_buf = stop_buffer_ticks * tick
    
    # Session detection (NYSE RTH: 09:30-16:00 ET)
    # Trade window: 09:45-11:00 ET
    in_rth, is_new_rth = _detect_session(open_time, 9, 30, 16, 0)
    in_trade, _ = _detect_session(open_time, 9, 45, 11, 0)
    in_18to16, is_new_18to16 = _detect_session(open_time, 18, 0, 16, 0)  # Handles midnight cross
    
    # Opening Range: 09:30-09:45 ET
    in_or, is_new_or = _detect_session(open_time, 9, 30, 9, 45)
    
    # Compute VWAPs
    open_vwap = _compute_vwap_session(close, high, low, volume, in_rth, is_new_rth)
    vwap_18to16 = _compute_vwap_session(close, high, low, volume, in_18to16, is_new_18to16)
    
    # Compute EMAs
    ema5 = _compute_ema(close, ema_fast_len)
    ema8 = _compute_ema(close, ema_mid_len)
    ema13 = _compute_ema(close, ema_slow_len)
    
    # Track VWAP crosses (price vs NYSE Open VWAP)
    vwap_cross_count = 0
    vwap_cross_blocked = np.zeros(n, dtype=bool)
    
    # Track state variables
    reclaimed_long = False
    reclaimed_short = False
    below_count = 0
    above_count = 0
    pending_long = False
    pending_short = False
    long_setup_high = np.nan
    long_setup_low = np.nan
    short_setup_high = np.nan
    short_setup_low = np.nan
    
    for i in range(n):
        # Reset on new RTH session
        if is_new_rth[i]:
            vwap_cross_count = 0
            reclaimed_long = False
            reclaimed_short = False
            below_count = 0
            above_count = 0
            pending_long = False
            pending_short = False
            long_setup_high = np.nan
            long_setup_low = np.nan
            short_setup_high = np.nan
            short_setup_low = np.nan
        
        # Count VWAP crosses
        if i > 0 and in_rth[i] and not is_new_rth[i]:
            if not np.isnan(open_vwap[i]) and not np.isnan(open_vwap[i-1]):
                cross_up = close[i-1] <= open_vwap[i-1] and close[i] > open_vwap[i]
                cross_down = close[i-1] >= open_vwap[i-1] and close[i] < open_vwap[i]
                if cross_up or cross_down:
                    vwap_cross_count += 1
        
        allow_new_trades = vwap_cross_count < max_vwap_crosses
        
        # VWAP acceptance logic
        close_above_vwap = in_rth[i] and not np.isnan(open_vwap[i]) and close[i] > open_vwap[i]
        close_below_vwap = in_rth[i] and not np.isnan(open_vwap[i]) and close[i] < open_vwap[i]
        
        if close_below_vwap:
            below_count += 1
        else:
            below_count = 0
        
        if close_above_vwap:
            above_count += 1
        else:
            above_count = 0
        
        accept_below_vwap = below_count >= 2
        accept_above_vwap = above_count >= 2
        
        # Reclaim logic
        if accept_below_vwap:
            reclaimed_long = False
        elif i > 0 and close_above_vwap:
            if reclaim_mode == "2-closes":
                reclaimed_long = close_above_vwap and (i > 0 and close[i-1] > open_vwap[i-1] if not np.isnan(open_vwap[i-1]) else False)
            else:
                prev_at_or_below = i > 0 and (close[i-1] <= open_vwap[i-1] if not np.isnan(open_vwap[i-1]) else True)
                reclaimed_long = close_above_vwap and (prev_at_or_below or reclaimed_long)
        
        if accept_above_vwap:
            reclaimed_short = False
        elif i > 0 and close_below_vwap:
            if reclaim_mode == "2-closes":
                reclaimed_short = close_below_vwap and (i > 0 and close[i-1] < open_vwap[i-1] if not np.isnan(open_vwap[i-1]) else False)
            else:
                prev_at_or_above = i > 0 and (close[i-1] >= open_vwap[i-1] if not np.isnan(open_vwap[i-1]) else True)
                reclaimed_short = close_below_vwap and (prev_at_or_above or reclaimed_short)
        
        # EMA confluence
        ema_long_ok = (not np.isnan(ema5[i]) and not np.isnan(ema8[i]) and not np.isnan(ema13[i]) and
                       ema5[i] > ema8[i] and ema8[i] > ema13[i] and
                       i > 0 and ema5[i] > ema5[i-1] and ema8[i] > ema8[i-1] and ema13[i] > ema13[i-1])
        
        ema_short_ok = (not np.isnan(ema5[i]) and not np.isnan(ema8[i]) and not np.isnan(ema13[i]) and
                        ema5[i] < ema8[i] and ema8[i] < ema13[i] and
                        i > 0 and ema5[i] < ema5[i-1] and ema8[i] < ema8[i-1] and ema13[i] < ema13[i-1])
        
        # Volume confirmation
        vol_ok = not use_vol_confirm or (i > 0 and volume[i] > volume[i-1])
        
        # Touch VWAP detection
        touched_vwap_long = in_rth[i] and not np.isnan(open_vwap[i]) and (low[i] <= open_vwap[i] + buf)
        touched_vwap_short = in_rth[i] and not np.isnan(open_vwap[i]) and (high[i] >= open_vwap[i] - buf)
        
        # Hold candle detection
        hold_candle_long = (allow_new_trades and in_trade[i] and reclaimed_long and ema_long_ok and
                           touched_vwap_long and close[i] > open_vwap[i] and vol_ok)
        hold_candle_short = (allow_new_trades and in_trade[i] and reclaimed_short and ema_short_ok and
                            touched_vwap_short and close[i] < open_vwap[i] and vol_ok)
        
        # Update pending setups
        if hold_candle_long:
            pending_long = True
            long_setup_high = high[i]
            long_setup_low = low[i]
        
        if hold_candle_short:
            pending_short = True
            short_setup_high = high[i]
            short_setup_low = low[i]
        
        # Invalidate pending setups
        if pending_long and (accept_below_vwap or not ema_long_ok or not reclaimed_long or not in_trade[i] or not allow_new_trades):
            pending_long = False
            long_setup_high = np.nan
            long_setup_low = np.nan
        
        if pending_short and (accept_above_vwap or not ema_short_ok or not reclaimed_short or not in_trade[i] or not allow_new_trades):
            pending_short = False
            short_setup_high = np.nan
            short_setup_low = np.nan
        
        # Entry logic (stop orders converted to next-bar signals)
        long_entry_stop = long_setup_high + tick if not np.isnan(long_setup_high) else np.nan
        long_stop_price = long_setup_low - stop_buf if not np.isnan(long_setup_low) else np.nan
        short_entry_stop = short_setup_low - tick if not np.isnan(short_setup_low) else np.nan
        short_stop_price = short_setup_high + stop_buf if not np.isnan(short_setup_high) else np.nan
        
        # Risk check
        long_risk_ok = (pending_long and not np.isnan(long_stop_price) and
                       ((long_entry_stop - long_stop_price) / tick) <= max_stop_ticks and
                       long_entry_stop > long_stop_price)
        short_risk_ok = (pending_short and not np.isnan(short_stop_price) and
                        ((short_stop_price - short_entry_stop) / tick) <= max_stop_ticks and
                        short_stop_price > short_entry_stop)
        
        # Generate signals (position intent for next bar)
        if i < n - 1:  # No signal on last bar (would need next bar data)
            if pending_long and long_risk_ok and allow_new_trades and in_trade[i]:
                if high[i] >= long_entry_stop:  # Stop triggered
                    signals[i+1] = 1
            elif pending_short and short_risk_ok and allow_new_trades and in_trade[i]:
                if low[i] <= short_entry_stop:  # Stop triggered
                    signals[i+1] = -1
    
    return signals
