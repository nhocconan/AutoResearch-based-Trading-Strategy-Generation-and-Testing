"""
ORB + Key Session Levels Strategy
Converted from TradingView Pine Script
Core ORB breakout logic only - HTF bias and visual features omitted
"""

import numpy as np
import pandas as pd
from datetime import time

name = "orb-key-session-levels"
timeframe = "5m"
leverage = 1

def _parse_time_string(time_str):
    """Parse HHMM string to time object"""
    hour = int(time_str[:2])
    minute = int(time_str[2:4])
    return time(hour, minute)

def _is_in_session(dt, start_time, end_time):
    """Check if datetime is within session window (handles overnight sessions)"""
    dt_time = dt.time()
    
    if start_time <= end_time:
        return start_time <= dt_time <= end_time
    else:
        return dt_time >= start_time or dt_time <= end_time

def _get_date(dt):
    """Extract date from datetime-like object (handles numpy.datetime64 and pandas Timestamp)"""
    if hasattr(dt, 'date'):
        return dt.date()
    elif isinstance(dt, np.datetime64):
        return pd.Timestamp(dt).date()
    else:
        return pd.Timestamp(dt).date()

def _calculate_orb_levels(open_arr, high_arr, low_arr, close_arr, open_time, 
                          orb_start_time, orb_end_time):
    """
    Calculate ORB high, low, mid for each day
    Returns arrays of orb_high, orb_low, orb_mid, orb_complete flag
    """
    n = len(open_arr)
    orb_high = np.full(n, np.nan)
    orb_low = np.full(n, np.nan)
    orb_mid = np.full(n, np.nan)
    orb_complete = np.zeros(n, dtype=bool)
    
    current_day = None
    day_orb_high = np.nan
    day_orb_low = np.nan
    day_orb_complete = False
    
    for i in range(n):
        dt = open_time[i]
        day = _get_date(dt)
        
        if day != current_day:
            current_day = day
            day_orb_high = np.nan
            day_orb_low = np.nan
            day_orb_complete = False
        
        in_session = _is_in_session(dt, orb_start_time, orb_end_time)
        
        if in_session and not day_orb_complete:
            if np.isnan(day_orb_high) or high_arr[i] > day_orb_high:
                day_orb_high = high_arr[i]
            if np.isnan(day_orb_low) or low_arr[i] < day_orb_low:
                day_orb_low = low_arr[i]
        elif not in_session and not np.isnan(day_orb_high) and not day_orb_complete:
            day_orb_complete = True
            orb_high[i] = day_orb_high
            orb_low[i] = day_orb_low
            orb_mid[i] = (day_orb_high + day_orb_low) / 2
            orb_complete[i] = True
        
        if day_orb_complete:
            orb_high[i] = day_orb_high
            orb_low[i] = day_orb_low
            orb_mid[i] = (day_orb_high + day_orb_low) / 2
            orb_complete[i] = True
    
    return orb_high, orb_low, orb_mid, orb_complete

def generate_signals(prices):
    """
    Generate ORB breakout signals
    
    Args:
        prices: pandas DataFrame with columns:
            - open_time: datetime-like (timezone-aware preferred)
            - open, high, low, close, volume: float
    
    Returns:
        numpy.ndarray of position intent:
            1 = long, -1 = short, 0 = flat
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    if n < 10:
        return signals
    
    open_arr = prices['open'].values
    high_arr = prices['high'].values
    low_arr = prices['low'].values
    close_arr = prices['close'].values
    open_time = prices['open_time'].values
    
    orb_start_time = _parse_time_string("0930")
    orb_end_time = _parse_time_string("0945")
    
    orb_high, orb_low, orb_mid, orb_complete = _calculate_orb_levels(
        open_arr, high_arr, low_arr, close_arr, open_time,
        orb_start_time, orb_end_time
    )
    
    entry_mode = "Breakout"
    sl_method = "Midpoint"
    tp1_rr = 1.0
    tp2_rr = 2.0
    tp3_rr = 3.0
    
    position = 0
    entry_price = np.nan
    sl_price = np.nan
    tp1_price = np.nan
    tp2_price = np.nan
    tp3_price = np.nan
    signal_fired = False
    orb_breakout_pending = False
    orb_breakout_dir = 0
    orb_breakout_bar = -1
    retest_timeout = 20
    
    for i in range(n):
        if position != 0:
            if position == 1:
                if low_arr[i] <= sl_price:
                    position = 0
                    signal_fired = False
                elif high_arr[i] >= tp3_price and tp3_price > 0:
                    position = 0
                    signal_fired = False
            else:
                if high_arr[i] >= sl_price:
                    position = 0
                    signal_fired = False
                elif low_arr[i] <= tp3_price and tp3_price > 0:
                    position = 0
                    signal_fired = False
        
        if position == 0 and signal_fired:
            signal_fired = False
            entry_price = np.nan
            sl_price = np.nan
            tp1_price = np.nan
            tp2_price = np.nan
            tp3_price = np.nan
            orb_breakout_pending = False
            orb_breakout_dir = 0
        
        if position == 0 and not signal_fired and orb_complete[i]:
            current_orb_high = orb_high[i]
            current_orb_low = orb_low[i]
            current_orb_mid = orb_mid[i]
            
            if np.isnan(current_orb_high) or np.isnan(current_orb_low):
                continue
            
            prev_close = close_arr[i-1] if i > 0 else close_arr[i]
            orb_high_break = close_arr[i] > current_orb_high and prev_close <= current_orb_high
            orb_low_break = close_arr[i] < current_orb_low and prev_close >= current_orb_low
            
            if entry_mode == "Breakout":
                if orb_high_break:
                    position = 1
                    entry_price = close_arr[i]
                    signal_fired = True
                    
                    if sl_method == "Midpoint":
                        sl_price = current_orb_mid
                    elif sl_method == "Opposite Side":
                        sl_price = current_orb_low
                    else:
                        sl_price = entry_price - 20.0
                    
                    risk = abs(entry_price - sl_price)
                    if risk > 0:
                        tp1_price = entry_price + risk * tp1_rr
                        tp2_price = entry_price + risk * tp2_rr
                        tp3_price = entry_price + risk * tp3_rr
                
                elif orb_low_break:
                    position = -1
                    entry_price = close_arr[i]
                    signal_fired = True
                    
                    if sl_method == "Midpoint":
                        sl_price = current_orb_mid
                    elif sl_method == "Opposite Side":
                        sl_price = current_orb_high
                    else:
                        sl_price = entry_price + 20.0
                    
                    risk = abs(sl_price - entry_price)
                    if risk > 0:
                        tp1_price = entry_price - risk * tp1_rr
                        tp2_price = entry_price - risk * tp2_rr
                        tp3_price = entry_price - risk * tp3_rr
            
            elif entry_mode in ["Retest Zone", "Retest Midpoint"]:
                if orb_high_break and not orb_breakout_pending:
                    orb_breakout_pending = True
                    orb_breakout_dir = 1
                    orb_breakout_bar = i
                elif orb_low_break and not orb_breakout_pending:
                    orb_breakout_pending = True
                    orb_breakout_dir = -1
                    orb_breakout_bar = i
                
                if orb_breakout_pending and i > orb_breakout_bar:
                    retest_lvl = current_orb_mid if entry_mode == "Retest Midpoint" else (
                        current_orb_high if orb_breakout_dir == 1 else current_orb_low
                    )
                    
                    if orb_breakout_dir == 1:
                        if low_arr[i] <= retest_lvl and close_arr[i] > current_orb_high:
                            position = 1
                            entry_price = close_arr[i]
                            signal_fired = True
                            orb_breakout_pending = False
                            
                            if sl_method == "Midpoint":
                                sl_price = current_orb_mid
                            elif sl_method == "Opposite Side":
                                sl_price = current_orb_low
                            else:
                                sl_price = entry_price - 20.0
                            
                            risk = abs(entry_price - sl_price)
                            if risk > 0:
                                tp1_price = entry_price + risk * tp1_rr
                                tp2_price = entry_price + risk * tp2_rr
                                tp3_price = entry_price + risk * tp3_rr
                        elif close_arr[i] <= current_orb_high:
                            orb_breakout_pending = False
                    
                    elif orb_breakout_dir == -1:
                        if high_arr[i] >= retest_lvl and close_arr[i] < current_orb_low:
                            position = -1
                            entry_price = close_arr[i]
                            signal_fired = True
                            orb_breakout_pending = False
                            
                            if sl_method == "Midpoint":
                                sl_price = current_orb_mid
                            elif sl_method == "Opposite Side":
                                sl_price = current_orb_high
                            else:
                                sl_price = entry_price + 20.0
                            
                            risk = abs(sl_price - entry_price)
                            if risk > 0:
                                tp1_price = entry_price - risk * tp1_rr
                                tp2_price = entry_price - risk * tp2_rr
                                tp3_price = entry_price - risk * tp3_rr
                        elif close_arr[i] >= current_orb_low:
                            orb_breakout_pending = False
                    
                    if orb_breakout_pending and (i - orb_breakout_bar) >= retest_timeout:
                        orb_breakout_pending = False
        
        signals[i] = position
    
    return signals
