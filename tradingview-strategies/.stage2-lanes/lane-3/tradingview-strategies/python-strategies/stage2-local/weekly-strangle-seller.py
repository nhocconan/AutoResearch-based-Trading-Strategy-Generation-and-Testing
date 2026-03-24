name = "weekly-strangle-seller"
timeframe = "4h"

import numpy as np
import pandas as pd
from datetime import datetime, time


def _atr(high, low, close, length):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr


def _supertrend(high, low, close, multiplier, length):
    """Calculate SuperTrend line and direction."""
    atr = _atr(high, low, close, length)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    st = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    prev_st = lower.iloc[0]
    prev_dir = 1
    
    for i in range(len(close)):
        if i == 0:
            st.iloc[i] = lower.iloc[i]
            direction.iloc[i] = 1
            continue
        
        if close.iloc[i] > prev_st:
            st.iloc[i] = lower.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] < prev_st:
            st.iloc[i] = upper.iloc[i]
            direction.iloc[i] = -1
        else:
            st.iloc[i] = prev_st
            direction.iloc[i] = prev_dir
        
        prev_st = st.iloc[i]
        prev_dir = direction.iloc[i]
    
    return st, direction


def _rsi(close, length):
    """Calculate RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _vwap(high, low, close, volume):
    """Calculate VWAP."""
    typical = (high + low + close) / 3
    vwap = (typical * volume).cumsum() / volume.cumsum()
    return vwap


def _is_in_time_window(dt, window_str, tz_offset_hours=5.5):
    """
    Check if datetime is within time window (IST assumed).
    window_str format: "HHMM-HHMM" e.g., "1000-1400"
    tz_offset_hours: IST is UTC+5:30
    """
    if not isinstance(dt, (datetime, pd.Timestamp)):
        return False
    
    try:
        parts = window_str.split("-")
        start_str, end_str = parts[0], parts[1]
        start_h, start_m = int(start_str[:2]), int(start_str[2:])
        end_h, end_m = int(end_str[:2]), int(end_str[2:])
        
        start_time = time(start_h, start_m)
        end_time = time(end_h, end_m)
        
        check_time = dt.time()
        return start_time <= check_time <= end_time
    except:
        return False


def _get_day_name(dt):
    """Get day name from datetime."""
    if not isinstance(dt, (datetime, pd.Timestamp)):
        return None
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days[dt.weekday()]


def generate_signals(prices):
    """
    Generate position signals for Weekly Strangle Seller strategy.
    
    Args:
        prices: pandas.DataFrame with columns:
            - open_time: datetime-like
            - open, high, low, close, volume: numeric
    
    Returns:
        numpy.ndarray of position signals (-1, 0, 1) with len(prices)
        Note: Strangle sell = 0 (neutral/flat), as we're selling volatility
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    if n < 50:
        return signals
    
    df = prices.copy()
    
    # Inputs (from Pine defaults)
    atr_len = 14
    atr_mult = 1.8
    s_round = 100
    use_vwap = True
    
    mon_entry = True
    tue_entry = True
    wed_entry = False
    entry_time = "1000-1400"
    
    use_st = True
    st_len = 10
    st_mul = 2.0
    use_rsi = True
    rsi_len = 14
    rsi_ob = 70
    rsi_os = 30
    min_width = 3.0
    
    sl_mult = 2.0
    exit_day = "Thursday"
    exit_time = "1515"
    
    # Calculate indicators
    df['atr'] = _atr(df['high'], df['low'], df['close'], atr_len)
    df['vwap'] = _vwap(df['high'], df['low'], df['close'], df['volume'])
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['center'] = df['vwap'] if use_vwap else df['ema20']
    
    df['raw_ce'] = df['center'] + (df['atr'] * atr_mult)
    df['raw_pe'] = df['center'] - (df['atr'] * atr_mult)
    df['ce_level'] = np.ceil(df['raw_ce'] / s_round) * s_round
    df['pe_level'] = np.floor(df['raw_pe'] / s_round) * s_round
    df['width_pct'] = (df['ce_level'] - df['pe_level']) / df['close'] * 100
    
    # SuperTrend
    st_line, st_dir = _supertrend(df['high'], df['low'], df['close'], st_mul, st_len)
    df['st_dir'] = st_dir
    df['is_bull'] = st_dir < 0
    df['is_bear'] = st_dir > 0
    
    # RSI
    df['rsi'] = _rsi(df['close'], rsi_len)
    df['rsi_bad'] = use_rsi & ((df['rsi'] > rsi_ob) | (df['rsi'] < rsi_os))
    
    # Day of week
    df['day_name'] = df['open_time'].apply(_get_day_name)
    df['is_mon'] = df['day_name'] == "Monday"
    df['is_tue'] = df['day_name'] == "Tuesday"
    df['is_wed'] = df['day_name'] == "Wednesday"
    df['is_thu'] = df['day_name'] == "Thursday"
    df['is_fri'] = df['day_name'] == "Friday"
    
    # Entry day check
    df['is_entry_day'] = False
    if mon_entry:
        df['is_entry_day'] = df['is_entry_day'] | df['is_mon']
    if tue_entry:
        df['is_entry_day'] = df['is_entry_day'] | df['is_tue']
    if wed_entry:
        df['is_entry_day'] = df['is_entry_day'] | df['is_wed']
    
    # Expiry day check
    df['is_exp_day'] = False
    if exit_day == "Thursday":
        df['is_exp_day'] = df['is_thu']
    elif exit_day == "Wednesday":
        df['is_exp_day'] = df['is_wed']
    elif exit_day == "Friday":
        df['is_exp_day'] = df['is_fri']
    
    # Time windows
    df['in_entry_window'] = df['open_time'].apply(
        lambda x: _is_in_time_window(x, entry_time)
    )
    df['at_expiry'] = df['is_exp_day'] & df['open_time'].apply(
        lambda x: _is_in_time_window(x, f"{exit_time}-1530")
    )
    
    # Filters
    df['width_ok'] = df['width_pct'] >= min_width
    df['trend_ok'] = ~use_st | (~df['is_bull'] & ~df['is_bear']) | (df['width_pct'] >= 4.0)
    df['rsi_ok'] = ~df['rsi_bad']
    df['filter_pass'] = df['width_ok'] & df['trend_ok'] & df['rsi_ok']
    
    # State tracking (weekly)
    week_taken = False
    in_trade = False
    entry_ce = 0.0
    entry_pe = 0.0
    sl_ce = 0.0
    sl_pe = 0.0
    prev_week = None
    
    for i in range(n):
        current_week = df['open_time'].iloc[i].isocalendar()[1] if hasattr(df['open_time'].iloc[i], 'isocalendar') else None
        
        # Reset on new week (Monday)
        if df['is_mon'].iloc[i] and (prev_week is None or current_week != prev_week):
            week_taken = False
        prev_week = current_week
        
        # Entry conditions
        can_enter = (df['is_entry_day'].iloc[i] and 
                     df['in_entry_window'].iloc[i] and 
                     not week_taken and 
                     df['filter_pass'].iloc[i])
        
        # Calculate SL levels when entering
        if can_enter and not in_trade:
            center = df['center'].iloc[i]
            ce_level = df['ce_level'].iloc[i]
            pe_level = df['pe_level'].iloc[i]
            
            entry_ce = ce_level
            entry_pe = pe_level
            sl_ce = ce_level + (ce_level - center) * (sl_mult - 1)
            sl_pe = pe_level - (center - pe_level) * (sl_mult - 1)
            in_trade = True
            week_taken = True
            
            # Signal: strangle sell = neutral position (0)
            # We mark entry with a special flag in a separate array if needed
            signals[i] = 0
        
        # Exit conditions
        ce_sl_hit = in_trade and df['high'].iloc[i] >= sl_ce
        pe_sl_hit = in_trade and df['low'].iloc[i] <= sl_pe
        exp_exit = in_trade and df['at_expiry'].iloc[i]
        fri_safe = in_trade and df['is_fri'].iloc[i] and df['open_time'].iloc[i].hour >= 15 and df['open_time'].iloc[i].minute >= 20
        
        if ce_sl_hit or pe_sl_hit or exp_exit or fri_safe:
            in_trade = False
            signals[i] = 0
        
        # While in trade, maintain neutral signal
        if in_trade:
            signals[i] = 0
    
    return signals
