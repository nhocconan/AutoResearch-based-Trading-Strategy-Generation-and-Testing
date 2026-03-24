name = "superbreak-elite-orb-gold"
timeframe = "5m"
leverage = 1

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Generate position signals for SuperBreak Elite ORB Gold strategy.
    
    Returns:
        numpy.ndarray: Position intent (-1=short, 0=flat, 1=long) for each bar
    
    Limitations:
        - External bias (DXY/VIX/US10Y) not available - forced signals use EMA trend only
        - Session times approximated for CET (UTC+1)
        - No actual order execution - signals indicate position intent only
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    if n < 200:
        return signals
    
    # Extract OHLCV
    open_time = prices['open_time'].values
    open_p = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Parameters (from Pine inputs)
    tp_points = 1500
    sl_multiplier = 3.0
    use_trend_filter = True
    use_bb_filter = False
    bb_length = 20
    bb_std_dev = 2.0
    min_bb_width = 1.0
    orb_minutes = 15
    
    # Session times (CET/UTC+1)
    london_start_hour = 9
    london_end_hour = 10
    ny_start_hour = 14
    ny_end_hour = 15
    ny_start_minute = 30
    ny_end_minute = 30
    
    # Calculate ORB bars (assuming 5m timeframe)
    tf_minutes = 5
    orb_bars = max(1, int(orb_minutes / tf_minutes))
    session_bars = max(1, int(60 / tf_minutes))
    
    # Calculate EMAs
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    
    # Calculate Bollinger Bands
    bb_middle = _sma(close, bb_length)
    bb_std = _stdev(close, bb_length)
    bb_upper = bb_middle + bb_std_dev * bb_std
    bb_lower = bb_middle - bb_std_dev * bb_std
    bb_width = np.where(bb_middle != 0, ((bb_upper - bb_lower) / bb_middle) * 100, 0)
    
    # Trend filters
    bull_trend = ema50 > ema200
    bear_trend = ema50 < ema200
    
    # Volatility filter
    vol_ok = np.ones(n, dtype=bool)
    if use_bb_filter:
        vol_ok = bb_width > min_bb_width
    
    # Track position state
    position = 0  # 0=flat, 1=long, -1=short
    london_signal_given = False
    ny_signal_given = False
    
    # ORB tracking variables
    london_orb_high = np.nan
    london_orb_low = np.nan
    london_bar_count = 0
    ny_orb_high = np.nan
    ny_orb_low = np.nan
    ny_bar_count = 0
    
    # Track session state
    in_london_prev = False
    in_ny_prev = False
    
    for i in range(n):
        dt = open_time[i]
        if isinstance(dt, (int, np.integer)):
            dt = pd.Timestamp(dt, unit='ms')
        elif not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        # Convert to CET (UTC+1) for session detection
        try:
            if dt.tzinfo is None:
                dt_cet = dt + timedelta(hours=1)
            else:
                dt_cet = dt.tz_convert('CET') if dt.tzinfo else dt + timedelta(hours=1)
        except:
            dt_cet = dt + timedelta(hours=1)
        
        hour = dt_cet.hour
        minute = dt_cet.minute
        
        # Detect London session (09:00-10:00 CET)
        in_london = (hour == london_start_hour and minute >= 0) or \
                    (hour == london_end_hour and minute == 0) or \
                    (hour > london_start_hour and hour < london_end_hour)
        
        # Detect NY session (14:30-15:30 CET)
        in_ny = (hour == ny_start_hour and minute >= ny_start_minute) or \
                (hour == ny_end_hour and minute <= ny_end_minute) or \
                (hour > ny_start_hour and hour < ny_end_hour)
        
        # New session detection
        new_london_bar = in_london and not in_london_prev
        new_ny_bar = in_ny and not in_ny_prev
        
        # Reset ORB on new session
        if new_london_bar:
            london_bar_count = 1
            london_orb_high = high[i]
            london_orb_low = low[i]
            london_signal_given = False
        elif in_london and london_bar_count > 0:
            london_bar_count += 1
            if london_bar_count <= orb_bars:
                london_orb_high = max(london_orb_high, high[i])
                london_orb_low = min(london_orb_low, low[i])
        
        if new_ny_bar:
            ny_bar_count = 1
            ny_orb_high = high[i]
            ny_orb_low = low[i]
            ny_signal_given = False
        elif in_ny and ny_bar_count > 0:
            ny_bar_count += 1
            if ny_bar_count <= orb_bars:
                ny_orb_high = max(ny_orb_high, high[i])
                ny_orb_low = min(ny_orb_low, low[i])
        
        # Check for end-of-session forced signal (last 5 minutes)
        london_end_phase = in_london and hour == london_end_hour and minute >= 55
        ny_end_phase = in_ny and hour == ny_end_hour and minute >= 55
        
        # Trading signals
        buy_signal = False
        sell_signal = False
        
        # London ORB breakout
        if in_london and london_bar_count > orb_bars and not london_signal_given:
            if not np.isnan(london_orb_high) and not np.isnan(london_orb_low):
                if close[i] > london_orb_high and (not use_trend_filter or bull_trend[i]) and vol_ok[i]:
                    buy_signal = True
                    london_signal_given = True
                elif close[i] < london_orb_low and (not use_trend_filter or bear_trend[i]) and vol_ok[i]:
                    sell_signal = True
                    london_signal_given = True
        
        # London forced signal (end of session)
        if london_end_phase and not london_signal_given:
            if bull_trend[i]:
                buy_signal = True
                london_signal_given = True
            elif bear_trend[i]:
                sell_signal = True
                london_signal_given = True
        
        # NY ORB breakout
        if in_ny and ny_bar_count > orb_bars and not ny_signal_given:
            if not np.isnan(ny_orb_high) and not np.isnan(ny_orb_low):
                if close[i] > ny_orb_high and (not use_trend_filter or bull_trend[i]) and vol_ok[i]:
                    buy_signal = True
                    ny_signal_given = True
                elif close[i] < ny_orb_low and (not use_trend_filter or bear_trend[i]) and vol_ok[i]:
                    sell_signal = True
                    ny_signal_given = True
        
        # NY forced signal (end of session)
        if ny_end_phase and not ny_signal_given:
            if bull_trend[i]:
                buy_signal = True
                ny_signal_given = True
            elif bear_trend[i]:
                sell_signal = True
                ny_signal_given = True
        
        # Position management (no same-bar fills - next bar execution)
        if i > 0:
            if buy_signal and position <= 0:
                position = 1
                signals[i] = 1
            elif sell_signal and position >= 0:
                position = -1
                signals[i] = -1
            
            # Check TP/SL exits (simplified - based on points from entry)
            if position == 1 and i > 0:
                entry_idx = np.where(signals[:i] == 1)[0]
                if len(entry_idx) > 0:
                    entry_price = close[entry_idx[-1]]
                    if high[i] >= entry_price + tp_points or low[i] <= entry_price - tp_points * sl_multiplier:
                        position = 0
                        signals[i] = 0
            
            if position == -1 and i > 0:
                entry_idx = np.where(signals[:i] == -1)[0]
                if len(entry_idx) > 0:
                    entry_price = close[entry_idx[-1]]
                    if low[i] <= entry_price - tp_points or high[i] >= entry_price + tp_points * sl_multiplier:
                        position = 0
                        signals[i] = 0
        
        in_london_prev = in_london
        in_ny_prev = in_ny
    
    return signals


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    result = np.zeros_like(data, dtype=np.float64)
    multiplier = 2 / (period + 1)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]
    return result


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average."""
    result = np.zeros_like(data, dtype=np.float64)
    for i in range(len(data)):
        if i < period - 1:
            result[i] = np.nan
        else:
            result[i] = np.mean(data[i-period+1:i+1])
    return result


def _stdev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Standard Deviation."""
    result = np.zeros_like(data, dtype=np.float64)
    for i in range(len(data)):
        if i < period - 1:
            result[i] = np.nan
        else:
            result[i] = np.std(data[i-period+1:i+1], ddof=0)
    return result