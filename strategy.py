# 6h_TimeOfDay_VolatilityBreakout - Exploit overnight/early morning volatility in BTC/ETH
# Uses time-of-day filter (UTC 0-6) + volatility breakout from 1d ATR range
# Works in both bull/bear by capturing breakout moves during low-liquidity periods
# Target: 50-150 total trades over 4 years = 12-37/year

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation (needed for volatility breakout)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-day ATR for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Previous day's close for breakout levels
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First period uses current close
    prev_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # Time-of-day filter: UTC 0-6 (overnight/early morning low liquidity)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    time_filter = (hours >= 0) & (hours <= 6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Wait for ATR calculation
    
    for i in range(start_idx, n):
        # Skip if required data is NaN
        if np.isnan(atr_1d_aligned[i]) or np.isnan(prev_close_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Only trade during low-liquidity hours (0-6 UTC)
        if not time_filter[i]:
            # Hold position or flat outside trading hours
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility breakout: price moves beyond 0.5 * ATR from previous close
        breakout_threshold = 0.5 * atr_1d_aligned[i]
        
        # Long breakout: price > prev_close + threshold
        long_breakout = close[i] > prev_close_1d_aligned[i] + breakout_threshold
        
        # Short breakout: price < prev_close - threshold
        short_breakout = close[i] < prev_close_1d_aligned[i] - breakout_threshold
        
        # Exit when price returns to previous close (mean reversion)
        long_exit = close[i] < prev_close_1d_aligned[i] and position == 1
        short_exit = close[i] > prev_close_1d_aligned[i] and position == -1
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_TimeOfDay_VolatilityBreakout"
timeframe = "6h"
leverage = 1.0